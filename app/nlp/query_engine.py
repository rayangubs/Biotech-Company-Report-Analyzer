"""Conversational NLP query engine for FSR records.

Upgrade from pure intent dispatch to a context-aware assistant that:
  - Maintains conversation context across turns (last intent, referenced IDs)
  - Analyses patterns across records and surfaces insights proactively
  - Proposes solutions, flags recurring issues, compares across visits
  - References related data to enrich every answer
  - Speaks in natural, flowing language rather than bullet-dump responses

All logic is rule-based and runs fully offline.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from .normalize import (
    clean_text, is_empty, load_lexicon, best_match,
    split_lines, tag_action_line, classify_remark, normalize_date,
    similarity_ratio as _ratio,
)
from ..store import load_records


def _field(rec: dict, key: str) -> str:
    d = rec.get("data") or {}
    return clean_text(str(d.get(key) or ""))


# Simple per-process memo: raw date string → parsed date (or None).
# This avoids re-running normalize_date() on the same string every time
# _date_of() is called inside a loop over all records.
_DATE_MEMO: dict[str, date | None] = {}


def _date_of(rec: dict) -> date | None:
    raw = _field(rec, "dateStart")
    if raw in _DATE_MEMO:
        return _DATE_MEMO[raw]
    result: date | None = None
    res = normalize_date(raw)
    if res.get("ok") and res.get("value"):
        try:
            result = date.fromisoformat(res["value"])
        except ValueError:
            pass
    _DATE_MEMO[raw] = result
    return result


# ---------------------------------------------------------------------------
# Similarity / matching
# ---------------------------------------------------------------------------

def _fuzzy_in(needle: str, haystack: str, threshold: float = 0.6) -> bool:
    """True if needle appears in or is similar to haystack.

    threshold is expressed as a 0-1 fraction; _ratio() returns 0-100 so we
    multiply accordingly.
    """
    n, h = needle.lower(), haystack.lower()
    if n in h:
        return True
    threshold_100 = threshold * 100.0
    return any(_ratio(n, w) >= threshold_100 for w in h.split())


# ---------------------------------------------------------------------------
# Stop words and tokeniser
# ---------------------------------------------------------------------------
_STOP = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","shall","should","may","might",
    "must","can","could","i","me","my","we","our","you","your","it","its",
    "this","that","these","those","and","or","but","if","in","on","at","to",
    "for","of","with","by","from","about","how","many","much","any","some",
    "all","no","not","what","which","who","when","where","why","show","me",
    "give","list","get","find","tell","please","report","reports","also",
    "then","there","here","just","like","know","want","need","look",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
_MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
    "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
    "aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12,
}
_JOB_ALIASES = {
    "preventive maintenance":"preventive maintenance","preventive":"preventive maintenance",
    "pm":"preventive maintenance","corrective maintenance":"corrective maintenance",
    "corrective":"corrective maintenance","repair":"corrective maintenance",
    "equipment handling":"equipment handling","handling":"equipment handling",
    "installation":"equipment handling","uninstallation":"equipment handling",
    "pull out":"equipment handling","pullout":"equipment handling",
    "result validation":"result validation","calibration":"result validation",
    "training":"updates/training","updates":"updates/training",
}
_STATUS_ALIASES = {
    "operational":"Operational","working":"Operational","functional":"Operational",
    "non-operational":"Non-Operational","nonoperational":"Non-Operational",
    "not working":"Non-Operational","broken":"Non-Operational","down":"Non-Operational",
    "monitoring":"For further monitoring","monitor":"For further monitoring",
    "for further monitoring":"For further monitoring","for parts":"For Parts",
}


def _extract_month_year(text: str) -> tuple[int | None, int | None]:
    t = text.lower()
    month = next((num for name, num in sorted(_MONTHS.items(), key=lambda x: -len(x[0]))
                  if re.search(r"\b" + name + r"\b", t)), None)
    ym = re.search(r"\b(202[0-9])\b", t)
    return month, int(ym.group(1)) if ym else None


def _extract_report_id(text: str) -> str | None:
    m = re.search(r"\b([A-Za-z]\s?\d{4,6})\b", text)
    return re.sub(r"\s+", "", m.group(1)).upper() if m else None


def parse_intent(question: str, context: dict | None = None) -> dict:
    """Detect intent, incorporating conversation context."""
    q = question.strip()
    qt = q.lower()
    toks = _tokens(qt)
    ctx = context or {}

    # --- Report lookup / summary -------------------------------------------
    report_id = _extract_report_id(q)
    # Also resolve "this report", "that one", "it" from context
    if not report_id and any(w in qt for w in ("this report","that report","it","this one","that one")):
        raw_ctx_id = ctx.get("last_report_id", "")
        # Normalise — strip spaces so A 15252 and A15252 both work
        report_id = re.sub(r"\s+", "", raw_ctx_id).upper() if raw_ctx_id else None
    if report_id and any(w in qt for w in
            ("summary","summarize","summarise","about","detail","describe","overview","what happened")):
        return {"intent": "summary", "params": {"report_id": report_id}}
    if report_id and any(w in qt for w in
            ("compare","vs","versus","similar","other visits","other reports","history")):
        return {"intent": "compare_equipment", "params": {"report_id": report_id}}
    # Causal / explain takes priority over plain lookup
    if report_id and (re.search(r"\bwhy\b|\bexplain\b|\bwhat caused\b|\bwhat happened\b|\breason\b", qt)):
        return {"intent": "explain", "params": {"report_id": report_id}}
    if report_id:
        return {"intent": "lookup", "params": {"report_id": report_id}}

    # --- Trend / analysis questions ----------------------------------------
    if any(w in qt for w in ("trend","pattern","recurring","repeat","frequent","most common",
                              "usually","typically","often","always","every time")):
        return {"intent": "analyse_trends", "params": {}}

    # --- Explain / causal (checked early so "why" beats status/suggest) ----
    # Must be checked before status/job/anything else that might match the
    # rest of the sentence (e.g. "why was it non-operational" contains "non-operational")
    _WHY_WORDS = ("why","reason what caused","what happened","explain","how come")
    if any(w in qt for w in _WHY_WORDS) or re.search(r"\bwhy\b|\bexplain\b", qt):
        rid = ctx.get("last_report_id","")
        if rid:
            return {"intent": "explain", "params": {"report_id": re.sub(r'\s+','',rid).upper()}}
        if report_id:
            return {"intent": "explain", "params": {"report_id": report_id}}
        return {"intent": "analyse_trends", "params": {}}

    # IMPROVEMENT: overdue / scheduling awareness
    if any(w in qt for w in ("overdue","due for","when was","last visit","last service",
                              "how long","interval","schedule","upcoming","next pm",
                              "not been serviced","hasn't been")):
        return {"intent": "overdue_pm", "params": {}}

    # --- Recommendations (pending/follow-up items — checked before suggest) ---
    if any(w in qt for w in ("recommend","follow","follow-up","followup","pending",
                              "action item","next step","due")):
        return {"intent": "search_recommendations", "params": {}}

    if any(w in qt for w in ("suggest","suggestion","advice","what should","what can",
                              "how to fix","what to do","what do you think","improve","concern",
                              "problem with","issue with","worry","risk")):
        return {"intent": "suggest", "params": {"keywords": toks}}

    if any(w in qt for w in ("compare","vs","versus","difference","between","which is better")):
        return {"intent": "compare", "params": {"keywords": toks}}

    # --- Date range --------------------------------------------------------
    month, year = _extract_month_year(qt)
    if any(w in qt for w in ("last month","this month","last year","this year","recent","latest")):
        today = date.today()
        if "last month" in qt:
            m = today.month-1 or 12
            y = today.year if today.month > 1 else today.year-1
            month, year = m, y
        elif "this month" in qt:
            month, year = today.month, today.year
        elif "last year" in qt:
            month, year = None, today.year-1
        elif "this year" in qt:
            month, year = None, today.year
    if month or year:
        return {"intent": "search_date", "params": {"month": month, "year": year}}

    # --- Spare parts -------------------------------------------------------
    if any(w in qt for w in ("part","parts","reagent","reagents","spare","replacement",
                              "consumable","consumables","cassette","gas","supplies","most parts",
                              "most consumables","uses the most")):
        return {"intent": "search_parts", "params": {}}

    # --- Status ------------------------------------------------------------
    for alias, canonical in sorted(_STATUS_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in qt:
            return {"intent": "search_status", "params": {"status": canonical}}

    # --- Job type ----------------------------------------------------------
    for alias, canonical in sorted(_JOB_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in qt:
            # count vs list — check both original qt and toks (how/many are stop words)
            is_count = (any(w in toks for w in ("count","total","number"))
                        or any(w in qt for w in ("how many","how much","total number")))
            intent = "count" if is_count else "search_job"
            return {"intent": intent, "params": {"job_type": canonical}}

    # --- Institution -------------------------------------------------------
    institutions = load_lexicon("institutions")
    for inst in institutions:
        if inst.lower() in qt:
            return {"intent": "search_institution", "params": {"institution": inst}}
    if institutions:
        cand, score = best_match(qt, institutions)
        if cand and score >= 70:
            return {"intent": "search_institution", "params": {"institution": cand}}

    # --- Equipment ---------------------------------------------------------
    equipment_list = load_lexicon("equipment")
    for eq in equipment_list:
        if eq.lower() in qt:
            return {"intent": "search_equipment", "params": {"equipment": eq}}
    if equipment_list:
        cand, score = best_match(qt, equipment_list)
        if cand and score >= 70:
            return {"intent": "search_equipment", "params": {"equipment": cand}}

    # --- Specific actions --------------------------------------------------
    action_words = ["check","clean","test","calibrate","replace","update","pump","optic",
                    "cable","sticker","spin","temperature","timer","surface","pack","barcode"]
    for aw in action_words:
        if aw in qt:
            return {"intent": "search_actions", "params": {"keyword": aw}}

    # --- Count / stats -----------------------------------------------------
    if any(w in qt for w in ("how many","total","count","number of","statistics","stats",
                              "overview","summary","breakdown","distribution","all reports")):
        return {"intent": "count", "params": {}}

    # --- Provider ----------------------------------------------------------
    provider_words = ["navarro","lacaste","migo","provider","technician","engineer",
                      "performed by","done by","service by"]
    for pw in provider_words:
        if pw in qt:
            # Extract only the name word, not the whole sentence
            m2 = re.search(r"(?:navarro|migo|lacaste)", qt)
            name = m2.group(0).strip() if m2 else pw
            return {"intent": "search_provider", "params": {"name": name}}

    # --- General fallback --------------------------------------------------
    # If context has a last institution or equipment, treat as follow-up
    if toks:
        if ctx.get("last_institution") and any(w in ("there","it","them","their","these") for w in toks):
            return {"intent": "search_institution",
                    "params": {"institution": ctx["last_institution"]}}
        if ctx.get("last_equipment") and any(w in ("it","them","this","that","machine") for w in toks):
            return {"intent": "search_equipment",
                    "params": {"equipment": ctx["last_equipment"]}}
        return {"intent": "search_general", "params": {"keywords": toks}}

    return {"intent": "unknown", "params": {}}


# ---------------------------------------------------------------------------
# Insight helpers — these cross-reference records to surface analysis
# ---------------------------------------------------------------------------

def _get_recurring_issues(records: list[dict]) -> list[tuple[str, int]]:
    """Find problems / actions that appear across multiple records."""
    action_counter: Counter = Counter()
    for rec in records:
        for ln in split_lines(_field(rec, "actionsTaken")):
            tags = tag_action_line(ln)
            for tag in tags:
                action_counter[tag] += 1
    return action_counter.most_common()


def _compute_visit_gaps(records: list[dict]) -> dict[str, int]:
    """IMPROVEMENT: Compute average days between visits per equipment."""
    eq_dates: dict[str, list[date]] = defaultdict(list)
    for r in records:
        d = _date_of(r)
        eq = _field(r, "equipment")
        if d and eq:
            eq_dates[eq].append(d)
    gaps: dict[str, int] = {}
    for eq, dates in eq_dates.items():
        if len(dates) < 2:
            continue
        dates_sorted = sorted(dates)
        diffs = [(dates_sorted[i+1] - dates_sorted[i]).days
                 for i in range(len(dates_sorted)-1)]
        gaps[eq] = round(sum(diffs) / len(diffs))
    return gaps


def _get_parts_per_equipment(records: list[dict]) -> dict[str, Counter]:
    """IMPROVEMENT: Parts/reagents used grouped by equipment."""
    eq_parts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        eq = _field(r, "equipment")
        parts = (r.get("data") or {}).get("spareParts") or []
        for p in parts:
            desc = (p.get("desc") or "").strip()
            if desc and eq:
                eq_parts[eq][desc] += 1
    return dict(eq_parts)


def _get_equipment_history(records: list[dict], equipment: str) -> list[dict]:
    return [r for r in records
            if _fuzzy_in(equipment, _field(r, "equipment"), 0.55)]


def _get_institution_summary(records: list[dict], institution: str) -> dict:
    hits = [r for r in records if _fuzzy_in(institution, _field(r, "institution"), 0.55)]
    eq_counter: Counter = Counter(_field(r, "equipment") for r in hits)
    status_counter: Counter = Counter(_field(r, "statusAfter") for r in hits)
    job_counter: Counter = Counter(_field(r, "jobType").split("–")[0].strip() for r in hits)
    recs_with_followup = [r for r in hits if _match_recommendations(r)]
    return {
        "hits": hits,
        "equipment": eq_counter,
        "status": status_counter,
        "jobs": job_counter,
        "followups": recs_with_followup,
    }


def _nonop_analysis(records: list[dict]) -> dict:
    """Analyse non-operational machines and what caused them."""
    nonop = [r for r in records
             if "non-operational" in _field(r, "statusAfter").lower()
             or "non operational" in _field(r, "statusAfter").lower()]
    causes: list[str] = []
    for r in nonop:
        c = _field(r, "causeOfProblem")
        p = _field(r, "reportedProblem")
        if not is_empty(c): causes.append(c)
        elif not is_empty(p): causes.append(p)
    return {"records": nonop, "causes": causes}


def _match_recommendations(rec: dict) -> bool:
    remarks = _field(rec, "remarks")
    sources = [p.strip() for p in re.split(r"[\n;]+", remarks) if p.strip()]
    ocr_rem = (rec.get("ocr_zones") or {}).get("remarks") or {}
    for ln in ocr_rem.get("lines") or []:
        t = str(ln.get("text","")).strip() if isinstance(ln,dict) else str(ln).strip()
        if t: sources.append(t)
    nlp = rec.get("nlp") or {}
    for ln_obj in (nlp.get("remarks") or {}).get("lines") or []:
        if isinstance(ln_obj,dict) and ln_obj.get("kind") == "recommendation":
            return True
    if any(classify_remark(ln) == "recommendation" for ln in sources):
        return True
    return bool(re.search(
        r"\b(replace|repair|schedule|monitor|order|request|check|update|upgrade|follow|recalibrate)\b",
        remarks, re.I))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _short(rec: dict) -> str:
    d = rec.get("data") or {}
    return (f"**{rec.get('id','?')}** — {d.get('institution','?')} | "
            f"{d.get('equipment','?')} | {d.get('dateStart','?')} | "
            f"Job: {d.get('jobType','?')} | Status: {d.get('statusAfter','?')}")


def _bullet_list(recs: list[dict], limit: int = 10) -> str:
    items = [f"• {_short(r)}" for r in recs[:limit]]
    if len(recs) > limit:
        items.append(f"  _…and {len(recs)-limit} more_")
    return "\n".join(items)


def _fmt_actions(rec: dict) -> str:
    lines = split_lines(_field(rec, "actionsTaken"))
    if not lines: return "No actions recorded."
    out = []
    for ln in lines:
        tags = tag_action_line(ln)
        badge = f" _{', '.join(tags)}_" if tags else ""
        out.append(f"  • {ln}{badge}")
    return "\n".join(out)


def _fmt_remarks(rec: dict) -> str:
    lines = split_lines(_field(rec, "remarks"))
    if not lines: return "No remarks recorded."
    out = []
    for ln in lines:
        kind = classify_remark(ln)
        badge = {"recommendation": " ⚠", "status": " ✓"}.get(kind, "")
        out.append(f"  • {ln}{badge}")
    return "\n".join(out)


def _reply(text: str, data: dict, new_ctx: dict | None = None) -> dict:
    return {"text": text, "data": data, "context_update": new_ctx or {}}


# ---------------------------------------------------------------------------
# Answer builders — conversational, analytical, cross-referencing
# ---------------------------------------------------------------------------

def _answer_count(records: list[dict], params: dict) -> dict:
    job = params.get("job_type")
    filtered = [r for r in records
                if (_field(r, "jobType").lower().find(job.lower()) >= 0)] if job else records
    total = len(filtered)
    if total == 0:
        return _reply(
            f"I don't have any {'*' + job + '* ' if job else ''}reports in the database yet. "
            f"Upload and confirm some field service reports to get started.",
            {"count": 0}
        )

    counter: Counter = Counter()
    for r in filtered:
        jt = _field(r, "jobType") or "Unknown"
        for cat in ("Preventive Maintenance","Corrective Maintenance",
                    "Equipment Handling","Result Validation","Updates/Training"):
            if cat.lower() in jt.lower():
                counter[cat] += 1; break
        else:
            counter[jt[:40]] += 1

    status_counter: Counter = Counter(_field(r,"statusAfter") or "Unknown" for r in filtered)
    inst_counter: Counter = Counter(_field(r,"institution") for r in filtered)
    nonop = status_counter.get("Non-Operational", 0)

    lines = [f"There are **{total}** confirmed report(s) in the database.\n"]
    lines.append("**By job type:**")
    for cat, n in counter.most_common():
        pct = round(n/total*100)
        lines.append(f"  • {cat}: {n} ({pct}%)")
    lines.append("\n**By outcome:**")
    for s, n in status_counter.most_common():
        lines.append(f"  • {s}: {n}")

    # Proactive insight
    if nonop > 0:
        nonop_recs = [r for r in filtered if "non-operational" in _field(r,"statusAfter").lower()]
        institutions = list({_field(r,"institution") for r in nonop_recs})
        lines.append(f"\n⚠ **{nonop} machine(s) were left non-operational** after service — "
                     f"at {', '.join(institutions[:3])}. "
                     f"You may want to review those reports and check for pending follow-ups.")

    recs_with_followup = [r for r in filtered if _match_recommendations(r)]
    if recs_with_followup:
        lines.append(f"\n📋 **{len(recs_with_followup)} report(s) have pending recommendations** "
                     f"that may need follow-up action.")

    top_inst = inst_counter.most_common(1)
    if top_inst:
        lines.append(f"\n🏥 Most-serviced institution: **{top_inst[0][0]}** ({top_inst[0][1]} visit(s)).")

    return _reply("\n".join(lines), {
        "count": total, "breakdown": dict(counter), "status": dict(status_counter)
    })


def _answer_lookup(records: list[dict], report_id: str) -> dict:
    hit = next((r for r in records
                if r.get("id","").replace(" ","").upper() == report_id.upper()), None)
    if not hit:
        return _reply(
            f"I couldn't find report **{report_id}**. Double-check the report number, "
            f"or try searching by institution or date.",
            {"found": False}
        )
    d = hit.get("data") or {}

    # Cross-reference: other visits to the same equipment
    same_eq = [r for r in records
               if _fuzzy_in(_field(hit,"equipment"), _field(r,"equipment"), 0.7)
               and r.get("id") != hit.get("id")]

    lines = [
        f"**Report {hit['id']}** — confirmed {hit.get('confirmed_at','?')[:10]}",
        f"",
        f"**{d.get('institution','?')}** | {d.get('address','?')}",
        f"**Equipment:** {d.get('equipment','?')}  SN: {d.get('serial','?')}",
        f"**Service date:** {d.get('dateStart','?')} → {d.get('dateEnd','?')}",
        f"**Job type:** {d.get('jobType','?')}",
        f"",
    ]

    if not is_empty(d.get("reportedProblem","")):
        lines.append(f"**Problem reported:** {d.get('reportedProblem','?')}")
    if not is_empty(d.get("causeOfProblem","")):
        lines.append(f"**Root cause:** {d.get('causeOfProblem','?')}")
    lines += ["", f"**Actions taken:**", _fmt_actions(hit), "", f"**Remarks:**", _fmt_remarks(hit), ""]
    lines.append(f"**Outcome:** {d.get('statusAfter','?')}  |  "
                 f"Provider: {d.get('serviceProvider','?')}  |  "
                 f"Time: {d.get('timeIn','?')} → {d.get('timeOut','?')}")

    # Proactive cross-reference
    if same_eq:
        nonop_hist = [r for r in same_eq if "non-operational" in _field(r,"statusAfter").lower()]
        lines.append(f"\n---\n📊 **{len(same_eq)} other visit(s)** on record for this equipment.")
        if nonop_hist:
            lines.append(f"  ⚠ It was left non-operational {len(nonop_hist)} time(s) — "
                         f"this may indicate a recurring issue worth escalating.")

    if _match_recommendations(hit):
        rems = split_lines(_field(hit, "remarks"))
        recs_found = [ln for ln in rems if classify_remark(ln) == "recommendation"]
        if recs_found:
            lines.append(f"\n📋 **Pending follow-up:** {'; '.join(recs_found[:2])}")

    return _reply("\n".join(lines), {"found": True, "id": hit["id"]},
                  new_ctx={"last_report_id": hit["id"],
                            "last_institution": _field(hit,"institution"),
                            "last_equipment": _field(hit,"equipment")})


def _answer_summary(records: list[dict], report_id: str) -> dict:
    hit = next((r for r in records
                if r.get("id","").replace(" ","").upper() == report_id.upper()), None)
    if not hit:
        return _reply(f"I couldn't find report **{report_id}**.", {"found": False})
    d = hit.get("data") or {}
    at_lines = split_lines(_field(hit,"actionsTaken"))
    rem_lines = split_lines(_field(hit,"remarks"))
    recs_found = [ln for ln in rem_lines if classify_remark(ln) == "recommendation"]
    status_lines = [ln for ln in rem_lines if classify_remark(ln) == "status"]

    action_tags = set()
    for ln in at_lines:
        action_tags.update(tag_action_line(ln))

    text = (
        f"Report **{hit['id']}** covers a **{d.get('jobType','?')}** at "
        f"**{d.get('institution','?')}** on {d.get('dateStart','?')}. "
        f"The technician serviced a **{d.get('equipment','?')}**, "
        f"performing {len(at_lines)} action(s)"
        + (f" — mainly {', '.join(sorted(action_tags))}" if action_tags else "") + ". "
        f"The machine was left **{d.get('statusAfter','?')}**."
    )
    if status_lines:
        text += f" Status note: _{status_lines[0]}_."
    if recs_found:
        text += f" **Follow-up needed:** {'; '.join(recs_found[:2])}."
    if not recs_found:
        text += " No follow-up actions were flagged."

    return _reply(text, {"found": True, "id": hit["id"]},
                  new_ctx={"last_report_id": hit["id"],
                            "last_institution": _field(hit,"institution"),
                            "last_equipment": _field(hit,"equipment")})


def _answer_institution(records: list[dict], institution: str) -> dict:
    summ = _get_institution_summary(records, institution)
    hits = summ["hits"]
    if not hits:
        return _reply(
            f"I don't have any reports for **{institution}** yet. "
            f"If the name is spelled differently on the forms, try a partial name.",
            {"count": 0}
        )

    eq_list = ", ".join(f"{eq} ({n}x)" for eq, n in summ["equipment"].most_common(5))
    job_list = ", ".join(f"{j} ({n}x)" for j, n in summ["jobs"].most_common(3))
    nonop = summ["status"].get("Non-Operational", 0)
    followups = summ["followups"]

    lines = [
        f"**{institution}** has **{len(hits)}** service report(s) on record.\n",
        f"**Equipment covered:** {eq_list}",
        f"**Service types:** {job_list}",
        f"**Outcomes:** " + ", ".join(f"{s}: {n}" for s, n in summ["status"].most_common()),
    ]

    if nonop > 0:
        nonop_recs = [r for r in hits if "non-operational" in _field(r,"statusAfter").lower()]
        eq_nonop = list({_field(r,"equipment") for r in nonop_recs})
        lines.append(f"\n⚠ **{nonop} visit(s) ended with a non-operational machine** "
                     f"({', '.join(eq_nonop)}). Consider prioritising follow-up for these units.")

    if followups:
        lines.append(f"\n📋 **{len(followups)} report(s) have open recommendations** "
                     f"that may still need action.")

    lines += ["", _bullet_list(hits)]
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]},
                  new_ctx={"last_institution": institution})


def _answer_equipment(records: list[dict], equipment: str) -> dict:
    hits = _get_equipment_history(records, equipment)
    if not hits:
        return _reply(
            f"No service history found for **{equipment}**. "
            f"It may be listed under a slightly different name on the forms.",
            {"count": 0}
        )

    serials = list({_field(r,"serial") for r in hits if _field(r,"serial")})
    inst_list = list({_field(r,"institution") for r in hits})
    action_counter: Counter = Counter()
    for r in hits:
        for ln in split_lines(_field(r,"actionsTaken")):
            for tag in tag_action_line(ln):
                action_counter[tag] += 1
    nonop = [r for r in hits if "non-operational" in _field(r,"statusAfter").lower()]

    lines = [
        f"Service history for **{equipment}**: **{len(hits)}** visit(s) across "
        f"{len(inst_list)} institution(s).\n",
        f"**Serials on record:** {', '.join(serials) if serials else 'not recorded'}",
        f"**Installed at:** {', '.join(inst_list[:5])}",
        f"**Most common actions:** " +
        (", ".join(f"{t} ({n}x)" for t, n in action_counter.most_common(4)) or "none recorded"),
    ]

    if nonop:
        lines.append(f"\n⚠ Left non-operational in **{len(nonop)}** visit(s). "
                     f"If this keeps happening, it could point to a hardware issue "
                     f"or a need for more frequent calibration.")

    recs_w_followup = [r for r in hits if _match_recommendations(r)]
    if recs_w_followup:
        lines.append(f"\n📋 **{len(recs_w_followup)} report(s) recommend follow-up** for this equipment.")

    lines += ["", _bullet_list(hits)]
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]},
                  new_ctx={"last_equipment": equipment})


def _answer_trends(records: list[dict]) -> dict:
    if not records:
        return _reply("No records yet — confirm some reports first.", {"count": 0})

    action_counter: Counter = Counter()
    for r in records:
        for ln in split_lines(_field(r,"actionsTaken")):
            for tag in tag_action_line(ln):
                action_counter[tag] += 1

    inst_counter: Counter  = Counter(_field(r,"institution") for r in records)
    eq_counter: Counter    = Counter(_field(r,"equipment")   for r in records)
    status_counter: Counter = Counter(_field(r,"statusAfter") for r in records)
    nonop = status_counter.get("Non-Operational", 0)
    followup_count = sum(1 for r in records if _match_recommendations(r))

    # Identify equipment with repeated non-op
    eq_nonop: Counter = Counter()
    for r in records:
        if "non-operational" in _field(r,"statusAfter").lower():
            eq_nonop[_field(r,"equipment")] += 1
    chronic = [(eq, n) for eq, n in eq_nonop.most_common(3) if n >= 2]

    # IMPROVEMENT — Time between visits per equipment
    visit_gaps = _compute_visit_gaps(records)

    lines = [
        f"Here's a cross-database analysis of your **{len(records)} confirmed report(s)**:\n",
        f"**Most frequent actions performed:**",
    ]
    for tag, n in action_counter.most_common(5):
        lines.append(f"  • {tag.capitalize()}: {n}x across all reports")

    lines.append(f"\n**Most-serviced equipment:**")
    for eq, n in eq_counter.most_common(4):
        lines.append(f"  • {eq}: {n} visit(s)")

    lines.append(f"\n**Most-serviced institutions:**")
    for inst, n in inst_counter.most_common(3):
        lines.append(f"  • {inst}: {n} visit(s)")

    lines.append(f"\n**Outcomes across all visits:** " +
                 " | ".join(f"{s}: {n}" for s, n in status_counter.most_common()))

    if chronic:
        lines.append(f"\n⚠ **Chronic non-operational pattern detected:**")
        for eq, n in chronic:
            lines.append(f"  • **{eq}** was left non-operational in {n} separate visits — "
                         f"this warrants a deeper inspection or escalation.")

    if followup_count > 0:
        pct = round(followup_count / len(records) * 100)
        lines.append(f"\n📋 **{followup_count} of {len(records)} reports ({pct}%) have open recommendations.** "
                     f"Consider reviewing them for unresolved issues.")

    # Service interval analysis
    if visit_gaps:
        lines.append(f"\n📅 **Service intervals (avg days between visits):**")
        for eq, avg_days in sorted(visit_gaps.items(), key=lambda x: x[1], reverse=True)[:4]:
            flag = " ⚠ overdue?" if avg_days > 180 else ""
            lines.append(f"  • {eq}: ~{avg_days} days avg{flag}")

    return _reply("\n".join(lines), {"count": len(records)})


def _answer_suggest(records: list[dict], keywords: list[str]) -> dict:
    """Analyse the database and propose actionable suggestions."""
    nonop_analysis = _nonop_analysis(records)
    nonop_recs = nonop_analysis["records"]
    followup_recs = [r for r in records if _match_recommendations(r)]

    # Build context from all records
    action_counter: Counter = Counter()
    for r in records:
        for ln in split_lines(_field(r, "actionsTaken")):
            for tag in tag_action_line(ln): action_counter[tag] += 1

    eq_nonop: Counter = Counter()
    for r in nonop_recs:
        eq_nonop[_field(r,"equipment")] += 1

    lines = ["Based on the service records I have, here are some observations and suggestions:\n"]

    if nonop_recs:
        lines.append(f"**🔴 Non-operational machines ({len(nonop_recs)} cases):**")
        for eq, n in eq_nonop.most_common(3):
            inst_list = list({_field(r,"institution") for r in nonop_recs
                              if _fuzzy_in(eq, _field(r,"equipment"))})
            lines.append(f"  • **{eq}** at {', '.join(inst_list[:2])}: left non-operational {n}x. "
                         f"Recommend scheduling a dedicated corrective maintenance visit "
                         f"and verifying parts availability.")

    if followup_recs:
        lines.append(f"\n**📋 Open follow-up items ({len(followup_recs)} reports):**")
        for r in followup_recs[:4]:
            rems = split_lines(_field(r,"remarks"))
            recs_found = [ln for ln in rems if classify_remark(ln) == "recommendation"]
            if recs_found:
                lines.append(f"  • **{r['id']}** ({_field(r,'institution')}): {recs_found[0]}")

    top_action = action_counter.most_common(1)
    if top_action:
        lines.append(f"\n**💡 Most common action across all visits is '{top_action[0][0]}'.** "
                     f"If this keeps recurring, it may be worth reviewing the PM checklist "
                     f"to prevent issues before they escalate.")

    if not nonop_recs and not followup_recs:
        lines.append("All confirmed machines are operational with no open follow-up items. "
                     "The current service routine appears to be effective.")

    return _reply("\n".join(lines), {"suggestions": True})


def _answer_compare_equipment(records: list[dict], report_id: str) -> dict:
    hit = next((r for r in records
                if r.get("id","").replace(" ","").upper() == report_id.upper()), None)
    if not hit:
        return _reply(f"I couldn't find report **{report_id}** to compare against.", {"found": False})

    eq = _field(hit, "equipment")
    history = [r for r in records if _fuzzy_in(eq, _field(r,"equipment"), 0.7)
               and r.get("id") != hit.get("id")]

    if not history:
        return _reply(
            f"**{hit['id']}** is the only record I have for **{eq}**. "
            f"Once more visits are confirmed, I'll be able to compare service patterns.",
            {"found": True, "history_count": 0}
        )

    # Compare outcomes
    outcomes = Counter(_field(r,"statusAfter") for r in history)
    nonop_hist = [r for r in history if "non-operational" in _field(r,"statusAfter").lower()]
    this_outcome = _field(hit, "statusAfter")

    lines = [
        f"Comparing **{hit['id']}** against **{len(history)} other visit(s)** for **{eq}**:\n",
        f"**This visit:** {_field(hit,'jobType')} on {_field(hit,'dateStart')} — outcome: **{this_outcome}**",
        f"**Previous visit outcomes:** " + ", ".join(f"{s}: {n}" for s, n in outcomes.most_common()),
    ]

    if "non-operational" in this_outcome.lower() and len(nonop_hist) > 0:
        lines.append(f"\n⚠ This equipment has been left non-operational "
                     f"in {len(nonop_hist)+1} out of {len(history)+1} visits total — "
                     f"a recurring pattern that may need escalation.")
    elif "non-operational" in this_outcome.lower():
        lines.append(f"\nThis is the first non-operational outcome for this equipment in the records.")
    else:
        lines.append(f"\nThis visit's outcome is consistent with previous visits.")

    # Action comparison
    this_actions = set()
    for ln in split_lines(_field(hit,"actionsTaken")):
        this_actions.update(tag_action_line(ln))

    all_actions: Counter = Counter()
    for r in history:
        for ln in split_lines(_field(r,"actionsTaken")):
            for tag in tag_action_line(ln): all_actions[tag] += 1

    new_actions = this_actions - set(all_actions.keys())
    if new_actions:
        lines.append(f"\n🆕 New action type(s) in this visit (not seen before): "
                     f"**{', '.join(new_actions)}** — this may indicate a new issue.")

    return _reply("\n".join(lines), {"found": True, "history_count": len(history)},
                  new_ctx={"last_equipment": eq, "last_report_id": hit["id"]})


def _answer_date(records: list[dict], month: int | None, year: int | None) -> dict:
    hits = [r for r in records if _match_date_filter(r, month, year)]
    month_names = ["","January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    label = (f"{month_names[month]} {year}" if month and year
             else month_names[month] if month else str(year) if year else "that period")
    if not hits:
        return _reply(
            f"No reports found for **{label}**. "
            f"Records currently span " + _date_range_str(records) + ".",
            {"count": 0}
        )
    status_counter = Counter(_field(r,"statusAfter") for r in hits)
    lines = [
        f"Found **{len(hits)}** report(s) for **{label}**.\n",
        _bullet_list(hits),
        f"\n**Outcomes this period:** " + ", ".join(f"{s}: {n}" for s, n in status_counter.most_common()),
    ]
    followups = [r for r in hits if _match_recommendations(r)]
    if followups:
        lines.append(f"📋 {len(followups)} had open follow-up recommendations.")
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]})


def _match_date_filter(rec: dict, month: int | None, year: int | None) -> bool:
    d = _date_of(rec)
    if d is None: return False
    if month and year: return d.month == month and d.year == year
    if month: return d.month == month
    if year: return d.year == year
    return False


def _date_range_str(records: list[dict]) -> str:
    dates = [_date_of(r) for r in records if _date_of(r)]
    if not dates: return "no date range available"
    return f"{min(dates)} to {max(dates)}"


def _answer_job(records: list[dict], job_type: str) -> dict:
    hits = [r for r in records if job_type.lower() in _field(r,"jobType").lower()]
    if not hits:
        return _reply(f"No **{job_type}** reports found yet.", {"count": 0})
    status_counter = Counter(_field(r,"statusAfter") for r in hits)
    inst_counter = Counter(_field(r,"institution") for r in hits)
    lines = [
        f"Found **{len(hits)}** **{job_type}** visit(s).\n",
        f"**Outcomes:** " + ", ".join(f"{s}: {n}" for s, n in status_counter.most_common()),
        f"**Institutions visited:** " + ", ".join(f"{i} ({n}x)" for i, n in inst_counter.most_common(4)),
        "",
        _bullet_list(hits),
    ]
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]})


def _answer_status(records: list[dict], status: str) -> dict:
    hits = [r for r in records if status.lower() in _field(r,"statusAfter").lower()]
    if not hits:
        return _reply(f"No reports with status **{status}** found — that's good news if "
                      f"you were looking for non-operational machines.", {"count": 0})
    eq_list = ", ".join(f"{eq} ({n}x)" for eq, n in Counter(_field(r,"equipment") for r in hits).most_common(5))
    lines = [
        f"Found **{len(hits)}** report(s) where the machine was left **{status}**.\n",
        f"**Affected equipment:** {eq_list}",
        "",
        _bullet_list(hits),
    ]
    if "non-operational" in status.lower():
        recs_w_followup = [r for r in hits if _match_recommendations(r)]
        if recs_w_followup:
            lines.append(f"\n📋 {len(recs_w_followup)} of these have pending follow-up recommendations.")
        else:
            lines.append(f"\n💡 None of these non-operational reports have documented follow-up plans — "
                         f"consider adding recommendations when you review them.")
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]})


def _answer_actions(records: list[dict], keyword: str) -> dict:
    hits = [r for r in records if keyword.lower() in _field(r,"actionsTaken").lower()]
    if not hits:
        return _reply(f"No reports found with '{keyword}' in the actions taken.", {"count": 0})
    # Cross-reference: what were the outcomes on these visits?
    status_counter = Counter(_field(r,"statusAfter") for r in hits)
    lines = [
        f"Found **{len(hits)}** report(s) where '{keyword}' was performed.\n",
        f"**Outcomes:** " + ", ".join(f"{s}: {n}" for s, n in status_counter.most_common()),
        "",
        _bullet_list(hits),
    ]
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]})


def _answer_parts(records: list[dict]) -> dict:
    hits = [r for r in records
            if any((p.get("desc") or "").strip()
                   for p in ((r.get("data") or {}).get("spareParts") or []))]
    # IMPROVEMENT: also check records that mention parts in actions/remarks even if
    # the parts table was not filled in
    keyword_hits = [r for r in records
                    if r not in hits and
                    any(w in _field(r,"actionsTaken").lower() + " " + _field(r,"remarks").lower()
                        for w in ("gas","cassette","reagent","strip","cord","filter","bulb","lamp","fuse"))]

    all_hits = hits + keyword_hits
    if not all_hits:
        return _reply(
            "No reports with recorded spare parts or reagents. "
            "If parts are being used, make sure they're entered in the spare parts table during review.",
            {"count": 0}
        )

    # Global frequency
    all_parts: list[str] = []
    for r in hits:
        for p in (r.get("data") or {}).get("spareParts") or []:
            desc = (p.get("desc") or "").strip()
            if desc: all_parts.append(desc)
    counter = Counter(all_parts)

    # IMPROVEMENT: per-equipment breakdown
    eq_parts = _get_parts_per_equipment(hits)

    lines = [f"**{len(hits)}** report(s) used documented spare parts or reagents.\n"]
    if counter:
        lines.append("**Items used (all reports):**")
        for item, n in counter.most_common(10):
            lines.append(f"  • {item}: {n}x")

    if eq_parts:
        lines.append("\n**Parts breakdown by equipment:**")
        for eq, parts_ctr in sorted(eq_parts.items(), key=lambda x: -sum(x[1].values()))[:5]:
            top = ", ".join(f"{p} ({n}x)" for p, n in parts_ctr.most_common(3))
            lines.append(f"  • **{eq}**: {top}")

    if keyword_hits:
        lines.append(f"\n💡 {len(keyword_hits)} additional report(s) mention parts/consumables "
                     f"in their remarks but have no items in the parts table — "
                     f"consider updating those records.")

    lines += ["", _bullet_list(hits)]
    return _reply("\n".join(lines),
                  {"count": len(hits), "parts_frequency": dict(counter)})


def _answer_recommendations(records: list[dict]) -> dict:
    hits = [r for r in records if _match_recommendations(r)]
    if not hits:
        return _reply(
            "No open follow-up recommendations found in the current records. "
            "Everything looks resolved for now.",
            {"count": 0}
        )
    lines = [
        f"**{len(hits)}** report(s) have pending follow-up recommendations:\n",
    ]
    for r in hits[:8]:
        rems = split_lines(_field(r,"remarks"))
        recs_found = [ln for ln in rems if classify_remark(ln) == "recommendation"]
        snippet = "; ".join(recs_found[:1]) if recs_found else _field(r,"remarks")[:60]
        inst = _field(r,"institution")
        eq   = _field(r,"equipment")
        lines.append(f"• **{r['id']}** — {inst} | {eq}: _{snippet}_")
    if len(hits) > 8:
        lines.append(f"  _…and {len(hits)-8} more_")
    lines.append(f"\n💡 Consider scheduling follow-up visits for the non-operational cases "
                 f"and ordering any parts flagged in the remarks.")
    return _reply("\n".join(lines), {"count": len(hits), "ids": [r["id"] for r in hits]})


def _answer_provider(records: list[dict], name: str) -> dict:
    hits = [r for r in records
            if any(w in _field(r,"serviceProvider").lower()
                   for w in re.findall(r"[a-z]+", name.lower()) if len(w) >= 4)]
    if not hits:
        return _reply(f"No reports found for a provider matching '{name}'.", {"count": 0})

    eq_counter   = Counter(_field(r,"equipment")   for r in hits)
    inst_counter = Counter(_field(r,"institution")  for r in hits)
    # IMPROVEMENT: provider performance metrics
    status_counter = Counter(_field(r,"statusAfter") for r in hits)
    nonop_count    = sum(1 for r in hits if "non-operational" in _field(r,"statusAfter").lower())
    followup_count = sum(1 for r in hits if _match_recommendations(r))
    total          = len(hits)

    lines = [
        f"Found **{total}** report(s) for provider **{name}**.\n",
        f"**Equipment handled:** " + ", ".join(f"{e} ({n}x)" for e, n in eq_counter.most_common(4)),
        f"**Institutions visited:** " + ", ".join(list(inst_counter.keys())[:4]),
        f"**Outcomes:** " + ", ".join(f"{s}: {n}" for s, n in status_counter.most_common()),
    ]

    # Performance insight
    op_rate = round((total - nonop_count) / total * 100) if total else 0
    lines.append(f"\n📊 **Operational outcome rate: {op_rate}%** ({total - nonop_count}/{total} visits).")
    if nonop_count > 0:
        lines.append(f"  ⚠ {nonop_count} visit(s) ended with a non-operational machine.")
    if followup_count > 0:
        lines.append(f"  📋 {followup_count} report(s) have pending follow-up recommendations.")

    lines += ["", _bullet_list(hits)]
    return _reply("\n".join(lines), {"count": total, "ids": [r["id"] for r in hits]})


def _answer_general(records: list[dict], keywords: list[str]) -> dict:
    """IMPROVEMENT: search NLP-normalized fields first, fall back to raw data blob."""
    # Priority 1: search confirmed data fields (clean human-corrected values)
    data_hits = [r for r in records
                 if any(k in json.dumps(r.get("data") or {}).lower() for k in keywords)]
    # Priority 2: also search NLP-normalized values if available
    nlp_hits = [r for r in records
                if r not in data_hits and
                any(k in json.dumps((r.get("nlp") or {}).get("fields") or {}).lower()
                    for k in keywords)]
    all_hits = data_hits + nlp_hits

    if not all_hits:
        return _reply(
            "I couldn't find records matching those terms. "
            "Try asking about a specific institution, equipment model, job type, "
            "or report number. You can also ask me to analyse trends or suggest improvements.",
            {"count": 0}
        )
    lines = [f"Found **{len(all_hits)}** report(s) matching your query:\n", _bullet_list(all_hits)]
    return _reply("\n".join(lines), {"count": len(all_hits)})


def _answer_explain(records: list[dict], report_id: str) -> dict:
    """IMPROVEMENT: proper causal explanation using causeOfProblem, reportedProblem,
    action tags, and cross-reference with similar reports."""
    hit = next((r for r in records
                if r.get("id","").replace(" ","").upper() == report_id.upper()), None)
    if not hit:
        return _reply(f"I couldn't find report **{report_id}** to explain.", {"found": False})

    d = hit.get("data") or {}
    problem   = _field(hit, "reportedProblem")
    cause     = _field(hit, "causeOfProblem")
    actions   = split_lines(_field(hit, "actionsTaken"))
    outcome   = _field(hit, "statusAfter")
    equipment = _field(hit, "equipment")
    tags_seen: Counter = Counter()
    for ln in actions:
        for tag in tag_action_line(ln):
            tags_seen[tag] += 1

    lines = [f"**Explanation for report {hit['id']}** ({equipment} at {_field(hit,'institution')}):\n"]

    # Problem chain
    if not is_empty(problem):
        lines.append(f"**What was reported:** {problem}")
    if not is_empty(cause):
        lines.append(f"**Root cause identified:** {cause}")
    elif is_empty(problem) and is_empty(cause):
        lines.append("No problem or cause was recorded — this was likely a routine PM visit.")

    # What the technician did
    if tags_seen:
        dominant_tags = [t for t, _ in tags_seen.most_common(3)]
        lines.append(f"\n**Response:** The technician primarily performed "
                     f"**{', '.join(dominant_tags)}** actions ({len(actions)} steps total).")

    # Outcome explanation
    if "non-operational" in outcome.lower():
        lines.append(f"\n**Outcome: Non-Operational** — the machine could not be restored to service "
                     f"during this visit. This may be because parts were unavailable, "
                     f"the fault was beyond on-site repair, or the machine was decommissioned.")
        # Check remarks for more context
        rems = split_lines(_field(hit,"remarks"))
        for ln in rems:
            if classify_remark(ln) == "recommendation":
                lines.append(f"  The technician recommended: _{ln}_")
                break
    else:
        lines.append(f"\n**Outcome: {outcome}** — the machine was returned to service.")

    # Cross-reference: same equipment history for context
    history = [r for r in records
               if _fuzzy_in(equipment, _field(r,"equipment"), 0.7)
               and r.get("id") != hit.get("id")]
    if history:
        same_cause = [r for r in history
                      if not is_empty(_field(r,"causeOfProblem")) and
                      _ratio(_field(r,"causeOfProblem").lower(), cause.lower()) > 50]
        if same_cause:
            lines.append(f"\n🔁 This same cause was recorded in **{len(same_cause)}** previous "
                         f"visit(s) for this equipment — suggesting a recurring fault "
                         f"rather than a one-off incident.")
        prev_nonop = [r for r in history if "non-operational" in _field(r,"statusAfter").lower()]
        if prev_nonop:
            lines.append(f"⚠ This equipment has been left non-operational "
                         f"**{len(prev_nonop)} time(s)** before — worth escalating if unresolved.")

    return _reply("\n".join(lines), {"found": True, "id": hit["id"]},
                  new_ctx={"last_report_id": hit["id"],
                            "last_institution": _field(hit,"institution"),
                            "last_equipment": equipment})


def _answer_overdue_pm(records: list[dict]) -> dict:
    """IMPROVEMENT: Flag equipment that may be overdue for a PM visit."""
    from datetime import date as _date
    today = _date.today()

    # Group records by equipment, find last visit date
    eq_last: dict[str, _date] = {}
    eq_job: dict[str, str] = {}
    for r in records:
        eq = _field(r, "equipment")
        d  = _date_of(r)
        if eq and d:
            if eq not in eq_last or d > eq_last[eq]:
                eq_last[eq] = d
                eq_job[eq]  = _field(r, "jobType")

    if not eq_last:
        return _reply("No dated records found — cannot calculate service intervals.", {"count": 0})

    # Categorise by days since last visit
    overdue_180: list[tuple[str, int]] = []
    overdue_90:  list[tuple[str, int]] = []
    recent:      list[tuple[str, int]] = []

    for eq, last in sorted(eq_last.items(), key=lambda x: x[1]):
        days_ago = (today - last).days
        if days_ago >= 180:
            overdue_180.append((eq, days_ago))
        elif days_ago >= 90:
            overdue_90.append((eq, days_ago))
        else:
            recent.append((eq, days_ago))

    lines = [f"**Service currency check** — {len(eq_last)} equipment model(s) on record:\n"]

    if overdue_180:
        lines.append(f"🔴 **Possibly overdue (180+ days since last visit):**")
        for eq, d in overdue_180:
            lines.append(f"  • **{eq}**: last serviced {d} days ago ({eq_last[eq]})")

    if overdue_90:
        lines.append(f"\n🟡 **Monitor (90–179 days since last visit):**")
        for eq, d in overdue_90:
            lines.append(f"  • **{eq}**: last serviced {d} days ago ({eq_last[eq]})")

    if recent:
        lines.append(f"\n✅ **Recently serviced (< 90 days):**")
        for eq, d in recent:
            lines.append(f"  • {eq}: {d} days ago")

    if not overdue_180 and not overdue_90:
        lines.append("All equipment has been serviced within the last 90 days. 👍")
    else:
        lines.append(f"\n💡 Consider scheduling visits for the overdue equipment, "
                     f"especially if they are critical diagnostic instruments.")

    return _reply("\n".join(lines), {
        "overdue_180": len(overdue_180),
        "overdue_90":  len(overdue_90),
        "recent":      len(recent),
    })


def _answer_unknown() -> dict:
    return _reply(
        "I'm the SBSI FSR Assistant. I can help you with:\n\n"
        "• **Report details:** _Tell me about report A15273_\n"
        "• **Explain a report:** _Why was A15273 left non-operational?_\n"
        "• **Institution history:** _Reports for Butuan Medical Center_\n"
        "• **Equipment analysis:** _Show me all DG Spin visits_\n"
        "• **Trends & patterns:** _What patterns do you see across the records?_\n"
        "• **Suggestions:** _What should we focus on next?_\n"
        "• **Overdue PM:** _Which equipment is overdue for a service visit?_\n"
        "• **Non-operational machines:** _Which machines need attention?_\n"
        "• **Follow-up items:** _What recommendations are still pending?_\n"
        "• **Spare parts tracking:** _Which equipment uses the most consumables?_\n"
        "• **Provider performance:** _How has Migo Navarro been performing?_\n"
        "• **Comparisons:** _Compare A15273 against other visits_\n"
        "• **Statistics:** _Give me an overview of all reports_",
        {"intent": "help"}
    )


# ---------------------------------------------------------------------------
# Public entry point — accepts conversation history
# ---------------------------------------------------------------------------

def answer(question: str, history: list[dict] | None = None) -> dict:
    """
    Parse `question` with conversation context, query records, return:
        {text, data, intent, params, context_update, total_records}

    `history` is a list of previous turns: [{"role": "user"|"assistant", "content": str}, ...]
    """
    # Build context from last assistant turn
    ctx: dict = {}
    if history:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and turn.get("context_update"):
                ctx = turn["context_update"]
                break

    intent_obj = parse_intent(question, context=ctx)
    intent = intent_obj["intent"]
    params = intent_obj["params"]
    records = load_records()

    if   intent == "lookup":           result = _answer_lookup(records, params["report_id"])
    elif intent == "summary":          result = _answer_summary(records, params["report_id"])
    elif intent == "count":            result = _answer_count(records, params)
    elif intent == "search_institution": result = _answer_institution(records, params["institution"])
    elif intent == "search_equipment": result = _answer_equipment(records, params["equipment"])
    elif intent == "search_date":      result = _answer_date(records, params.get("month"), params.get("year"))
    elif intent == "search_job":       result = _answer_job(records, params["job_type"])
    elif intent == "search_status":    result = _answer_status(records, params["status"])
    elif intent == "search_actions":   result = _answer_actions(records, params["keyword"])
    elif intent == "search_parts":     result = _answer_parts(records)
    elif intent == "search_recommendations": result = _answer_recommendations(records)
    elif intent == "search_provider":  result = _answer_provider(records, params["name"])
    elif intent == "analyse_trends":   result = _answer_trends(records)
    elif intent == "suggest":          result = _answer_suggest(records, params.get("keywords",[]))
    elif intent == "overdue_pm":       result = _answer_overdue_pm(records)
    elif intent == "compare_equipment": result = _answer_compare_equipment(records, params.get("report_id",""))
    elif intent == "compare":          result = _answer_trends(records)  # general compare → trends
    elif intent == "explain":
        rid = params.get("report_id") or ctx.get("last_report_id","")
        result = _answer_explain(records, rid) if rid else _answer_trends(records)
    elif intent == "search_general":   result = _answer_general(records, params["keywords"])
    else:                              result = _answer_unknown()

    result["intent"] = intent
    result["params"] = params
    result["total_records"] = len(records)
    return result
