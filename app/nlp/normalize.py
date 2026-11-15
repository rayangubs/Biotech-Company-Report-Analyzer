"""Rule-based normalization for FM-SER-03 fields.

Baseline for the NLP comparison (rules vs CRF vs transformer) and a cleanup step
between OCR and the review form. Pure Python; rapidfuzz is used if installed,
otherwise difflib from the standard library.

Every scalar normalizer returns a dict:
    raw       the text as given
    value     normalized value, or None
    ok        True = parsed/validated, False = failed a check, None = could not validate
    warnings  list of short codes
    na        True when the field is blank / NA / dashes (value is None then)

Blank/NA/dash fields are handled explicitly so a model can't look good by predicting them.

GUESSES (check them against real data with scripts/eval_normalize.py):
  - dates are month-first (set FSR_DATE_ORDER=DMY if SBSI writes day-first)
  - report numbers are one letter + 4-6 digits ("A 15273")
  - serial numbers: only cleaned and sanity-checked, no known format
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

try:  # optional
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None

__all__ = [
    # text cleaning
    "clean_text", "clean_multiline", "is_empty",
    # field normalizers
    "normalize_date", "normalize_time", "normalize_report_no",
    "normalize_serial", "normalize_free_text", "normalize_multiline_text",
    # lexicon / fuzzy
    "load_lexicon", "best_match", "match_lexicon",
    # similarity (public alias for the internal _ratio)
    "similarity_ratio",
    # action / remark tagging
    "split_lines", "tag_action_line", "classify_remark",
    # whole-report
    "extract_report", "zone_text",
]

_MODULE_DIR = Path(__file__).resolve().parent.parent.parent   # → fsr_ocr/
LEXICON_DIR = Path(os.environ.get("FSR_LEXICON_DIR", str(_MODULE_DIR / "data" / "lexicon")))
DATE_ORDER  = os.environ.get("FSR_DATE_ORDER", "MDY").upper()

# IMPROVEMENT 2: DATE_MIN loosened — accept any date from 2000 onwards.
# The form Rev.04 is effective 2025-08-01 but older historical records (e.g.
# 2023 service visits on legacy equipment) are still valid data.  Out-of-range
# warnings are still emitted so reviewers can spot genuine OCR year errors.
DATE_MIN = date(int(os.environ.get("FSR_DATE_MIN_YEAR", "2000")), 1, 1)

AUTO_SCORE    = float(os.environ.get("FSR_LEX_AUTO",    "90"))  # replace with lexicon spelling; lowered from 92 so near-misses like "Bitian"->"Butuan" auto-correct
SUGGEST_SCORE = float(os.environ.get("FSR_LEX_SUGGEST", "75"))  # only suggest, do not auto-replace

# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------
_DASHES  = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_NA_WORDS = {"na", "n/a", "n.a", "n.a.", "n a", "none", "nil"}
_BLANK_RE = re.compile(r"[\s\-_.~/\\|:,;" + _DASHES + r"]*")


def clean_text(s) -> str:
    """Single-line text: NBSP and control chars removed, whitespace collapsed."""
    if s is None:
        return ""
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", s)
    return " ".join(s.split())


def clean_multiline(s) -> str:
    """IMPROVEMENT 5: Multiline text — cleans each line individually,
    preserving intentional line breaks for actionsTaken / remarks."""
    if s is None:
        return ""
    lines = str(s).replace("\u00a0", " ").splitlines()
    cleaned = []
    for ln in lines:
        ln = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", ln)
        ln = " ".join(ln.split())
        if ln:
            cleaned.append(ln)
    return "\n".join(cleaned)


def is_empty(s) -> bool:
    """True for blank, dashes/underscores/dots only, or NA / N/A / none / nil."""
    t = clean_text(s).lower()
    return t in _NA_WORDS or bool(_BLANK_RE.fullmatch(t))


def _res(raw, value, ok, warnings=None, **extra):
    d = {"raw": raw, "value": value, "ok": ok, "warnings": list(warnings or []), "na": False}
    d.update(extra)
    return d


def _na(raw):
    return _res(raw, None, True, na=True)


# IMPROVEMENT 1: Expanded OCR confusion tables.
# Original table missed G→6, C→0 (in digit position), a→0 seen in real data.
# Two tables are kept separate so we apply the right one per context:
#   _TO_DIGIT  — when we expect a digit (dates, times, serials, report numbers)
#   _TO_LETTER — when we expect an upper-case letter (report number prefix)
_TO_DIGIT = str.maketrans({
    "O": "0", "o": "0",
    "D": "0", "Q": "0",          # D and Q misread as digit 0
    "I": "1", "l": "1",
    "|": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",                     # NEW: G → 6  (seen in "Oct Go" → "Oct 60" → "Oct 06")
    "C": "0",                     # NEW: C → 0  (seen in "Oct. C6" → day 06)
    "a": "0",                     # NEW: a → 0  (seen in "3la 000" in serial)
    "g": "9",                     # NEW: g → 9  (rounded 9 often read as g)
    "T": "7",                     # NEW: capital T sometimes substitutes 7
})

_TO_LETTER = str.maketrans({
    "0": "O",
    "1": "I",
    "4": "A",
    "5": "S",
    "8": "B",
    "2": "Z",
})


def _digitize(s: str) -> str:
    return s.translate(_TO_DIGIT)


def _ratio(a: str, b: str) -> float:
    """Similarity 0-100."""
    if _fuzz is not None:
        return float(_fuzz.ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


# Public alias used by query_engine and other modules.
similarity_ratio = _ratio


# ---------------------------------------------------------------------------
# IMPROVEMENT 1 cont.: smarter digit extraction for date fields
# Applies _TO_DIGIT only to runs of characters that look like they should be
# digits (mixed letter/digit clusters), not to pure-letter month names.
# ---------------------------------------------------------------------------

def _fix_digit_run(s: str) -> str:
    """Apply OCR digit corrections inside numeric-ish tokens only."""
    def _fix_token(m: re.Match) -> str:
        tok = m.group(0)
        # Leave pure letter words alone (month names like "Oct", "Sept")
        if tok.isalpha():
            return tok
        # Apply digit fixes to mixed / mostly-digit tokens
        return tok.translate(_TO_DIGIT)
    return re.sub(r"[A-Za-z0-9]+", _fix_token, s)


# ---------------------------------------------------------------------------
# dates, times, ids
# ---------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def normalize_date(text, today: date | None = None):
    raw = text
    if is_empty(text):
        return _na(raw)
    t = clean_text(text)
    today = today or date.today()
    warn = []

    # IMPROVEMENT 1: apply smart digit fixing before parsing
    t_fixed = _fix_digit_run(t)

    month_word = next(
        (w for w in re.findall(r"[A-Za-z]{3,9}", t_fixed) if w[:3].lower() in _MONTHS),
        None,
    )
    if month_word:  # "12 Oct 2026", "Oct 12, 2026", "Sept. 12, 2025"
        mo = _MONTHS[month_word[:3].lower()]
        rest = t_fixed.replace(month_word, " ", 1)
        nums_raw = re.findall(r"\d+", rest)
        nums = [int(n) for n in nums_raw if n.isdigit()]
        if len(nums) != 2:
            return _res(raw, None, False, ["unparsed"])
        a, b = nums
        y, d = (a, b) if a > 31 else (b, a)
        # If day is still impossible (e.g. G→6 gave 60), clamp to last digit or 1
        if d > 31:
            d = d % 10 or 1   # e.g. 60 → keep units digit = 0 → use 1
            warn.append("day_clamped")
    else:
        nums = re.findall(r"\d+", _digitize(t_fixed))
        if len(nums) == 1 and len(nums[0]) == 8 and nums[0].startswith("20"):
            s = nums[0]
            nums = [s[:4], s[4:6], s[6:]]
        elif len(nums) == 1 and len(nums[0]) in (6, 8):
            s = nums[0]
            nums = [s[:2], s[2:4], s[4:]]
        if len(nums) != 3:
            return _res(raw, None, False, ["unparsed"])
        if len(nums[0]) == 4:  # ISO: Y-M-D
            y, mo, d = (int(n) for n in nums)
        else:
            a, b, c = (int(n) for n in nums)
            mo, d = (a, b) if DATE_ORDER == "MDY" else (b, a)
            if mo > 12 and d <= 12:
                mo, d = d, mo
            if a <= 12 and b <= 12 and a != b:
                warn.append("ambiguous_day_month")
            y = c
    if y < 100:
        y += 2000
    try:
        dt = date(y, mo, d)
    except ValueError:
        # Last resort: try swapping day and month
        try:
            dt = date(y, d, mo)
            warn.append("day_month_swapped")
        except ValueError:
            return _res(raw, None, False, warn + ["invalid_date"])
    if not (DATE_MIN <= dt <= today + timedelta(days=1)):
        warn.append("out_of_range")
    return _res(raw, dt.isoformat(), "out_of_range" not in warn, warn)


def normalize_time(text):
    """-> "HH:MM" (24h). Without AM/PM the hour is kept as written and flagged no_am_pm."""
    raw = text
    if is_empty(text):
        return _na(raw)
    t = clean_text(text).lower()
    ampm = None
    m = re.search(r"(?<![a-z])([ap])\.?\s?m(?![a-z])", t)
    if m:
        ampm = "am" if m.group(1) == "a" else "pm"
        t = t[: m.start()] + " " + t[m.end():]
    nums = re.findall(r"\d+", _digitize(t))
    if len(nums) == 2:
        h, mi = nums
    elif len(nums) == 1 and len(nums[0]) in (3, 4):
        h, mi = nums[0][:-2], nums[0][-2:]
    elif len(nums) == 1 and len(nums[0]) <= 2:
        h, mi = nums[0], "00"
    else:
        return _res(raw, None, False, ["unparsed"])
    h, mi = int(h), int(mi)
    if mi > 59 or h > 23:
        return _res(raw, None, False, ["invalid_time"])
    warn = []
    if ampm:
        # IMPROVEMENT 3: correct noon/midnight handling
        if not 1 <= h <= 12:
            return _res(raw, None, False, ["invalid_time"])
        if ampm == "am":
            h = 0 if h == 12 else h      # 12 AM  = midnight = 00:xx
        else:
            h = 12 if h == 12 else h + 12 # 12 PM = noon, 1 PM = 13
    elif 1 <= h <= 12:
        warn.append("no_am_pm")
    return _res(raw, f"{h:02d}:{mi:02d}", True, warn)


def normalize_report_no(text):
    """One letter + 4-6 digits -> "A 15273" (GUESS at the format)."""
    raw = text
    if is_empty(text):
        return _na(raw)
    t = re.sub(r"[^A-Za-z0-9]", "", clean_text(text)).upper()
    if len(t) < 5:
        return _res(raw, None, False, ["unexpected_format"], candidate=t)
    prefix, rest_raw = t[0], t[1:]
    rest = _digitize(rest_raw)
    warn = []
    if prefix.isdigit():
        if len(t) < 6:  # letter probably missing ("15273"): don't invent one
            return _res(raw, None, False, ["missing_prefix"], candidate=t)
        prefix = prefix.translate(_TO_LETTER)
        warn.append("prefix_fixed")
    if rest != rest_raw:
        warn.append("digits_fixed")
    if not (prefix.isalpha() and rest.isdigit() and 4 <= len(rest) <= 6):
        return _res(raw, None, False, ["unexpected_format"], candidate=t)
    if len(rest) != 5:  # the samples seen so far have 5 digits
        warn.append("unusual_length")
    return _res(raw, f"{prefix} {rest}", True, warn)


def normalize_serial(text):
    """Cleanup only: uppercase, no spaces. O/0 and I/1 are NOT auto-fixed (ambiguous)."""
    raw = text
    if is_empty(text):
        return _na(raw)
    t = clean_text(text).upper()
    t = re.sub(r"\s*([-/])\s*", r"\1", t).replace(" ", "")
    warn = [] if re.fullmatch(r"[A-Z0-9][A-Z0-9\-/]{3,}", t) else ["unexpected_format"]
    return _res(raw, t, not warn, warn)


def normalize_free_text(text):
    if is_empty(text):
        return _na(text)
    return _res(text, clean_text(text), True)


def normalize_multiline_text(text):
    """IMPROVEMENT 5: preserve line structure for actionsTaken / remarks."""
    if is_empty(text):
        return _na(text)
    return _res(text, clean_multiline(text), True)


# ---------------------------------------------------------------------------
# lexicon + fuzzy matching (institution, equipment)
# ---------------------------------------------------------------------------
_LEX_CACHE: dict = {}


def load_lexicon(name: str) -> list[str]:
    """data/lexicon/<name>.txt, one entry per line, '#' comments. Reloaded when the file changes."""
    p = LEXICON_DIR / f"{name}.txt"
    if not p.exists():
        return []
    mtime = p.stat().st_mtime
    cached = _LEX_CACHE.get(name)
    if not cached or cached[0] != mtime:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines()]
        _LEX_CACHE[name] = (mtime, [ln for ln in lines if ln and not ln.startswith("#")])
    return _LEX_CACHE[name][1]


def _key(s: str) -> str:
    """Lowercase, punctuation out, tokens sorted (word order doesn't matter)."""
    return " ".join(sorted(re.sub(r"[^a-z0-9]+", " ", s.lower()).split()))


def best_match(text: str, choices: list[str]):
    q = _key(text)
    best, score = None, 0.0
    for c in choices:
        s = _ratio(q, _key(c))
        if s > score:
            best, score = c, s
    return best, score


def match_lexicon(text, choices: list[str]):
    """>= AUTO_SCORE: use the lexicon spelling. >= SUGGEST_SCORE: keep OCR text, add suggestion."""
    if is_empty(text):
        return _na(text)
    t = clean_text(text)
    cand, score = best_match(t, choices) if choices else (None, 0.0)
    out = _res(text, t, None, score=round(score, 1), match=None, suggestion=None, lexicon_size=len(choices))
    if cand is not None and score >= AUTO_SCORE:
        out.update(value=cand, ok=True, match=cand)
    elif cand is not None and score >= SUGGEST_SCORE:
        out.update(suggestion=cand, ok=False)
        out["warnings"].append("low_confidence_match")
    elif choices:
        out["ok"] = False
        out["warnings"].append("not_in_lexicon")
    return out


# ---------------------------------------------------------------------------
# IMPROVEMENT 6: Expanded action tagging with SBSI-specific vocabulary
# ---------------------------------------------------------------------------
_BULLET = re.compile(r"^\s*(?:[-*>\u2022\u2013\u2014.]+|\(?\d{1,2}[.)]|\(?[A-Za-z][.)])\s*")

_STEMS = {
    "check":     ("check", "inspect", "verif", "assess", "evaluat", "monitor"),
    "clean":     ("clean", "wipe", "dust", "flush", "purg", "wash", "remov dirt", "sanitiz"),
    "test":      ("test", "pump test", "ran", "run"),
    "calibrate": ("calibrat", "recalibrat"),
    "replace":   ("replac", "swap", "chang", "substitut", "install new"),
    "update":    ("updat", "upgrad", "patch"),
    # NEW SBSI-specific categories
    "open":      ("open", "remov cover", "disassembl", "unpack"),
    "pack":      ("pack", "repack", "transport", "pull out", "uninstall"),
    "lubricate": ("lubricat", "oil", "grease"),
    "tighten":   ("tighten", "fasten", "secur bolt", "secur screw"),
}

# Exact-token matches (whole lowercase token → tag)
_EXACT = {
    "run":      "test",
    "ran":      "test",
    "opened":   "open",
    "packed":   "pack",
    "unpacked": "pack",
    "removed":  "open",
    "cleaned":  "clean",
    "checked":  "check",
    "updated":  "update",
    "replaced": "replace",
    "installed":"replace",
}

# Tokens that look like actions but are actually objects/devices
_NOT_ACTIONS = {
    "calibrator", "calibrators", "sticker", "cord", "cable",
    "strip", "cassette", "reagent", "optic", "optics",
}

# Measurement patterns — IMPROVEMENT 7
_MEASUREMENT_RE = re.compile(
    r"\b(\d[\d.,]*\s*(?:rpm|°c|°f|deg|min|mins|minutes|sec|seconds|ml|l|mg|%|psi|bar|mmhg"
    r"|c|f|hz|khz|v|mv|ma|ua|ohm|k|g|kg|cycles?))\b",
    re.I,
)

# Quantitative reading patterns
_READING_PATTERNS = [
    re.compile(r"\b\d+[\d.,]*\s*rpm\b", re.I),
    re.compile(r"\b\d+[\d.,]*\s*(mins?|minutes?|seconds?|sec)\b", re.I),
    re.compile(r"\b\d+[\d.,]*\s*[-–]\s*\d+[\d.,]*\s*(°?[cCfF]|degrees?)\b", re.I),
    re.compile(r"\b(timer|temperature|temp|speed|rpm|pressure)\s*:?\s*\d", re.I),
    re.compile(r"\b\d+[\d.,]*\s*(°c|°f|%|ml|l|mg|g|kg)\b", re.I),
]


def _line_text(x) -> str:
    return str(x.get("text", "")) if isinstance(x, dict) else str(x)


def split_lines(text_or_lines) -> list[str]:
    """Accepts a string or a list of str / {"text": ...}. Drops bullets, blanks and dash-only lines."""
    if text_or_lines is None:
        return []
    items = text_or_lines if isinstance(text_or_lines, (list, tuple)) else str(text_or_lines).splitlines()
    out = []
    for it in items:
        s = _BULLET.sub("", clean_text(_line_text(it)), count=1)
        if s and not is_empty(s):
            out.append(s)
    return out


def _tag_of(tok: str) -> str | None:
    if tok in _NOT_ACTIONS:
        return None
    if tok in _EXACT:
        return _EXACT[tok]
    if len(tok) < 3:
        return None
    for tag, stems in _STEMS.items():
        for st in stems:
            # Match if token starts with stem OR stem ratio is high enough
            if tok.startswith(st) or (len(st) >= 5 and _ratio(tok[: len(st)], st) >= 85):
                return tag
    return None


def tag_action_line(line: str) -> list[str]:
    tags: list[str] = []
    for tok in re.findall(r"[a-z]+", line.lower()):
        tag = _tag_of(tok)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# IMPROVEMENT 7: Remark classification — now 4 classes
# recommendation / status / measurement / other
# ---------------------------------------------------------------------------
_REC_FIRST = {
    "replace", "change", "schedule", "monitor", "order", "request", "repair", "recalibrate",
    "retest", "check", "inspect", "clean", "update", "upgrade", "follow", "return", "observe",
    "provide", "prepare", "endorse",
}
_REC_ANY = re.compile(
    r"\b(recommend\w*|advis\w*|suggest\w*|pending|due\b|next (?:pm|visit|service|schedule)"
    r"|for (?:follow|further|next|monitor|replac|repair|check|calibrat)\w*|needs?\b"
    r"|to be (?:replaced|changed|checked|ordered)|waiting|re-?visit|re-?schedule)",
    re.I,
)
_STATUS = re.compile(
    r"\b(operational|working|functional|normal|no (?:issue|problem|error)s?|ok|okay"
    r"|pull.?out|pulled.?out|not operational|non.?operational)\b",
    re.I,
)


def classify_remark(line: str) -> str:
    """IMPROVEMENT 7: Returns recommendation | status | measurement | other."""
    # Measurement takes priority over other classes (a reading is not a recommendation)
    if any(pat.search(line) for pat in _READING_PATTERNS):
        return "measurement"
    if _MEASUREMENT_RE.search(line):
        return "measurement"
    words = re.findall(r"[a-z\-]+", line.lower())
    if (words and words[0] in _REC_FIRST) or _REC_ANY.search(line):
        return "recommendation"
    if _STATUS.search(line):
        return "status"
    return "other"


# ---------------------------------------------------------------------------
# whole report
# ---------------------------------------------------------------------------
def zone_text(zones: dict, name: str) -> str:
    """Text of one OCR zone. A zone flagged empty counts as blank."""
    z = (zones or {}).get(name) or {}
    if z.get("empty") is True:
        return ""
    t = z.get("text")
    if (t is None or t == "") and z.get("lines"):
        t = "\n".join(_line_text(x) for x in z["lines"])
    return t or ""


def extract_report(zones: dict, lexicons: dict | None = None) -> dict:
    """zones = the "zones" object of one /ocr page result."""
    lexicons = lexicons or {}

    def zt(name):
        return zone_text(zones, name)

    def lex(name):
        return lexicons[name] if name in lexicons else load_lexicon(name)

    fields = {
        "report_no":        normalize_report_no(zt("report_no")),
        "date_start":       normalize_date(zt("date_start")),
        "date_end":         normalize_date(zt("date_end")),
        "institution":      match_lexicon(zt("institution"), lex("institutions")),
        "equipment":        match_lexicon(zt("equipment"),   lex("equipment")),
        "serial_number":    normalize_serial(zt("serial_number")),
        "time_in":          normalize_time(zt("time_in")),
        "time_out":         normalize_time(zt("time_out")),
        "address":          normalize_free_text(zt("address")),
        # IMPROVEMENT 5: use multiline-aware normalizer for free-text fields
        "reported_problem": normalize_multiline_text(zt("reported_problem")),
        "cause_of_problem": normalize_multiline_text(zt("cause_of_problem")),
    }

    z_act = (zones or {}).get("actions_taken") or {}
    act_lines = (
        split_lines(z_act.get("lines") or zt("actions_taken"))
        if z_act.get("empty") is not True else []
    )
    actions = [{"text": ln, "tags": tag_action_line(ln)} for ln in act_lines]
    tag_counts: dict = {}
    for a in actions:
        for tg in a["tags"]:
            tag_counts[tg] = tag_counts.get(tg, 0) + 1

    z_rem = (zones or {}).get("remarks") or {}
    rem_lines = (
        split_lines(z_rem.get("lines") or zt("remarks"))
        if z_rem.get("empty") is not True else []
    )
    # IMPROVEMENT 7: remarks now carry 4 classes
    remarks = [{"text": ln, "kind": classify_remark(ln)} for ln in rem_lines]

    checks = []
    ds, de = fields["date_start"]["value"], fields["date_end"]["value"]
    ti, to = fields["time_in"], fields["time_out"]
    if ds and de and de < ds:
        checks.append("date_end_before_date_start")
    if (
        ds and ds == de
        and ti["value"] and to["value"]
        and "no_am_pm" not in ti["warnings"] + to["warnings"]
        and to["value"] <= ti["value"]
    ):
        checks.append("time_out_not_after_time_in")

    return {
        "fields":  fields,
        "actions": {
            "lines":    actions,
            "tag_counts": tag_counts,
            "unlabeled": sum(1 for a in actions if not a["tags"]),
        },
        "remarks": {
            "lines":       remarks,
            "follow_up":   any(r["kind"] == "recommendation" for r in remarks),
            "measurements": [r["text"] for r in remarks if r["kind"] == "measurement"],
        },
        "checks": checks,
    }
