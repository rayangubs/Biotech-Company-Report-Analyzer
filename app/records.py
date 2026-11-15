"""Confirmed-record store.

When a record is saved via POST /records the NLP extraction pipeline runs
automatically so every stored record always contains a normalised ``nlp``
sub-object alongside the raw ``data`` from the review form.

Mimics a Bitrix 'create' call; swap the body of create_record() for a real
Bitrix REST push if SBSI grants API access.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .store import append_record, load_records, read_records

router = APIRouter()


# ---------------------------------------------------------------------------
# NLP enrichment
# ---------------------------------------------------------------------------

def _enrich(rec: dict) -> dict:
    """Run the NLP normalisation on the confirmed data fields and attach the
    result under rec["nlp"].  Any error is caught so the record is always saved."""
    try:
        from .nlp.normalize import extract_report
        zones = _data_to_zones(rec.get("data") or {})
        rec["nlp"] = extract_report(zones)
    except Exception as exc:  # noqa: BLE001
        rec["nlp"] = {"error": str(exc)}
    return rec


def _data_to_zones(data: dict) -> dict:
    """Convert the flat review-form data dict into the zones format that
    extract_report() expects ({"zone_name": {"text": "...", "empty": bool}})."""
    from .nlp.normalize import is_empty as _is_empty
    mapping = {
        "report_no":        data.get("reportNo", ""),
        "date_start":       data.get("dateStart", ""),
        "date_end":         data.get("dateEnd", ""),
        "institution":      data.get("institution", ""),
        "address":          data.get("address", ""),
        "equipment":        data.get("equipment", ""),
        "serial_number":    data.get("serial", ""),
        "time_in":          data.get("timeIn", ""),
        "time_out":         data.get("timeOut", ""),
        "reported_problem": data.get("reportedProblem", ""),
        "cause_of_problem": data.get("causeOfProblem", ""),
        "actions_taken":    data.get("actionsTaken", ""),
        "remarks":          data.get("remarks", ""),
    }
    zones = {}
    for k, v in mapping.items():
        text = str(v).strip()
        zones[k] = {"text": text, "empty": _is_empty(text)}
    return zones


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.post("/records")
def create_record(rec: dict) -> dict:
    """Accept a confirmed record, run NLP enrichment, persist to JSONL."""
    record_id = str(rec.get("id", "")).strip()
    if not record_id:
        raise HTTPException(400, "Record must include a non-empty 'id' field")
    rec = _enrich(rec)
    append_record(rec)
    return {"ok": True, "id": rec.get("id"), "nlp_ok": "error" not in rec.get("nlp", {})}


@router.get("/records")
def list_records() -> list:
    return load_records()


@router.get("/records/{record_id}")
def get_record(record_id: str) -> dict:
    store = read_records()
    # Try exact match first, then case/space-insensitive
    if record_id in store:
        return store[record_id]
    normalised = record_id.upper().replace(" ", "")
    for k, v in store.items():
        if k.upper().replace(" ", "") == normalised:
            return v
    raise HTTPException(404, f"Record '{record_id}' not found")
