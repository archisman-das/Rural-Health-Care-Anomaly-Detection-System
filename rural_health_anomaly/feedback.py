"""Clinician feedback capture and retraining helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_FEEDBACK_META_KEYS = {
    "is_true_positive",
    "feedback_label",
    "reviewer",
    "clinician_id",
    "reviewed_at",
    "notes",
    "feedback_notes",
    "comment",
    "alert_id",
    "prediction",
    "patient",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_feedback_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(bool(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "tp", "true_positive"}:
            return 1
        if text in {"0", "false", "no", "n", "fp", "false_positive"}:
            return 0
    raise ValueError("Feedback records must include a binary true-positive/false-positive label.")


def normalize_feedback_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a clinician feedback payload into a ledger row."""

    if "patient" in payload and isinstance(payload["patient"], dict):
        raw_record = dict(payload["patient"])
    else:
        raw_record = {key: value for key, value in payload.items() if key not in _FEEDBACK_META_KEYS}

    if not raw_record:
        raise ValueError("Feedback payload must include a patient record.")

    label_source = payload.get("is_true_positive", payload.get("feedback_label"))
    if label_source is None:
        raise ValueError("Feedback payload must include 'is_true_positive' or 'feedback_label'.")

    label = _coerce_feedback_label(label_source)
    reviewed_at = str(payload.get("reviewed_at") or _now_iso())

    alert_id = payload.get("alert_id")
    if alert_id is None:
        alert_id = raw_record.get("alert_id")
    if alert_id is None:
        alert_id = raw_record.get("patient_id") or raw_record.get("record_id") or raw_record.get("sample_id")

    reviewer = payload.get("reviewer") or payload.get("clinician_id")
    notes = payload.get("notes") or payload.get("feedback_notes") or payload.get("comment")
    prediction = payload.get("prediction")

    return {
        "alert_id": alert_id,
        "patient_id": raw_record.get("patient_id"),
        "recorded_at": raw_record.get("recorded_at"),
        "feedback_label": int(label),
        "feedback_label_text": "true_positive" if label == 1 else "false_positive",
        "is_true_positive": bool(label),
        "reviewed_at": reviewed_at,
        "reviewer": reviewer,
        "notes": notes,
        "prediction_json": json.dumps(prediction, ensure_ascii=False, sort_keys=True) if prediction is not None else None,
        "record_json": json.dumps(raw_record, ensure_ascii=False, sort_keys=True),
    }


def append_feedback_records(path: str | Path, records: list[dict[str, Any]]) -> int:
    """Append clinician feedback records to a JSONL ledger."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_feedback_record(record) for record in records]
    with output_path.open("a", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return len(normalized)


def load_feedback_ledger(path: str | Path) -> pd.DataFrame:
    """Load a feedback ledger from JSONL."""

    input_path = Path(path)
    if not input_path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if "record_json" not in record and ("patient" in record or "feedback_label" in record or "is_true_positive" in record):
                record = normalize_feedback_record(record)
            rows.append(record)
    return pd.DataFrame(rows)


def feedback_to_training_frame(feedback_ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Materialize clinician feedback rows back into training features and labels."""

    if feedback_ledger.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    records: list[dict[str, Any]] = []
    labels: list[int] = []
    for _, row in feedback_ledger.iterrows():
        record_json = row.get("record_json")
        if not isinstance(record_json, str) or not record_json.strip():
            continue
        record = json.loads(record_json)
        records.append(record)
        labels.append(int(row.get("feedback_label", 0)))

    if not records:
        return pd.DataFrame(), pd.Series(dtype=int)

    frame = pd.DataFrame(records)
    return frame, pd.Series(labels, dtype=int)


def build_retraining_dataset(
    base_data: pd.DataFrame,
    feedback_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine raw base data with clinician feedback for periodic retraining."""

    feedback_frame, feedback_labels = feedback_to_training_frame(feedback_ledger)
    if feedback_frame.empty:
        return base_data.copy(), pd.Series([0] * len(base_data), dtype=int)

    combined = pd.concat([base_data.copy(), feedback_frame], ignore_index=True, sort=False)
    labels = pd.concat([pd.Series([0] * len(base_data), dtype=int), feedback_labels], ignore_index=True)
    return combined, labels
