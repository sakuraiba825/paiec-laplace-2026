"""Conventional Jeffreys Beta(0.5,0.5) posterior-mean baseline.

    p = (0.5 + successes) / (1 + label_count)

Uses only revealed labels with matching subject attributes and benchmark_id.
Prior parameters are standard statistical reference choices, prospectively
fixed before this round, not fitted to private feedback. Interface handling
is reused from the project's historical flat-Beta implementation; the full
project has historical private-feedback analyses, which this round does not
use and which require separate organizer clarification. No API or training.
"""
from __future__ import annotations

import math
from typing import Any

KAPPA = 1.0
HALF = KAPPA / 2.0

SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
)


def _text(record: Any, field: str) -> str:
    if not isinstance(record, dict):
        return ""
    value = record.get(field, "")
    return "" if value is None else str(value)


def _subject_key(record: Any) -> tuple:
    return tuple(_text(record, field) for field in SUBJECT_FIELDS)


def _label(value: Any):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


def predict(input: list, labeled: list) -> float:
    """Return P(correct) for one subject-item target."""
    try:
        if not isinstance(input, (list, tuple)) or len(input) != 2:
            return 0.5
        subject, item = input
        if not isinstance(subject, dict):
            return 0.5
        key = _subject_key(subject)
        benchmark = _text(item, "benchmark_id")

        n = 0
        s = 0
        for entry in labeled or ():
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            pair, raw = entry
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            other_subject, other_item = pair
            if benchmark and _text(other_item, "benchmark_id") != benchmark:
                continue
            if _subject_key(other_subject) != key:
                continue
            label = _label(raw)
            if label is None:
                continue
            n += 1
            s += label

        p = (HALF + s) / (KAPPA + n)
        if not math.isfinite(p):
            return 0.5
        return float(min(1.0, max(0.0, p)))
    except Exception:
        return 0.5