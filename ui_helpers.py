"""Helper utilities for the Streamlit user interface."""

from __future__ import annotations

from typing import Dict, Iterable, List


def build_ready_made_queries(columns: Iterable[str]) -> List[str]:
    columns = list(columns)
    suggestions: List[str] = [
        "Summarize the overall cohort composition.",
        "Highlight notable risk factors in this cohort.",
        "List recent admissions or encounters.",
    ]

    if not columns:
        return suggestions

    # Surface a couple of column-aware prompts.
    focus_candidates = columns[:4]
    for col in focus_candidates:
        suggestions.append(f"Show patients where {col} is unusual or extreme.")
    if "diagnosis" in "|".join(columns).lower():
        suggestions.append("Which diagnoses co-occur most frequently?")
    if "med" in "|".join(columns).lower():
        suggestions.append("What medications are commonly prescribed together?")
    return suggestions[:6]


def human_readable_size(num_bytes: int) -> str:
    """Convert the byte count into a human readable string."""

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < step:
            return f"{size:.2f} {unit}"
        size /= step
    return f"{size:.2f} PB"

