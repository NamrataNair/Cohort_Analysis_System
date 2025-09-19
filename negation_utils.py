"""Utility helpers for negation and assertion detection.

The heuristics implemented here do not try to be a full blown
clinical NLP pipeline.  Instead, they provide lightweight and fast
rules that work well for the typical structured CSV exports used in
the cohort explorer.  The module exposes two pieces of functionality:

* ``detect_query_polarity`` – classifies a question into *affirmed*,
  *negated* or *neutral* buckets so that retrieval can prefer rows
  expressed with matching polarity.
* ``label_text_polarity`` – inspects a row converted to natural text
  and tags it with the same polarity labels.  The result is stored as
  metadata in the vector store and later used for filtered search.

The rules were inspired by common phrasing found in clinical
documentation (e.g. “denies chest pain”, “no history of”, “has a
diagnosis of”).  They are intentionally conservative: if both
assertive and negated cues are found the function returns ``"mixed"``
so that the retriever can still surface the chunk as a fall-back.
"""

from __future__ import annotations

import re
from typing import Iterable


NEGATION_KEYWORDS = {
    "no",
    "not",
    "denies",
    "without",
    "free of",
    "ruled out",
    "never",
    "negative for",
    "absent",
    "lack of",
    "resolved",
}

ASSERTION_KEYWORDS = {
    "has",
    "with",
    "history of",
    "diagnosed",
    "positive for",
    "reports",
    "experiencing",
    "complains of",
    "presents with",
    "found to have",
}


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase the incoming text."""

    return re.sub(r"\s+", " ", text).strip().lower()


def _keyword_count(text: str, keywords: Iterable[str]) -> int:
    """Return how many keyword occurrences are found in ``text``."""

    count = 0
    for kw in keywords:
        if kw in text:
            # Use a lightweight heuristic: count the keyword once per
            # sentence to avoid over-counting repeated mentions.
            count += len(re.findall(re.escape(kw), text))
    return count


def label_text_polarity(text: str) -> str:
    """Label the polarity of a text span.

    The return value is one of ``"affirmed"``, ``"negated"`` or
    ``"mixed"``.  ``"mixed"`` doubles as a neutral value when no clear
    cues are present.
    """

    normalized = _normalize(text)
    if not normalized:
        return "mixed"

    negated = _keyword_count(normalized, NEGATION_KEYWORDS)
    affirmed = _keyword_count(normalized, ASSERTION_KEYWORDS)

    if negated and not affirmed:
        return "negated"
    if affirmed and not negated:
        return "affirmed"
    return "mixed"


def detect_query_polarity(query: str) -> str:
    """Classify the intent of a user question by polarity.

    The categorisation mirrors :func:`label_text_polarity` so that the
    retriever can prefer matches with similar polarity.  When neither
    negation nor assertion cues are detected ``"mixed"`` is returned.
    """

    normalized = _normalize(query)
    if not normalized:
        return "mixed"

    for kw in NEGATION_KEYWORDS:
        if kw in normalized:
            return "negated"

    for kw in ASSERTION_KEYWORDS:
        if kw in normalized:
            return "affirmed"

    return "mixed"

