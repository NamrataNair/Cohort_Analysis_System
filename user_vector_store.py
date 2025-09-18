"""User scoped Chroma vector store management.

Each customer of the cohort explorer receives an isolated vector
database that persists on disk under ``user_vectordbs/<user_id>``.
Collections are never deleted which satisfies the "no VDB gets
deleted once made" requirement.  A soft quota keeps storage usage per
user under control.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from negation_utils import label_text_polarity


def _sanitize_identifier(raw: str) -> str:
    """Return a filesystem safe identifier based on ``raw``."""

    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw.strip())
    return cleaned or "anonymous"


def _format_row(row: pd.Series) -> str:
    """Create a paragraph style string out of a dataframe row."""

    values = []
    for column, value in row.items():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        values.append(f"{column}: {text}")
    return "\n".join(values)


def _directory_size(path: Path) -> int:
    """Calculate the size of a directory recursively in bytes."""

    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _estimate_documents_size(documents: Iterable[Document]) -> int:
    """Rough byte estimate for documents that will be stored."""

    total = 0
    for doc in documents:
        total += len(doc.page_content.encode("utf-8"))
        for value in doc.metadata.values():
            total += len(str(value).encode("utf-8"))
    return total


@dataclass
class UserQuota:
    limit_mb: int = 50

    @property
    def limit_bytes(self) -> int:
        return self.limit_mb * 1024 * 1024


class UserVectorStore:
    """Manage per-user Chroma collections backed by Ollama embeddings."""

    def __init__(self, base_dir: str = "user_vectordbs", quota: Optional[UserQuota] = None):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.quota = quota or UserQuota()
        self._embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_collection_path(self, user_id: str) -> Path:
        return self.base_path / _sanitize_identifier(user_id)

    def get_vectordb(self, user_id: str) -> Optional[Chroma]:
        path = self.get_collection_path(user_id)
        if not path.exists():
            return None
        return Chroma(
            collection_name=_sanitize_identifier(user_id),
            persist_directory=str(path),
            embedding_function=self._embeddings,
        )

    def ingest_dataframe(self, user_id: str, df: pd.DataFrame) -> Dict[str, str]:
        """Persist the dataframe as documents for the user.

        Returns basic metadata (row_count and polarity distribution)
        that can be used by the UI.
        """

        if df.empty:
            raise ValueError("The uploaded dataframe is empty.")

        path = self.get_collection_path(user_id)
        path.mkdir(parents=True, exist_ok=True)

        documents: List[Document] = []
        polarity_counter = {"affirmed": 0, "negated": 0, "mixed": 0}

        for idx, row in df.iterrows():
            content = _format_row(row)
            if not content:
                continue
            polarity = label_text_polarity(content)
            polarity_counter[polarity] += 1

            doc = Document(
                page_content=content,
                metadata={
                    "row_index": int(idx),
                    "polarity": polarity,
                    "row_json": json.dumps(row.fillna("").to_dict()),
                },
            )
            documents.append(doc)

        if not documents:
            raise ValueError("No usable rows were found in the dataset.")

        estimated_addition = _estimate_documents_size(documents)
        current_size = _directory_size(path)
        if current_size + estimated_addition > self.quota.limit_bytes:
            raise ValueError(
                "Quota exceeded – remove data from the dataset or request a larger allowance."
            )

        vectordb = Chroma(
            collection_name=_sanitize_identifier(user_id),
            persist_directory=str(path),
            embedding_function=self._embeddings,
        )

        vectordb.add_documents(documents)
        vectordb.persist()

        summary_payload = {
            "row_count": len(documents),
            "polarity": polarity_counter,
            "columns": list(df.columns),
        }
        summary_path = path / "dataset_summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2))

        return {"rows": str(len(documents)), "path": str(path)}

    def get_usage_report(self, user_id: str) -> Dict[str, str]:
        path = self.get_collection_path(user_id)
        current_size = _directory_size(path)
        percent = (current_size / self.quota.limit_bytes) * 100 if self.quota.limit_bytes else 0
        return {
            "used_bytes": current_size,
            "quota_bytes": self.quota.limit_bytes,
            "used_percent": round(percent, 2),
            "exists": path.exists(),
        }

    def load_dataset_summary(self, user_id: str) -> Optional[Dict[str, object]]:
        path = self.get_collection_path(user_id) / "dataset_summary.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

