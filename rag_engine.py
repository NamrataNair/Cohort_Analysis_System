"""Negation aware retrieval augmented generation pipeline."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from negation_utils import detect_query_polarity
from user_vector_store import UserVectorStore


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a cohort analysis copilot.  Answer clinician questions using the
provided context.  Respect negations explicitly stated in the
documents.  If the answer is not present say so.

Structure the response as follows:

1. A concise natural language answer.
2. A bullet list titled "Key Findings" capturing relevant rows.
3. A "Sources" section that references each contributing row as
   `Row <row_index>` followed by a one sentence rationale.
            """,
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Suggest three follow-up questions that dig deeper into the dataset."
            "Focus on nuanced cohort splits, temporal changes, or risk" " signals."
            "Return them as a JSON list of strings.",
        ),
        (
            "human",
            "Dataset columns: {columns}\nLatest answer: {answer}",
        ),
    ]
)


class NegationAwareRAG:
    """High level orchestrator of the RAG workflow."""

    def __init__(self, store: UserVectorStore, generator_model: str = "mistral"):
        self.store = store
        self.generator = OllamaLLM(model=generator_model)

    # ------------------------------------------------------------------
    def _load_documents(
        self, user_id: str, query: str, k: int = 4
    ) -> Tuple[List[Document], str]:
        vectordb = self.store.get_vectordb(user_id)
        if vectordb is None:
            return [], "mixed"

        polarity = detect_query_polarity(query)
        documents: List[Document] = []

        def _search(filter_value: Optional[str], remaining: int) -> None:
            if remaining <= 0:
                return
            search_kwargs = {"k": remaining}
            if filter_value:
                search_kwargs["filter"] = {"polarity": filter_value}
            docs = vectordb.similarity_search(query, **search_kwargs)
            for doc in docs:
                if doc not in documents:
                    documents.append(doc)

        if polarity in {"affirmed", "negated"}:
            _search(polarity, k)
            if len(documents) < k:
                _search("mixed", k - len(documents))
        else:
            _search(None, k)

        return documents, polarity

    def _prepare_context(self, documents: List[Document]) -> str:
        context_parts = []
        for doc in documents:
            row_idx = doc.metadata.get("row_index", "?")
            context_parts.append(f"Row {row_idx}: {doc.page_content}")
        return "\n\n".join(context_parts)

    def answer_question(self, user_id: str, question: str, k: int = 4) -> Dict[str, object]:
        documents, polarity = self._load_documents(user_id, question, k=k)

        if not documents:
            return {
                "answer": "I do not have any data for this cohort yet. Upload a CSV to get started.",
                "documents": [],
                "polarity": polarity,
                "follow_ups": [],
            }

        context = self._prepare_context(documents)
        prompt = ANSWER_PROMPT.format(question=question, context=context)
        raw_answer = self.generator.invoke(prompt)

        cleaned_answer = raw_answer.replace("<think>", "").replace("</think>", "").strip()

        follow_ups = self._suggest_questions(user_id, cleaned_answer)

        return {
            "answer": cleaned_answer,
            "documents": documents,
            "polarity": polarity,
            "follow_ups": follow_ups,
        }

    def _suggest_questions(self, user_id: str, latest_answer: str) -> List[str]:
        summary = self.store.load_dataset_summary(user_id)
        if not summary:
            return []
        prompt = FOLLOW_UP_PROMPT.format(columns=", ".join(summary.get("columns", [])), answer=latest_answer)
        response = self.generator.invoke(prompt)
        cleaned = response.replace("<think>", "").replace("</think>", "").strip()
        try:
            suggestions = json.loads(cleaned)
            if isinstance(suggestions, list):
                return [str(item) for item in suggestions if item]
        except json.JSONDecodeError:
            pass
        return []

