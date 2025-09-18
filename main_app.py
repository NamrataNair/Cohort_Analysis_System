"""Streamlit entry point for the Cohort Analysis System."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from rag_engine import NegationAwareRAG
from ui_helpers import build_ready_made_queries, human_readable_size
from user_vector_store import UserVectorStore


st.set_page_config(page_title="LLM Cohort Copilot", layout="wide")


def _init_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = UserVectorStore()
    if "rag" not in st.session_state:
        st.session_state.rag = NegationAwareRAG(st.session_state.store)
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _render_sidebar() -> str:
    st.sidebar.title("Cohort Controls")
    user_id = st.sidebar.text_input("User / Workspace ID", help="All vector stores are isolated per user.")

    if not user_id:
        st.sidebar.info("Enter your workspace identifier to continue.")
        return ""

    report = st.session_state.store.get_usage_report(user_id)
    st.sidebar.metric(
        "Storage usage",
        human_readable_size(report["used_bytes"]),
        help=f"Quota: {human_readable_size(report['quota_bytes'])} ({report['used_percent']}% used)",
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload cohort CSV", type=["csv"], help="Rows are embedded into a private vector index for this user only."
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            info = st.session_state.store.ingest_dataframe(user_id, df)
            st.sidebar.success(
                f"Ingested {info['rows']} rows into your personal vector database. The index persists on disk."
            )
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(str(exc))

    summary = st.session_state.store.load_dataset_summary(user_id)
    if summary:
        st.sidebar.subheader("Dataset snapshot")
        st.sidebar.write(f"Rows embedded: **{summary['row_count']}**")
        st.sidebar.caption(f"Columns: {', '.join(summary['columns'])}")

    return user_id


def _render_ready_queries(columns: List[str]) -> None:
    st.subheader("Jump start your analysis")
    suggestions = build_ready_made_queries(columns)
    cols = st.columns(len(suggestions)) if suggestions else []
    for idx, query in enumerate(suggestions):
        button = cols[idx].button(query, use_container_width=True)
        if button:
            st.session_state.pending_query = query


def _display_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("follow_ups"):
                    st.markdown("**Suggested questions**")
                    follow_cols = st.columns(len(message["follow_ups"]))
                    for idx, text in enumerate(message["follow_ups"]):
                        if follow_cols[idx].button(text, key=f"follow_{len(st.session_state.messages)}_{idx}"):
                            st.session_state.pending_query = text
                if message.get("sources"):
                    with st.expander("Sources from cohort dataset", expanded=False):
                        st.dataframe(pd.DataFrame(message["sources"]))


def _ingest_sources(documents: List[Any]) -> List[Dict[str, Any]]:
    sources = []
    for doc in documents:
        row_meta = {"row_index": doc.metadata.get("row_index"), "polarity": doc.metadata.get("polarity")}
        row_json = doc.metadata.get("row_json")
        if row_json:
            try:
                row_data = json.loads(row_json)
                row_meta.update(row_data)
            except json.JSONDecodeError:
                row_meta["raw_row"] = row_json
        sources.append(row_meta)
    return sources


def _handle_user_prompt(user_id: str, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = st.session_state.rag.answer_question(user_id, prompt)
    sources = _ingest_sources(response["documents"])
    assistant_message = {
        "role": "assistant",
        "content": response["answer"],
        "sources": sources,
        "follow_ups": response.get("follow_ups", []),
    }
    st.session_state.messages.append(assistant_message)


def main() -> None:
    _init_state()
    st.title("Cohort Analysis Copilot")
    st.caption("An Ollama powered, negation aware RAG workspace with personalised vector stores.")

    user_id = _render_sidebar()
    if not user_id:
        return

    summary = st.session_state.store.load_dataset_summary(user_id)
    _render_ready_queries(summary.get("columns", []) if summary else [])

    _display_messages()

    pending = st.session_state.pending_query
    if pending:
        st.session_state.pending_query = None
        _handle_user_prompt(user_id, pending)
        st.experimental_rerun()

    prompt = st.chat_input("Ask a cohort intelligence question…")
    if prompt:
        _handle_user_prompt(user_id, prompt)
        st.experimental_rerun()


if __name__ == "__main__":
    main()

