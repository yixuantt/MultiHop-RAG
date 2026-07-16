"""LangGraph-compatible state for one independent MultiHop-RAG question."""

from __future__ import annotations

import operator
from typing import Any, Annotated, Literal, NotRequired, TypedDict


class AgentState(TypedDict):
    """Shared state read and updated by LangGraph nodes.

    Ground-truth fields such as answer, evidence_list and question_type are
    intentionally absent. They are attached only after graph execution for
    offline evaluation, preventing benchmark data leakage.
    """

    original_question: str
    current_query: str
    hop: int
    max_hops: int
    max_answer_retries: int
    answer_retry_count: int
    retrieve_top_k: int
    rrf_top_k: int
    rerank_top_n: int
    max_evidence_documents: int
    status: Literal[
        "running",
        "ready_to_answer",
        "max_hops_reached",
        "insufficient_evidence",
        "completed",
        "failed",
    ]
    evidence: list[dict[str, Any]]
    hop_traces: Annotated[list[dict[str, Any]], operator.add]
    search_history: Annotated[list[str], operator.add]
    missing_information: str
    final_answer: str
    answer_citations: list[str]
    errors: Annotated[list[str], operator.add]
    latency: dict[str, float]
    latest_retrieved_documents: NotRequired[list[dict[str, Any]]]
    latest_retrieval_seconds: NotRequired[float]
    latest_new_evidence_count: NotRequired[int]
    next_query_is_valid: NotRequired[bool]
    judge_decision: NotRequired[dict[str, Any]]
    verification: NotRequired[dict[str, Any]]


def make_initial_state(
    question: str,
    max_hops: int = 2,
    max_answer_retries: int = 1,
    retrieve_top_k: int = 10,
    rrf_top_k: int = 10,
    rerank_top_n: int = 5,
    max_evidence_documents: int = 6,
) -> AgentState:
    """Create a clean per-question state without benchmark-only labels."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if max_hops <= 0:
        raise ValueError("max_hops must be a positive integer")
    if max_answer_retries < 0:
        raise ValueError("max_answer_retries must be non-negative")
    if min(retrieve_top_k, rrf_top_k, rerank_top_n, max_evidence_documents) <= 0:
        raise ValueError("retrieval and evidence limits must be positive")

    return {
        "original_question": question,
        "current_query": question,
        "hop": 0,
        "max_hops": max_hops,
        "max_answer_retries": max_answer_retries,
        "answer_retry_count": 0,
        "retrieve_top_k": retrieve_top_k,
        "rrf_top_k": rrf_top_k,
        "rerank_top_n": rerank_top_n,
        "max_evidence_documents": max_evidence_documents,
        "status": "running",
        "evidence": [],
        "hop_traces": [],
        "search_history": [],
        "missing_information": "",
        "final_answer": "",
        "answer_citations": [],
        "errors": [],
        "latency": {},
    }