"""Node factories for the V4.0 two-hop retrieval graph."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from .evidence_store import merge_evidence, normalize_text
from .schemas import HopTrace, JudgeDecision
from .state import AgentState
from .tools import call_v3_hybrid_retriever

RetrieveFn = Callable[[str, int, int, int], list[dict[str, Any]]]
JudgeFn = Callable[[str, list[dict[str, Any]], list[str]], JudgeDecision | dict[str, Any]]


class V4Nodes:
    """Factories whose bound methods are LangGraph node functions."""

    def __init__(
        self,
        judge: JudgeFn,
        retriever: RetrieveFn = call_v3_hybrid_retriever,
    ) -> None:
        self.judge = judge
        self.retriever = retriever

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        """Execute one V3 Hybrid Retrieval hop."""
        hop = state["hop"] + 1
        query = state["current_query"]
        started = perf_counter()
        try:
            documents = self.retriever(
                query,
                state["retrieve_top_k"],
                state["rrf_top_k"],
                state["rerank_top_n"],
            )
            errors: list[str] = []
        except Exception as exc:  # graph records retrieval failures rather than crashing silently
            documents = []
            errors = [f"Hop {hop} retrieval failed: {exc}"]
        elapsed = perf_counter() - started

        latency = dict(state["latency"])
        latency["retrieval_seconds"] = latency.get("retrieval_seconds", 0.0) + elapsed
        return {
            "hop": hop,
            "latest_retrieved_documents": documents,
            "latest_retrieval_seconds": elapsed,
            "search_history": [query],
            "errors": errors,
            "latency": latency,
        }

    def merge_evidence(self, state: AgentState) -> dict[str, Any]:
        """Deduplicate current-hop documents into the state evidence store."""
        evidence, added_count = merge_evidence(
            existing=state["evidence"],
            incoming=state.get("latest_retrieved_documents", []),
            hop=state["hop"],
            max_documents=state["max_evidence_documents"],
        )
        return {"evidence": evidence, "latest_new_evidence_count": added_count}

    def judge_evidence(self, state: AgentState) -> dict[str, Any]:
        """Ask an injected structured judge whether another hop is needed."""
        try:
            raw_decision = self.judge(
                state["original_question"],
                state["evidence"],
                state["search_history"],
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, JudgeDecision)
                else JudgeDecision.model_validate(raw_decision)
            )
            errors: list[str] = []
        except Exception as exc:
            decision = JudgeDecision(
                enough=False,
                reason="Judge failed; stopping safely.",
            )
            errors = [f"Hop {state['hop']} judge failed: {exc}"]

        trace = HopTrace(
            hop=state["hop"],
            search_query=state["current_query"],
            retrieved_documents=state.get("latest_retrieved_documents", []),
            new_evidence_count=state.get("latest_new_evidence_count", 0),
            decision=decision,
            retrieval_seconds=state.get("latest_retrieval_seconds", 0.0),
        )
        return {
            "judge_decision": decision.model_dump(),
            "missing_information": decision.missing_information,
            "hop_traces": [trace.model_dump()],
            "errors": errors,
        }

    def plan_next_query(self, state: AgentState) -> dict[str, Any]:
        """Validate the Judge-proposed next query before another retrieval hop."""
        decision = JudgeDecision.model_validate(state["judge_decision"])
        next_query = decision.next_query.strip()
        searched = {normalize_text(query) for query in state["search_history"]}
        is_valid = bool(next_query) and normalize_text(next_query) not in searched
        if not is_valid:
            return {
                "next_query_is_valid": False,
                "errors": ["Judge did not produce a new, non-duplicate next query."],
            }
        return {"current_query": next_query, "next_query_is_valid": True}

    @staticmethod
    def finish_ready(state: AgentState) -> dict[str, Any]:
        return {"status": "ready_to_answer"}

    @staticmethod
    def finish_max_hops(state: AgentState) -> dict[str, Any]:
        return {"status": "max_hops_reached"}

    @staticmethod
    def finish_insufficient(state: AgentState) -> dict[str, Any]:
        return {"status": "insufficient_evidence"}