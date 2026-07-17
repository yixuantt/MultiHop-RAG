"""LangGraph V4.0: retrieve, merge evidence, judge, and optional second hop."""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from .judge import LLMEvidenceJudge
from .llm_client import (
    JsonLLMClient,
    OpenAICompatibleLLMClient,
    TransformersLLMClient,
)
from .nodes import JudgeFn, RetrieveFn, V4Nodes
from .schemas import JudgeDecision
from .state import AgentState
from .tools import call_v3_hybrid_retriever


def _route_after_judge(
    state: AgentState,
) -> Literal["plan_next_query", "finish_ready", "finish_max_hops", "finish_insufficient"]:
    decision = JudgeDecision.model_validate(state["judge_decision"])
    if decision.enough:
        return "finish_ready"
    if state["hop"] >= state["max_hops"]:
        return "finish_max_hops"
    if not decision.next_query.strip():
        return "finish_insufficient"
    return "plan_next_query"


def _route_after_planning(
    state: AgentState,
) -> Literal["retrieve", "finish_insufficient"]:
    return "retrieve" if state.get("next_query_is_valid") else "finish_insufficient"


def build_v4_retrieval_graph(
    judge: JudgeFn,
    retriever: RetrieveFn = call_v3_hybrid_retriever,
):
    """Compile the V4.0 retrieval graph with injected V3 tool and Judge."""
    nodes = V4Nodes(judge=judge, retriever=retriever)
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("merge_evidence", nodes.merge_evidence)
    builder.add_node("judge_evidence", nodes.judge_evidence)
    builder.add_node("plan_next_query", nodes.plan_next_query)
    builder.add_node("finish_ready", nodes.finish_ready)
    builder.add_node("finish_max_hops", nodes.finish_max_hops)
    builder.add_node("finish_insufficient", nodes.finish_insufficient)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "merge_evidence")
    builder.add_edge("merge_evidence", "judge_evidence")
    builder.add_conditional_edges("judge_evidence", _route_after_judge)
    builder.add_conditional_edges("plan_next_query", _route_after_planning)
    builder.add_edge("finish_ready", END)
    builder.add_edge("finish_max_hops", END)
    builder.add_edge("finish_insufficient", END)
    return builder.compile()


def build_transformers_v4_retrieval_graph(
    model_name_or_path: str,
    *,
    retriever: RetrieveFn = call_v3_hybrid_retriever,
    device_map: str = "auto",
    trust_remote_code: bool = False,
    json_retries: int = 1,
):
    """Build V4.0 graph with a lazy local/Hugging Face structured Judge."""
    text_client = TransformersLLMClient(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    judge = LLMEvidenceJudge(JsonLLMClient(text_client, max_retries=json_retries))
    return build_v4_retrieval_graph(judge=judge, retriever=retriever)


def build_dashscope_v4_retrieval_graph(
    *,
    retriever: RetrieveFn = call_v3_hybrid_retriever,
    json_retries: int = 1,
):
    """Build V4.0 graph from DASHSCOPE_* values in the project .env file.

    Required variables are DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL and
    DASHSCOPE_LLM_MODEL. The key remains in the environment and is never
    written to graph state, output traces or logs.
    """
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("DASHSCOPE_LLM_MODEL", "")
    missing = [
        name
        for name, value in {
            "DASHSCOPE_API_KEY": api_key,
            "DASHSCOPE_BASE_URL": base_url,
            "DASHSCOPE_LLM_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required .env configuration: " + ", ".join(missing)
        )

    text_client = OpenAICompatibleLLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    judge = LLMEvidenceJudge(JsonLLMClient(text_client, max_retries=json_retries))
    return build_v4_retrieval_graph(judge=judge, retriever=retriever)