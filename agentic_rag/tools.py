"""Adapters from V4 graph nodes to existing project tools."""

from __future__ import annotations

from typing import Any


def call_v3_hybrid_retriever(
    query: str,
    retrieve_top_k: int,
    rrf_top_k: int,
    rerank_top_n: int,
) -> list[dict[str, Any]]:
    """Lazily call the existing V3 retriever without importing it at graph load."""
    from hybrid_retriever import hybrid_retrieve

    return hybrid_retrieve(
        query=query,
        retrieve_top_k=retrieve_top_k,
        rrf_top_k=rrf_top_k,
        rerank_top_n=rerank_top_n,
    )