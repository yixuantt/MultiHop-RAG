"""Evidence merge, deduplication and prompt-context helpers for V4."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for stable document identity keys."""
    return " ".join(text.lower().split())


def evidence_key(document: dict[str, Any]) -> tuple[str, str, str]:
    """Return a metadata-aware identity key for one retrieved document."""
    return (
        str(document.get("title") or "").strip().lower(),
        str(document.get("source") or "").strip().lower(),
        normalize_text(str(document.get("text") or "")),
    )


def _rank_score(document: dict[str, Any]) -> float:
    """Use reranker score when available, then fall back to generic score."""
    value = document.get("rerank_score", document.get("score", float("-inf")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def merge_evidence(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    hop: int,
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Merge a retrieval hop into evidence, preserving IDs and origin hops."""
    if hop <= 0:
        raise ValueError("hop must be a positive integer")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")

    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    next_id = 1

    for document in existing:
        item = deepcopy(document)
        key = evidence_key(item)
        if key in merged_by_key:
            continue
        evidence_id = item.get("evidence_id") or f"E{next_id}"
        item["evidence_id"] = evidence_id
        if isinstance(evidence_id, str) and evidence_id.startswith("E"):
            try:
                next_id = max(next_id, int(evidence_id[1:]) + 1)
            except ValueError:
                pass
        item["retrieved_hops"] = list(dict.fromkeys(item.get("retrieved_hops", [])))
        merged_by_key[key] = item
        ordered_keys.append(key)

    added_count = 0
    for document in incoming:
        item = deepcopy(document)
        key = evidence_key(item)
        if not key[2]:
            continue
        if key not in merged_by_key:
            item["evidence_id"] = f"E{next_id}"
            next_id += 1
            item["retrieved_hops"] = [hop]
            merged_by_key[key] = item
            ordered_keys.append(key)
            added_count += 1
            continue

        current = merged_by_key[key]
        hops = list(dict.fromkeys(current.get("retrieved_hops", []) + [hop]))
        if _rank_score(item) > _rank_score(current):
            item["evidence_id"] = current["evidence_id"]
            item["retrieved_hops"] = hops
            merged_by_key[key] = item
        else:
            current["retrieved_hops"] = hops

    merged = [merged_by_key[key] for key in ordered_keys]
    merged.sort(key=lambda document: (-_rank_score(document), document["evidence_id"]))
    if max_documents is not None:
        merged = merged[:max_documents]
    return merged, added_count


def evidence_to_context(evidence: list[dict[str, Any]]) -> str:
    """Format evidence with stable citation IDs for Judge and answer prompts."""
    parts = []
    for document in evidence:
        evidence_id = document.get("evidence_id", "E?")
        title = document.get("title") or "Untitled"
        source = document.get("source") or "Unknown source"
        text = document.get("text") or ""
        parts.append(f"[{evidence_id}] Title: {title}\nSource: {source}\n{text}")
    return "\n\n--------------\n\n".join(parts)