"""Prompt construction for the future LLM-backed V4 nodes."""

from __future__ import annotations

from typing import Any

from .evidence_store import evidence_to_context


JUDGE_JSON_INSTRUCTIONS = """Return only valid JSON with these keys:
{
  \"enough\": boolean,
  \"missing_information\": string,
  \"next_query\": string,
  \"evidence_ids_used\": [string],
  \"reason\": string
}
If the evidence is insufficient, next_query must use an entity or fact from the evidence.
If it is sufficient, next_query must be an empty string."""


def build_judge_prompt(
    question: str,
    evidence: list[dict[str, Any]],
    search_history: list[str],
) -> str:
    """Build a data-leakage-free evidence sufficiency prompt."""
    history = "\n".join(f"- {query}" for query in search_history) or "- None"
    return (
        "Determine whether the retrieved evidence is sufficient to answer the original "
        "question. Do not use outside knowledge.\n\n"
        f"Original question:\n{question}\n\n"
        f"Previous search queries:\n{history}\n\n"
        f"Retrieved evidence:\n{evidence_to_context(evidence)}\n\n"
        f"{JUDGE_JSON_INSTRUCTIONS}"
    )