"""Validated payloads exchanged by future V4 graph nodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JudgeDecision(BaseModel):
    """Structured result from the evidence-sufficiency judge."""

    enough: bool
    missing_information: str = ""
    next_query: str = ""
    evidence_ids_used: list[str] = Field(default_factory=list)
    reason: str = ""


class VerificationResult(BaseModel):
    """Structured result from the answer verifier."""

    supported: bool
    complete: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    suggested_action: Literal["accept", "regenerate"] = "accept"
    reason: str = ""


class HopTrace(BaseModel):
    """One observable retrieval-and-decision step in an agent run."""

    hop: int
    search_query: str
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    new_evidence_count: int = 0
    decision: JudgeDecision | None = None
    retrieval_seconds: float = 0.0