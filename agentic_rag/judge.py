"""LLM-backed Evidence Judge adapter for the V4 LangGraph."""

from __future__ import annotations

from .llm_client import JsonLLMClient
from .prompts import build_judge_prompt
from .schemas import JudgeDecision


class LLMEvidenceJudge:
    """Callable Judge that returns validated decisions from a JSON-capable LLM."""

    def __init__(
        self,
        llm: JsonLLMClient,
        *,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(
        self,
        question: str,
        evidence: list[dict],
        search_history: list[str],
    ) -> JudgeDecision:
        prompt = build_judge_prompt(
            question=question,
            evidence=evidence,
            search_history=search_history,
        )
        return self.llm.generate_json(
            prompt,
            JudgeDecision,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )