"""LLM clients for structured Agentic RAG decisions.

The Transformers client is lazy: importing this module does not download or
load a model. Model weights are loaded only on the first generation request.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMJsonError(RuntimeError):
    """Raised when a model response cannot be parsed into the requested schema."""


class TextGenerationClient(ABC):
    """Small provider-neutral interface used by the V4 Judge."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate text for one prompt."""


class TransformersLLMClient(TextGenerationClient):
    """Lazy Hugging Face causal-LM client for local paths or Hub model names."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading agent LLM: {self.model_name_or_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device_map,
                torch_dtype="auto",
                trust_remote_code=self.trust_remote_code,
            )
        return self._model, self._tokenizer

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        model, tokenizer = self._ensure_loaded()
        if getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": input_ids.to(model.device)}
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generation_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


class JsonLLMClient:
    """Convert a TextGenerationClient response into a validated Pydantic model."""

    def __init__(self, text_client: TextGenerationClient, max_retries: int = 1) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.text_client = text_client
        self.max_retries = max_retries

    @staticmethod
    def _extract_json_object(response: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(response):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise LLMJsonError("No JSON object was found in the LLM response")

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> T:
        current_prompt = prompt
        errors: list[str] = []
        for attempt in range(self.max_retries + 1):
            response = self.text_client.generate(
                current_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            try:
                return schema.model_validate(self._extract_json_object(response))
            except Exception as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                current_prompt = (
                    "Your previous response was invalid. Return only one JSON object that "
                    "matches the required schema.\n\nOriginal task:\n"
                    f"{prompt}\n\nPrevious response:\n{response}"
                )
        raise LLMJsonError("; ".join(errors))

class OpenAICompatibleLLMClient(TextGenerationClient):
    """Client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "OpenAI-compatible LLM request failed: "
                f"status={response.status_code}, body={response.text}"
            )
        payload = response.json()
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise LLMJsonError(
                f"Unexpected OpenAI-compatible response structure: {payload}"
            ) from exc