


import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM answers from retrieval-result JSON using DashScope OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        default="output/hybrid_query_result.json",
        help="Retrieval-result JSON with query, answer, question_type and retrieval_list.",
    )
    parser.add_argument(
        "--output",
        default="qa_output/hybrid_qwen.json",
        help="JSON file for generated answers.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DashScope LLM model name. If omitted, use DASHSCOPE_LLM_MODEL from .env.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        default=True,
        help="Disable Qwen thinking mode when the API supports enable_thinking=False.",
    )
    return parser.parse_args()


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is not set. Please add it to your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def normalize_input_data(doc_data: Any) -> List[Dict[str, Any]]:
    """
    Accept both:
    1. a list of retrieval results
    2. a single retrieval result dict
    """
    if isinstance(doc_data, list):
        return doc_data

    if isinstance(doc_data, dict):
        return [doc_data]

    raise ValueError(
        "Input JSON must be either a list of retrieval results or a single retrieval result object."
    )


def build_prompt(item: Dict[str, Any]) -> str:
    prefix = (
        "Below is a question followed by some context from different sources. "
        "Please answer the question based on the context. The answer to the "
        "question is a word or entity. If the provided information is "
        "insufficient to answer the question, respond 'Insufficient Information'. "
        "Answer directly without explanation."
    )

    retrieval_list = item.get("retrieval_list", [])
    context = "\n\n--------------\n\n".join(
        evidence.get("text", "")
        for evidence in retrieval_list
    )

    query = item["query"]

    prompt = (
        f"{prefix}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n\n{context}"
    )

    return prompt


def generate_answer(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    disable_thinking: bool = True,
) -> str:
    extra_body = {}

    if disable_thinking:
        extra_body["enable_thinking"] = False

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        max_tokens=max_new_tokens,
        extra_body=extra_body if extra_body else None,
    )

    return response.choices[0].message.content.strip()


def main() -> None:
    load_dotenv()

    args = parse_args()

    model_name = args.model or os.getenv("DASHSCOPE_LLM_MODEL")

    if not model_name:
        raise ValueError(
            "LLM model is not set. Please add DASHSCOPE_LLM_MODEL=qwen3.7-max to .env "
            "or pass --model qwen3.7-max."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Retrieval result file was not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        raw_doc_data = json.load(file)

    doc_data = normalize_input_data(raw_doc_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"DashScope LLM: {model_name}")
    print(f"Number of questions: {len(doc_data)}")

    client = build_client()

    save_list = []

    for index, item in enumerate(tqdm(doc_data, desc="Generating answers"), start=1):
        prompt = build_prompt(item)

        print(f"Generating answer for query {index}/{len(doc_data)}...")

        response = generate_answer(
            client=client,
            model_name=model_name,
            prompt=prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=args.disable_thinking,
        )

        save_list.append(
            {
                "query": item["query"],
                "prompt": prompt,
                "model_answer": response,
                "gold_answer": item.get("answer"),
                "question_type": item.get("question_type"),
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(save_list, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(save_list)} model answers to: {output_path}")


if __name__ == "__main__":
    main()

"""
python qa_llama.py `
  --input output/hybrid_query_result.json `
  --output qa_output/hybrid_qwen.json


python qa_evaluate.py --file qa_output/hybrid_qwen.json
"""