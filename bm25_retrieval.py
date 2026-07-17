import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.schema import MetadataMode
from llama_index.text_splitter import SentenceSplitter

from util import JSONReader


class CustomExtractor(BaseExtractor):
    """保持和 simple_retrieval.py 相同的 metadata。"""

    async def aextract(self, nodes) -> List[Dict]:
        return [
            {
                "title": node.metadata["title"],
                "source": node.metadata["source"],
                "published_at": node.metadata["published_at"],
            }
            for node in nodes
        ]


def tokenize(text: str) -> List[str]:
    """
    MultiHop-RAG 是英文新闻语料。
    保留英文、数字，并统一转为小写。
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def get_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """避免对整个语料执行完整排序。"""
    top_k = min(top_k, len(scores))

    if top_k <= 0:
        return np.array([], dtype=int)

    if top_k == len(scores):
        return np.argsort(scores)[::-1]

    candidates = np.argpartition(scores, -top_k)[-top_k:]
    sorted_candidates = candidates[
        np.argsort(scores[candidates])[::-1]
    ]
    return sorted_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default="dataset/corpus.json",
    )
    parser.add_argument(
        "--queries",
        default="dataset/MultiHopRAG.json",
    )
    parser.add_argument(
        "--output",
        default="output/bm25_top50_chunk256.json",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first query to retrieve.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of queries to retrieve; omit to process all remaining queries.",
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 必须使用和 Dense Retriever 相同的切块方式。
    reader = JSONReader()
    documents = reader.load_data(args.corpus)

    text_splitter = SentenceSplitter(
        chunk_size=args.chunk_size
    )

    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            CustomExtractor(),
        ]
    )

    print("Splitting corpus into nodes...")
    nodes = pipeline.run(documents=documents)

    # 使用 MetadataMode.LLM，将 title、source、published_at 一并纳入 BM25。
    # 这也和原项目保存到 retrieval_list 的文本格式保持一致。
    node_texts = [
        node.get_content(metadata_mode=MetadataMode.LLM)
        for node in nodes
    ]

    print(f"Number of BM25 nodes: {len(node_texts)}")
    print("Tokenizing corpus...")

    tokenized_corpus = [
        tokenize(text)
        for text in tqdm(node_texts)
    ]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(args.queries, "r", encoding="utf-8") as file:
        query_data = json.load(file)

    if args.limit is not None:
        query_data = query_data[args.start:args.start + args.limit]
    else:
        query_data = query_data[args.start:]
    results = []

    print(f"Running BM25 retrieval on {len(query_data)} queries...")

    for item in tqdm(query_data):
        query = item["query"]

        scores = np.asarray(
            bm25.get_scores(tokenize(query)),
            dtype=np.float32,
        )

        top_indices = get_top_k_indices(
            scores=scores,
            top_k=args.topk,
        )

        retrieval_list = [
            {
                "text": node_texts[index],
                "score": float(scores[index]),
            }
            for index in top_indices
        ]

        results.append(
            {
                "query": query,
                "answer": item["answer"],
                "question_type": item["question_type"],
                "retrieval_list": retrieval_list,
                "gold_list": item["evidence_list"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved BM25 results to: {output_path}")


if __name__ == "__main__":
    main()