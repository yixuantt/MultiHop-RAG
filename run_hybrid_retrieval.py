"""Command-line entry point for the reusable hybrid retrieval tool."""

import argparse
import json
from pathlib import Path

from hybrid_retriever import (
    configure_default_retriever,
    hybrid_retrieve,
    hybrid_retrieve_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dense + BM25 + RRF + BGE hybrid retrieval."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--query",
        help="One custom query. This mode prints/saves only a document list.",
    )
    input_group.add_argument(
        "--queries",
        help="MultiHopRAG-style JSON file. This mode creates an evaluable result file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based first query record in --queries (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of original query records to process from --queries.",
    )
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--rrf_top_k", type=int, default=10)
    parser.add_argument("--rerank_top_n", type=int, default=10)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Dense-index chunk size in tokens. Defaults to HybridRetriever's "
        "own default (256).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=None,
        help="Chunk overlap in tokens. Defaults to HybridRetriever's own default "
        "(25). Changing chunk settings requires deleting the persisted index.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Results are printed if omitted.",
    )
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    # Only explicit flags override the retriever defaults, so omitting both keeps
    # the process-wide tool identical to HybridRetriever().
    retriever_overrides = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    retriever_overrides = {
        key: value for key, value in retriever_overrides.items() if value is not None
    }
    if retriever_overrides:
        configure_default_retriever(**retriever_overrides)
        print(f"Retriever overrides: {retriever_overrides}")

    if args.query:
        result = hybrid_retrieve(
            query=args.query,
            retrieve_top_k=args.retrieve_top_k,
            rrf_top_k=args.rrf_top_k,
            rerank_top_n=args.rerank_top_n,
        )
    else:
        with open(args.queries, "r", encoding="utf-8") as file:
            query_data = json.load(file)

        end = args.start + args.limit if args.limit is not None else None
        selected_queries = query_data[args.start:end]
        if not selected_queries:
            parser.error("The selected --start/--limit range contains no query records")

        print(
            f"Running hybrid retrieval for {len(selected_queries)} original query records "
            f"(indices {args.start} to {args.start + len(selected_queries) - 1})."
        )
        document_lists = hybrid_retrieve_batch(
            [item["query"] for item in selected_queries],
            retrieve_top_k=args.retrieve_top_k,
            rrf_top_k=args.rrf_top_k,
            rerank_top_n=args.rerank_top_n,
        )

        # Keep the exact schema expected by retrieval_evaluate.py.
        result = []
        for item, documents in zip(selected_queries, document_lists):
            retrieval_list = []
            for document in documents:
                retrieval_item = dict(document)
                retrieval_item["score"] = document["rerank_score"]
                retrieval_list.append(retrieval_item)

            result.append(
                {
                    "query": item["query"],
                    "answer": item["answer"],
                    "question_type": item["question_type"],
                    "retrieval_list": retrieval_list,
                    "gold_list": item["evidence_list"],
                }
            )

    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")
        if args.queries:
            print(f"Saved {len(result)} query results to: {output_path}")
        else:
            print(f"Saved {len(result)} documents to: {output_path}")
    else:
        print(result_json)


if __name__ == "__main__":
    main()

"""
python run_hybrid_retrieval.py `
  --queries dataset/MultiHopRAG.json `
  --start 0 `
  --limit 10 `
  --retrieve_top_k 10 `
  --rrf_top_k 10 `
  --rerank_top_n 10 `
  --output output/hybrid_query_result.json

python retrieval_evaluate.py `
  --file output/hybrid_query_result.json `
  --limit 10
"""