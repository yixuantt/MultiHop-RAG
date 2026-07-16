import json
import argparse
import os
from tqdm import tqdm
from FlagEmbedding import FlagReranker


DEFAULT_INPUT_FILE = "output/all-MiniLM-L6-v2_retrieval_test.json"
DEFAULT_OUTPUT_FILE = "output/all-MiniLM-L6-v2_bge_rerank_retrieval_test.json"
DEFAULT_MODEL_PATH = "./models/bge-reranker-v2-m3"


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank retrieval results with BGE reranker.")

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Input retrieval-result JSON file."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output reranked JSON file."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local reranker model path."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first question to rerank."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to rerank. Omit to process all remaining questions."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of reranked documents to keep for each query."
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use fp16. Only enable this if your GPU supports it."
    )

    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start must be non-negative")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if args.top_n <= 0:
        parser.error("--top_n must be a positive integer")

    return args


def ensure_score_list(scores):
    """
    FlagReranker.compute_score returns a float when there is only one pair,
    and a list when there are multiple pairs. This helper normalizes it to list.
    """
    if isinstance(scores, (int, float)):
        return [float(scores)]

    return [float(score) for score in scores]


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Reranker model path not found: {args.model}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Reranker model: {args.model}")
    print(f"Top N: {args.top_n}")

    reranker = FlagReranker(
        args.model,
        use_fp16=args.use_fp16
    )

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.start >= len(data):
        raise ValueError(
            f"--start must be smaller than the number of questions ({len(data)})"
        )

    end = args.start + args.limit if args.limit is not None else len(data)
    data_to_rerank = data[args.start:end]

    print(
        f"Reranking {len(data_to_rerank)} questions "
        f"(indices {args.start} to {args.start + len(data_to_rerank) - 1})"
    )

    rerank_results = []

    for item in tqdm(data_to_rerank):
        query = item["query"]
        retrieval_list = item["retrieval_list"]

        documents = [
            doc["text"]
            for doc in retrieval_list
        ]

        if len(documents) == 0:
            new_retrieval_list = []
        else:
            pairs = [
                [query, doc]
                for doc in documents
            ]

            scores = reranker.compute_score(
                pairs,
                normalize=True
            )
            scores = ensure_score_list(scores)

            ranked = sorted(
                zip(retrieval_list, scores),
                key=lambda x: x[1],
                reverse=True
            )

            ranked = ranked[:args.top_n]

            new_retrieval_list = []

            for doc, score in ranked:
                new_retrieval_list.append(
                    {
                        "text": doc["text"],
                        "score": score,
                        "original_score": doc.get("score", None)
                    }
                )

        new_item = {
            "query": item["query"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "retrieval_list": new_retrieval_list,
            "gold_list": item["gold_list"]
        }

        rerank_results.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            rerank_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Saved:", args.output)


if __name__ == "__main__":
    main()