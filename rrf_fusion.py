import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    """
    消除换行符和多余空格造成的文本 key 差异。
    不转小写，避免对原始输出造成不必要修改。
    """
    return " ".join(text.split())


def add_ranked_list(
    fused: Dict[str, Dict[str, Any]],
    retrieval_list: List[Dict[str, Any]],
    source_name: str,
    weight: float,
    rrf_k: float,
) -> None:
    for rank, item in enumerate(retrieval_list, start=1):
        text = item["text"]
        key = normalize_text(text)

        if key not in fused:
            fused[key] = {
                "text": text,
                "rrf_score": 0.0,
            }

        record = fused[key]

        record["rrf_score"] += weight / (rrf_k + rank)
        record[f"{source_name}_rank"] = rank
        record[f"{source_name}_score"] = item.get("score")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dense", required=True)
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--rrf_k", type=float, default=60.0)

    parser.add_argument(
        "--dense_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bm25_weight",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    with open(args.dense, "r", encoding="utf-8") as file:
        dense_data = json.load(file)

    with open(args.bm25, "r", encoding="utf-8") as file:
        bm25_data = json.load(file)

    if len(dense_data) != len(bm25_data):
        raise ValueError(
            "Dense 和 BM25 结果数量不同："
            f"{len(dense_data)} != {len(bm25_data)}"
        )

    output_data = []

    for index, (dense_item, bm25_item) in enumerate(
        zip(dense_data, bm25_data)
    ):
        if dense_item["query"] != bm25_item["query"]:
            raise ValueError(
                f"第 {index} 条 query 不一致：\n"
                f"Dense: {dense_item['query']}\n"
                f"BM25: {bm25_item['query']}"
            )

        fused: Dict[str, Dict[str, Any]] = {}

        add_ranked_list(
            fused=fused,
            retrieval_list=dense_item["retrieval_list"],
            source_name="dense",
            weight=args.dense_weight,
            rrf_k=args.rrf_k,
        )

        add_ranked_list(
            fused=fused,
            retrieval_list=bm25_item["retrieval_list"],
            source_name="bm25",
            weight=args.bm25_weight,
            rrf_k=args.rrf_k,
        )

        ranked_records = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )[: args.topk]

        retrieval_list = []

        for record in ranked_records:
            retrieval_list.append(
                {
                    "text": record["text"],

                    # 保留 score 字段，兼容现有 JSON 结构。
                    "score": record["rrf_score"],
                    "rrf_score": record["rrf_score"],

                    # 以下字段方便后续观察融合过程。
                    "dense_rank": record.get("dense_rank"),
                    "dense_score": record.get("dense_score"),
                    "bm25_rank": record.get("bm25_rank"),
                    "bm25_score": record.get("bm25_score"),
                }
            )

        result_item = dict(dense_item)
        result_item["retrieval_list"] = retrieval_list

        output_data.append(result_item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved RRF results to: {output_path}")


if __name__ == "__main__":
    main()