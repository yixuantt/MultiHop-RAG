"""Run the V4.0 LangGraph retrieval workflow over a MultiHopRAG JSON range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from agentic_rag.graph import build_dashscope_v4_retrieval_graph
from agentic_rag.state import make_initial_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V4.0 Agentic RAG and save full per-question traces."
    )
    parser.add_argument("--queries", default="dataset/MultiHopRAG.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to output/agentic_start<start>_limit<count>.json.",
    )
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--rrf_top_k", type=int, default=10)
    parser.add_argument("--rerank_top_n", type=int, default=10)
    parser.add_argument("--max_evidence_documents", type=int, default=10)
    #Evidence Store 最多保留 10 条跨跳证据，供 Judge 和后续第二跳使用。
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dataset indices already present in --output.",
    )
    return parser.parse_args()


def write_results(path: Path, results: list[dict]) -> None:
    """Atomically update the trace file after each completed question."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    if min(
        args.max_hops,
        args.retrieve_top_k,
        args.rrf_top_k,
        args.rerank_top_n,
        args.max_evidence_documents,
    ) <= 0:
        raise ValueError("all retrieval and hop limits must be positive")

    query_path = Path(args.queries)
    with query_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    selected = list(enumerate(dataset[args.start:args.start + args.limit], start=args.start))
    if not selected:
        raise ValueError("The selected --start/--limit range contains no questions")

    output_path = (
        Path(args.output)
        if args.output
        else Path("output") / f"agentic_start{args.start}_limit{len(selected)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    completed_indices: set[int] = set()
    if args.resume and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        completed_indices = {
            result["dataset_index"]
            for result in results
            if isinstance(result, dict) and "dataset_index" in result
        }
        print(f"Resume mode: found {len(completed_indices)} completed questions.")

    # The graph (and its DashScope Judge configuration) is created once. The
    # underlying HybridRetriever is also initialized only on its first use.
    graph = build_dashscope_v4_retrieval_graph()
    print(
        f"Running V4.0 Agentic RAG for {len(selected)} original records "
        f"(indices {selected[0][0]} to {selected[-1][0]})."
    )

    for position, (dataset_index, item) in enumerate(selected, start=1):
        if dataset_index in completed_indices:
            print(f"[{position}/{len(selected)}] Skipping completed index {dataset_index}.")
            continue

        print(f"\n===== Dataset question {dataset_index} ({position}/{len(selected)}) =====")
        started = perf_counter()
        # Only the query enters AgentState. Gold answer/evidence/type stay
        # outside the graph until execution has finished.
        state = make_initial_state(
            question=item["query"],
            max_hops=args.max_hops,
            retrieve_top_k=args.retrieve_top_k,
            rrf_top_k=args.rrf_top_k,
            rerank_top_n=args.rerank_top_n,
            max_evidence_documents=args.max_evidence_documents,
        )
        result = dict(graph.invoke(state))
        total_seconds = perf_counter() - started

        latency = dict(result.get("latency", {}))
        latency["total_seconds"] = total_seconds
        result["latency"] = latency

        # Attach evaluation-only fields after the Agent has completed.
        result["dataset_index"] = dataset_index
        result["gold_answer"] = item["answer"]
        result["question_type"] = item["question_type"]
        result["gold_list"] = item["evidence_list"]
        results.append(result)
        write_results(output_path, results)

        print(
            f"Saved trace. status={result['status']}, hops={result['hop']}, "
            f"evidence={len(result['evidence'])}, total={total_seconds:.2f}s"
        )

    print(f"Completed. Trace output: {output_path}")


if __name__ == "__main__":
    main()
"""
python run_agentic_rag.py --start 0 --limit 1
会自动输出到：
output/agentic_start0_limit1.json
"""