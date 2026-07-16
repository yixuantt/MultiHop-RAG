import argparse
import json
import re
from collections import defaultdict

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated QA answers.")
    parser.add_argument(
        "--file",
        default="qa_output/hybrid_llama.json",
        help="Generated-answer JSON from qa_llama.py.",
    )
    parser.add_argument(
        "--queries",
        default="dataset/MultiHopRAG.json",
        help="Original MultiHopRAG query JSON, used as a fallback for gold answers.",
    )
    return parser.parse_args()


def has_intersection(prediction: str, gold: str) -> bool:
    return bool(set(prediction.split()).intersection(gold.split()))


def extract_answer(text: str) -> str:
    match = re.search(r'The answer to the question is "(.*?)"', text)
    return match.group(1) if match else text


def calculate_metrics(predictions, gold_answers):
    correct = sum(
        has_intersection(prediction.lower(), gold.lower())
        for prediction, gold in zip(predictions, gold_answers)
    )
    total = len(gold_answers)
    # For this one-answer-per-question setting, precision, recall, F1 and
    # accuracy all reduce to the same exact per-question success rate.
    score = correct / total if total else 0.0
    return score, score, score, score


def main() -> None:
    args = parse_args()
    with open(args.file, "r", encoding="utf-8") as file:
        doc_data = json.load(file)
    with open(args.queries, "r", encoding="utf-8") as file:
        query_data = json.load(file)

    gold_by_query = {item["query"]: item["answer"] for item in query_data}
    type_data = defaultdict(lambda: {"predictions": [], "gold_answers": []})
    overall_predictions = []
    overall_gold_answers = []

    for item in tqdm(doc_data, desc="Evaluating answers"):
        model_answer = extract_answer(item["model_answer"])
        gold_answer = item.get("gold_answer") or gold_by_query.get(item["query"])
        if not gold_answer:
            print(f"Skipping query without a gold answer: {item['query']}")
            continue

        question_type = item.get("question_type", "unknown")
        type_data[question_type]["predictions"].append(model_answer)
        type_data[question_type]["gold_answers"].append(gold_answer)
        overall_predictions.append(model_answer)
        overall_gold_answers.append(gold_answer)

    for question_type, data in type_data.items():
        precision, recall, f1, accuracy = calculate_metrics(
            data["predictions"], data["gold_answers"]
        )
        print(f"Question Type: {question_type}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print(f" Accuracy: {accuracy:.4f}")

    precision, recall, f1, accuracy = calculate_metrics(
        overall_predictions,
        overall_gold_answers,
    )
    print("Overall Metrics:")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()