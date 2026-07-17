import unittest

from agentic_rag.graph import build_v4_retrieval_graph
from agentic_rag.schemas import JudgeDecision
from agentic_rag.state import make_initial_state


class GraphTests(unittest.TestCase):
    def test_graph_runs_two_hops_and_merges_evidence(self):
        calls = []

        def retriever(query, retrieve_top_k, rrf_top_k, rerank_top_n):
            calls.append(query)
            if query == "original question":
                return [{"text": "Company A is mentioned.", "title": "First", "source": "news"}]
            return [{"text": "Company B acquired Company A.", "title": "Second", "source": "news"}]

        def judge(question, evidence, history):
            if len(history) == 1:
                return JudgeDecision(
                    enough=False,
                    missing_information="Need the acquirer.",
                    next_query="Who acquired Company A?",
                )
            return JudgeDecision(enough=True, evidence_ids_used=["E1", "E2"])

        graph = build_v4_retrieval_graph(judge=judge, retriever=retriever)
        result = graph.invoke(make_initial_state("original question", max_hops=2))

        self.assertEqual(calls, ["original question", "Who acquired Company A?"])
        self.assertEqual(result["hop"], 2)
        self.assertEqual(result["status"], "ready_to_answer")
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(len(result["hop_traces"]), 2)

    def test_graph_stops_when_next_query_repeats(self):
        def retriever(*_):
            return [{"text": "Only one fact", "title": "T", "source": "S"}]

        def judge(question, evidence, history):
            return JudgeDecision(enough=False, next_query="original question")

        graph = build_v4_retrieval_graph(judge=judge, retriever=retriever)
        result = graph.invoke(make_initial_state("original question", max_hops=2))

        self.assertEqual(result["hop"], 1)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()