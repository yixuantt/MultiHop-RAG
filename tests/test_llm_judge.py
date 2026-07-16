import unittest

from agentic_rag.graph import build_v4_retrieval_graph
from agentic_rag.judge import LLMEvidenceJudge
from agentic_rag.llm_client import JsonLLMClient, TextGenerationClient
from agentic_rag.state import make_initial_state


class FakeTextClient(TextGenerationClient):
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompt, *, max_new_tokens=512, temperature=0.0):
        return next(self.responses)


class LLMJudgeTests(unittest.TestCase):
    def test_json_client_extracts_fenced_json(self):
        client = JsonLLMClient(FakeTextClient([
            'Decision:\n```json\n{"enough": true, "next_query": ""}\n```'
        ]))
        judge = LLMEvidenceJudge(client)
        decision = judge("question", [{"evidence_id": "E1", "text": "fact"}], ["question"])
        self.assertTrue(decision.enough)
        self.assertEqual(decision.next_query, "")

    def test_graph_uses_llm_judge(self):
        client = JsonLLMClient(FakeTextClient([
            '{"enough": false, "next_query": "search Company A"}',
            '{"enough": true, "next_query": ""}',
        ]))
        judge = LLMEvidenceJudge(client)

        def retriever(query, *_):
            return [{"text": query, "title": query, "source": "test"}]

        graph = build_v4_retrieval_graph(judge=judge, retriever=retriever)
        result = graph.invoke(make_initial_state("original", max_hops=2))
        self.assertEqual(result["status"], "ready_to_answer")
        self.assertEqual(result["hop"], 2)
        self.assertEqual(result["search_history"], ["original", "search Company A"])


if __name__ == "__main__":
    unittest.main()