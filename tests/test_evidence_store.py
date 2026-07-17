import unittest

from agentic_rag.evidence_store import evidence_to_context, merge_evidence


class EvidenceStoreTests(unittest.TestCase):
    def test_merge_deduplicates_and_tracks_hops(self):
        first_hop = [{"text": "A fact", "title": "T", "source": "S", "rerank_score": 0.2}]
        second_hop = [
            {"text": " A  fact ", "title": "T", "source": "S", "rerank_score": 0.8},
            {"text": "A new fact", "title": "T2", "source": "S2", "rerank_score": 0.3},
        ]

        evidence, first_added = merge_evidence([], first_hop, hop=1)
        evidence, second_added = merge_evidence(evidence, second_hop, hop=2)

        self.assertEqual(first_added, 1)
        self.assertEqual(second_added, 1)
        self.assertEqual(len(evidence), 2)
        duplicate = next(item for item in evidence if item["text"].strip() == "A  fact")
        self.assertEqual(duplicate["evidence_id"], "E1")
        self.assertEqual(duplicate["retrieved_hops"], [1, 2])
        self.assertIn("[E1]", evidence_to_context(evidence))


if __name__ == "__main__":
    unittest.main()