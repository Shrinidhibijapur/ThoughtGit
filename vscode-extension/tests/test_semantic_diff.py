import unittest
from unittest.mock import MagicMock
import numpy as np
from core.semantic_diff import SemanticDiffEngine

class TestSemanticDiffEngine(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock()
        self.engine = SemanticDiffEngine(self.mock_store)

    def test_cosine_distance(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.engine._cosine_distance(u, v), 0.0)

        v_orthogonal = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.engine._cosine_distance(u, v_orthogonal), 1.0)

    def test_analyze_drift_no_data(self):
        self.mock_store.get_all_chunks_for_diff.return_value = []
        result = self.engine.analyze_drift("RAG", [0.1]*768)
        self.assertEqual(result["snapshots"], [])
        self.assertEqual(result["drift_events"], [])

    def test_analyze_drift_evolution(self):
        # Create mock collections
        self.mock_store.list_collections.return_value = [
            "thoughts_main_2026_01",
            "thoughts_main_2026_02",
            "thoughts_main_2026_03"
        ]

        # Construct vector directions (length 768)
        vec_jan = [1.0] + [0.0]*767 
        
        # Make February very close to January (cosine similarity ~0.999, distance ~0.0)
        vec_feb = [0.999, 0.001] + [0.0]*766
        vec_feb = list(np.array(vec_feb) / np.linalg.norm(vec_feb))
        
        # Make March have a moderate direction shift (cosine distance ~0.293, between 0.25 and 0.40)
        vec_mar = [0.707, 0.707] + [0.0]*766
        vec_mar = list(np.array(vec_mar) / np.linalg.norm(vec_mar))

        # Mock items retrieved by diff query
        mock_chunks = [
            # January
            {
                "id": "chunk_jan_1",
                "text": "Basic RAG searches document embeddings and appends to context.",
                "embedding": vec_jan,
                "collection": "thoughts_main_2026_01"
            },
            {
                "id": "chunk_jan_2",
                "text": "RAG helps ground LLM answers to prevent hallucination.",
                "embedding": vec_jan,
                "collection": "thoughts_main_2026_01"
            },
            # February (Reinforced / slight refine)
            {
                "id": "chunk_feb_1",
                "text": "RAG systems use vector databases to store documents.",
                "embedding": vec_feb,
                "collection": "thoughts_main_2026_02"
            },
            {
                "id": "chunk_feb_2",
                "text": "We store chunked text vectors in a local ChromaDB index.",
                "embedding": vec_feb,
                "collection": "thoughts_main_2026_02"
            },
            # March (Deepened / moderate shift)
            {
                "id": "chunk_mar_1",
                "text": "To scale RAG we need hybrid keyword search plus semantic rerankers, query compression, and hierarchical retrieval summaries.",
                "embedding": vec_mar,
                "collection": "thoughts_main_2026_03"
            },
            {
                "id": "chunk_mar_2",
                "text": "Evaluating RAG requires monitoring faithfulness, answer relevance, and context precision using metrics like BERTScore.",
                "embedding": vec_mar,
                "collection": "thoughts_main_2026_03"
            }
        ]

        self.mock_store.get_all_chunks_for_diff.return_value = mock_chunks

        # Analyze drift
        result = self.engine.analyze_drift("RAG", vec_jan, min_cluster_size=2)

        # 3 months of snapshots
        self.assertEqual(len(result["snapshots"]), 3)
        self.assertEqual(result["snapshots"][0]["time_label"], "2026-01")
        self.assertEqual(result["snapshots"][1]["time_label"], "2026-02")
        self.assertEqual(result["snapshots"][2]["time_label"], "2026-03")

        # 2 transitions (Jan -> Feb, Feb -> Mar)
        self.assertEqual(len(result["drift_events"]), 2)

        # Jan -> Feb: Small distance, should be "reinforced" or "refined"
        event1 = result["drift_events"][0]
        self.assertEqual(event1["from_period"], "2026-01")
        self.assertEqual(event1["to_period"], "2026-02")
        self.assertLess(event1["distance"], 0.25)
        self.assertEqual(event1["drift_type"], "reinforced")

        # Feb -> Mar: Moderate distance, should be "deepened" because March texts are longer
        event2 = result["drift_events"][1]
        self.assertEqual(event2["from_period"], "2026-02")
        self.assertEqual(event2["to_period"], "2026-03")
        self.assertGreater(event2["distance"], 0.25)
        self.assertLess(event2["distance"], 0.40)
        self.assertEqual(event2["drift_type"], "deepened")
