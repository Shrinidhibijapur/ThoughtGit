import unittest
from unittest.mock import patch
import os
import shutil
import gc
from datetime import datetime

# Setup directories for testing
test_db_dir = "./test_api_db_store"
test_cache_path = "./test_api_cache.db"

if os.path.exists(test_db_dir):
    shutil.rmtree(test_db_dir, ignore_errors=True)
if os.path.exists(test_cache_path):
    try:
        os.remove(test_cache_path)
    except OSError:
        pass

# Patch the configuration *before* importing the app to redirect database storage
with patch('core.config.DB_DIR', test_db_dir), \
     patch('core.config.CACHE_DB_PATH', test_cache_path):
    from api.main import app, engine, store
    from fastapi.testclient import TestClient

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Clear collections from database before running each test to prevent test pollution
        for coll_name in store.list_collections():
            try:
                store.client.delete_collection(coll_name)
            except Exception:
                pass
                
        # Mock the ollama.embed call inside engine to avoid calling external service
        self.embed_patcher = patch('core.embedder.ollama.embed')
        self.mock_embed = self.embed_patcher.start()
        
        # Standard mock 768-dimension vector
        self.mock_vector = [0.1] * 768
        self.mock_embed.return_value = {"embeddings": [self.mock_vector]}

    def tearDown(self):
        self.embed_patcher.stop()

    @classmethod
    def tearDownClass(cls):
        # Force garbage collection to release file handles before cleaning up directory
        gc.collect()
        try:
            if os.path.exists(test_db_dir):
                shutil.rmtree(test_db_dir)
            if os.path.exists(test_cache_path):
                os.remove(test_cache_path)
        except PermissionError:
            pass

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

    def test_ingest_thought(self):
        payload = {
            "content": "Self-attention calculations map queries, keys, and values.",
            "source": "vscode",
            "timestamp": datetime(2026, 7, 17).isoformat(),
            "metadata": {"branch": "main"}
        }
        
        response = self.client.post("/ingest", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(response.json()["chunks_count"], 1)

    def test_recall_thought(self):
        # First ingest a note
        payload = {
            "content": "Retrieval-augmented generation grounds LLM output.",
            "source": "obsidian",
            "timestamp": datetime(2026, 6, 10).isoformat(),
            "metadata": {"branch": "main"}
        }
        self.client.post("/ingest", json=payload)
        
        # Run recall query
        response = self.client.get("/recall", params={"query": "RAG search", "n_results": 1})
        self.assertEqual(response.status_code, 200)
        
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Retrieval-augmented generation grounds LLM output.")
        self.assertNotIn("embedding", results[0]) # Should be stripped for response size

    def test_diff_endpoint(self):
        # Ingest thoughts for two months to trigger snapshots
        self.client.post("/ingest", json={
            "content": "RAG basics mapping documents.",
            "source": "vscode",
            "timestamp": datetime(2026, 5, 10).isoformat()
        })
        self.client.post("/ingest", json={
            "content": "Advanced RAG processes hybrid vectors.",
            "source": "vscode",
            "timestamp": datetime(2026, 6, 10).isoformat()
        })
        
        response = self.client.get("/diff", params={"topic": "RAG", "min_cluster_size": 1})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("snapshots", data)
        self.assertIn("drift_events", data)
        # Should have snapshots for both May and June
        self.assertEqual(len(data["snapshots"]), 2)
