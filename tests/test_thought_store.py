import unittest
from datetime import datetime
import os
import shutil
import gc
from unittest.mock import patch
from core.models import EmbeddedChunk
from core.thought_store import ThoughtStore

class TestThoughtStore(unittest.TestCase):
    def setUp(self):
        # Use a unique directory for each test case to prevent cross-test pollution on Windows file locking
        self.test_dir = f"./test_thoughtgit_data_store_{self._testMethodName}"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Patch the ChromaDB directory
        self.patcher = patch('core.thought_store.DB_DIR', self.test_dir)
        self.patcher.start()
        
        self.store = ThoughtStore()

    def tearDown(self):
        self.patcher.stop()
        self.store = None
        gc.collect() # Force garbage collection to release file handles in ChromaDB
        
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except PermissionError:
            pass # Ignore lock errors on Windows, isolation is guaranteed by unique test_dir

    def test_store_and_query(self):
        mock_embedding = [0.1] * 768
        
        chunk = EmbeddedChunk(
            chunk_id="chunk_test_1",
            text="Vector databases are useful for storing embeddings.",
            embedding=mock_embedding,
            source="vscode",
            timestamp=datetime(2026, 6, 15),
            collection_name="thoughts_main_2026_06",
            metadata={"source": "vscode", "timestamp": datetime(2026, 6, 15).isoformat()}
        )
        
        # Store chunk
        self.store.store_chunk(chunk)
        
        # Verify collection exists in list
        colls = self.store.list_collections(branch="main")
        self.assertIn("thoughts_main_2026_06", colls)
        
        # Query across time
        results = self.store.query_across_time(
            query_embedding=mock_embedding,
            n_results=1,
            branch="main"
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Vector databases are useful for storing embeddings.")
        self.assertEqual(results[0]["collection"], "thoughts_main_2026_06")

    def test_store_batch_and_diff_retrieval(self):
        mock_embedding = [0.2] * 768
        
        chunks = [
            EmbeddedChunk(
                chunk_id="chunk_a",
                text="RAG is Retrieval-Augmented Generation.",
                embedding=mock_embedding,
                source="obsidian",
                timestamp=datetime(2026, 5, 10),
                collection_name="thoughts_main_2026_05",
                metadata={"source": "obsidian"}
            ),
            EmbeddedChunk(
                chunk_id="chunk_b",
                text="Advanced RAG concepts include query rewriting.",
                embedding=mock_embedding,
                source="obsidian",
                timestamp=datetime(2026, 6, 10),
                collection_name="thoughts_main_2026_06",
                metadata={"source": "obsidian"}
            )
        ]
        
        self.store.store_chunks_batch(chunks)
        
        # Verify both collections created
        colls = self.store.list_collections(branch="main")
        self.assertEqual(len(colls), 2)
        self.assertEqual(colls[0], "thoughts_main_2026_05")
        self.assertEqual(colls[1], "thoughts_main_2026_06")
        
        # Get chunks for diff with high similarity (similarity > 0.7)
        diff_chunks = self.store.get_all_chunks_for_diff(
            query_embedding=mock_embedding,
            threshold=0.7,
            branch="main"
        )
        
        # We expect both to match because both use the query embedding exactly
        self.assertEqual(len(diff_chunks), 2)
