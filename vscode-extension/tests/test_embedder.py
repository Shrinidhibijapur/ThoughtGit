import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import os
import shutil
from core.models import RawThought
from core.embedder import EmbeddingEngine

class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        # Use a temporary database path for testing cache
        self.test_dir = "./test_thoughtgit_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.patcher = patch('core.embedder.CACHE_DB_PATH', os.path.join(self.test_dir, "test_cache.db"))
        self.patcher.start()
        
        self.engine = EmbeddingEngine()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_chunk_text_short(self):
        text = "This is a short note."
        chunks = self.engine.chunk_text(text, "test")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_long(self):
        # Create a text with 500 words
        words = ["word"] * 500
        text = " ".join(words)
        chunks = self.engine.chunk_text(text, "test")
        
        # Word limit is 400 with 50 overlap.
        # Chunk 1: index 0 to 400
        # Chunk 2: index 350 to 500
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].split()), 400)
        self.assertEqual(len(chunks[1].split()), 150)

    @patch('core.embedder.ollama.embed')
    def test_embedding_caching(self, mock_embed):
        # Set up mock response
        mock_vector = [0.1] * 768
        mock_embed.return_value = {"embeddings": [mock_vector]}
        
        text = "This is some test content to embed."
        
        # First call: should call ollama
        vector1 = self.engine.embed(text)
        self.assertEqual(vector1, mock_vector)
        self.assertEqual(mock_embed.call_count, 1)
        
        # Second call: should read from cache and NOT call ollama
        vector2 = self.engine.embed(text)
        self.assertEqual(vector2, mock_vector)
        self.assertEqual(mock_embed.call_count, 1) # Still 1

    @patch('core.embedder.ollama.embed')
    def test_embed_thought(self, mock_embed):
        mock_vector = [0.2] * 768
        mock_embed.return_value = {"embeddings": [mock_vector]}
        
        thought = RawThought(
            content="My RAG test thoughts",
            source="vscode",
            timestamp=datetime(2026, 7, 17),
            metadata={"branch": "main"}
        )
        
        chunks = self.engine.embed_thought(thought)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "My RAG test thoughts")
        self.assertEqual(chunks[0].collection_name, "thoughts_main_2026_07")
        self.assertEqual(chunks[0].metadata["source"], "vscode")
