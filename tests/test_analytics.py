import unittest
import os
import shutil
import gc
import math
import warnings
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Silence deprecation warnings from standard libraries (like utcnow())
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine
from core.models import EmbeddedChunk

# Dynamic paths for test isolation
test_dir = "./test_analytics_data"
test_db_dir = os.path.join(test_dir, "chroma")
test_metadata_path = os.path.join(test_dir, "metadata.db")
test_cache_path = os.path.join(test_dir, "cache.db")

with patch('core.config.BASE_DIR', test_dir), \
     patch('core.config.DB_DIR', test_db_dir), \
     patch('core.config.METADATA_DB_PATH', test_metadata_path), \
     patch('core.config.CACHE_DB_PATH', test_cache_path):
    from core.learning_velocity import LearningVelocityEngine
    from core.forgetting_curve import ForgettingCurveTracker
    from core.ai_mentor import AIMentor
    from core.semantic_commits import SemanticCommitLogger
    from core.memory_health import MemoryHealthEngine

class TestAnalyticsEngine(unittest.TestCase):
    def setUp(self):
        # Unique directory per test case to avoid database locks on Windows
        self.test_dir = f"./test_analytics_data_{self._testMethodName}"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_db_dir = os.path.join(self.test_dir, "chroma")
        self.test_metadata_path = os.path.join(self.test_dir, "metadata.db")
        self.test_cache_path = os.path.join(self.test_dir, "cache.db")
        
        os.makedirs(self.test_db_dir, exist_ok=True)

        # Dynamic patching of global module configurations to redirect storage paths
        self.patchers = [
            patch('core.config.BASE_DIR', self.test_dir),
            patch('core.config.DB_DIR', self.test_db_dir),
            patch('core.config.METADATA_DB_PATH', self.test_metadata_path),
            patch('core.config.CACHE_DB_PATH', self.test_cache_path),
            
            patch('core.thought_store.DB_DIR', self.test_db_dir),
            
            patch('core.forgetting_curve.METADATA_DB_PATH', self.test_metadata_path),
            patch('core.semantic_commits.METADATA_DB_PATH', self.test_metadata_path),
            patch('core.memory_health.METADATA_DB_PATH', self.test_metadata_path)
        ]
        for p in self.patchers:
            p.start()

        # Instantiate components
        self.store = ThoughtStore()
        self.diff_engine = SemanticDiffEngine(self.store)
        self.velocity_engine = LearningVelocityEngine(self.store, self.diff_engine)
        self.forgetting_tracker = ForgettingCurveTracker()
        self.ai_mentor = AIMentor(self.store)
        self.commit_logger = SemanticCommitLogger()
        self.health_engine = MemoryHealthEngine(self.store, self.forgetting_tracker)

    def tearDown(self):
        for p in self.patchers:
            p.stop()
            
        self.store = None
        self.diff_engine = None
        self.velocity_engine = None
        self.forgetting_tracker = None
        self.ai_mentor = None
        self.commit_logger = None
        self.health_engine = None
        gc.collect()
        
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except PermissionError:
            pass

    def test_learning_velocity_computations(self):
        mock_query_emb = [0.1] * 768
        
        # Test output when no data is present
        empty_res = self.velocity_engine.calculate_velocity("nonexistent", mock_query_emb)
        self.assertEqual(empty_res["volume_velocity"], 0.0)
        self.assertEqual(empty_res["conceptual_velocity"], 0.0)
        
        # Ingest synthetic chunks
        chunks = [
            EmbeddedChunk(
                chunk_id="chunk1",
                text="June note on vector stores.",
                embedding=mock_query_emb,
                source="vscode",
                timestamp=datetime(2026, 6, 10),
                collection_name="thoughts_main_2026_06",
                metadata={"timestamp": datetime(2026, 6, 10).isoformat(), "timestamp_epoch": datetime(2026, 6, 10).timestamp()}
            ),
            EmbeddedChunk(
                chunk_id="chunk2",
                text="June note 2 on embeddings.",
                embedding=mock_query_emb,
                source="vscode",
                timestamp=datetime(2026, 6, 20),
                collection_name="thoughts_main_2026_06",
                metadata={"timestamp": datetime(2026, 6, 20).isoformat(), "timestamp_epoch": datetime(2026, 6, 20).timestamp()}
            ),
            EmbeddedChunk(
                chunk_id="chunk3",
                text="July note on vector search.",
                embedding=mock_query_emb,
                source="vscode",
                timestamp=datetime(2026, 7, 5),
                collection_name="thoughts_main_2026_07",
                metadata={"timestamp": datetime(2026, 7, 5).isoformat(), "timestamp_epoch": datetime(2026, 7, 5).timestamp()}
            )
        ]
        self.store.store_chunks_batch(chunks)
        
        res = self.velocity_engine.calculate_velocity("vector", mock_query_emb)
        self.assertEqual(res["total_chunks"], 3)
        self.assertEqual(res["periods_tracked"], 2) # June and July
        self.assertEqual(res["volume_velocity"], 1.5) # 3 chunks / 2 periods
        self.assertEqual(res["volume_trend"], "decreasing") # June (2 chunks) -> July (1 chunk)

    def test_forgetting_curve_decay_and_scheduling(self):
        topic = "hdbscan clustering"
        
        # Record initial access
        self.forgetting_tracker.record_access(topic)
        
        # Immediate strength should be near 1.0 (since delta time t = 0)
        strength_now = self.forgetting_tracker.get_memory_strength(topic)
        self.assertAlmostEqual(strength_now, 1.0, places=2)
        
        # Simulate time delta: mock 2 days elapsed
        two_days_ago = datetime.utcnow() - timedelta(days=2)
        import sqlite3
        conn = sqlite3.connect(self.test_metadata_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE spaced_repetition SET last_accessed = ? WHERE topic = ?",
            (two_days_ago.isoformat(), topic)
        )
        conn.commit()
        conn.close()
        
        # S = 1.0, t = 2.0 -> R = e^(-2/1) = e^(-2) = 0.135
        strength_decayed = self.forgetting_tracker.get_memory_strength(topic)
        self.assertAlmostEqual(strength_decayed, math.exp(-2.0), places=3)
        
        # Get review schedule: decayed strength 0.135 < 0.40, topic should be listed
        schedule = self.forgetting_tracker.get_review_schedule()
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["topic"], topic)
        
        # Record review access -> stability should reset to 1.0 because retention decayed below 0.40
        self.forgetting_tracker.record_access(topic)
        
        # Check stability in database
        conn = sqlite3.connect(self.test_metadata_path)
        cursor = conn.cursor()
        cursor.execute("SELECT stability FROM spaced_repetition WHERE topic = ?", (topic,))
        stability = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(stability, 1.0)

    def test_ai_mentor_and_semantic_commits_fallbacks(self):
        mock_query_emb = [0.1] * 768
        
        # Test AI Mentor fallback suggestions
        chunk = EmbeddedChunk(
            chunk_id="ch",
            text="Mock context about reinforcement learning.",
            embedding=mock_query_emb,
            source="vscode",
            timestamp=datetime(2026, 7, 1),
            collection_name="thoughts_main_2026_07",
            metadata={"timestamp": datetime(2026, 7, 1).isoformat(), "timestamp_epoch": datetime(2026, 7, 1).timestamp()}
        )
        self.store.store_chunk(chunk)
        
        # Get suggestion (triggers offline mock fallback)
        advice = self.ai_mentor.get_mentor_suggestion("Reinforcement learning setup", mock_query_emb)
        self.assertIn("insight", advice)
        self.assertIn("action", advice)
        self.assertTrue(advice["insight"].startswith("Identified similarity to records"))
        
        # Test Semantic Commit fallback
        drift_event = {
            "from_period": "2026-06",
            "to_period": "2026-07",
            "distance": 0.28,
            "drift_type": "deepened",
            "summary": "Added deep policy gradients description."
        }
        
        commit_msg = self.commit_logger.create_commit("rl_policy", drift_event)
        self.assertIn("Shift in rl_policy: DEEPENED", commit_msg)
        
        # Retrieve commit history
        history = self.commit_logger.get_commit_history("rl_policy")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["message"], commit_msg)

    def test_memory_health_scores(self):
        # Initial health score (no activities, defaults to 30.0 spacing, 0 activity, 0 diversity)
        report = self.health_engine.calculate_health_report()
        self.assertEqual(report["health_score"], 30.0)
        self.assertEqual(report["metrics"]["spacing"]["score"], 30.0)
        
        # Ingest a chunk in the last 30 days
        mock_query_emb = [0.1] * 768
        now = datetime.utcnow()
        chunk = EmbeddedChunk(
            chunk_id="recent_ch",
            text="Today note about spacing models.",
            embedding=mock_query_emb,
            source="obsidian",
            timestamp=now,
            collection_name=f"thoughts_main_{now.year}_{now.month:02d}",
            metadata={"timestamp": now.isoformat(), "timestamp_epoch": now.timestamp()}
        )
        self.store.store_chunk(chunk)
        
        # Record topic accesses in the forgetting curve database
        self.forgetting_tracker.record_access("spacing models")
        self.forgetting_tracker.record_access("topic 2")
        
        # Recalculate health
        report_after = self.health_engine.calculate_health_report()
        self.assertTrue(report_after["health_score"] > 30.0)
        self.assertEqual(report_after["metrics"]["activity"]["recent_chunks_count"], 1)
        self.assertEqual(report_after["metrics"]["diversity"]["unique_topics_count"], 2)
        # Diversity score: 2 unique topics * 7.0 = 14.0
        self.assertEqual(report_after["metrics"]["diversity"]["score"], 14.0)
        # Activity score: 1 chunk * 3.5 = 3.5
        self.assertEqual(report_after["metrics"]["activity"]["score"], 3.5)
        # Spacing score: both topics have strength 1.0 (recent review) -> ratio 1.0 -> 30.0 pts
        self.assertEqual(report_after["metrics"]["spacing"]["score"], 30.0)
        # Total: 3.5 + 14.0 + 30.0 = 47.5
        self.assertEqual(report_after["health_score"], 47.5)
