import unittest
from datetime import datetime
import os
import shutil
import gc
import warnings
from unittest.mock import patch, MagicMock

# Silence deprecation warnings from standard libraries (like utcnow())
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from core.branch import BranchManager
from core.time_machine import TimeMachine
from core.duplicate_detector import DuplicateDetector
from core.decision_log import DecisionLogger
from core.dead_ideas import DeadIdeasTracker
from core.thought_store import ThoughtStore
from core.models import EmbeddedChunk

class TestExtendedFeatures(unittest.TestCase):
    def setUp(self):
        # Unique directory per test case to avoid database locks and test pollution on Windows
        self.test_dir = f"./test_extended_features_data_{self._testMethodName}"
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
            
            patch('core.branch.BASE_DIR', self.test_dir),
            patch('core.branch.DB_DIR', self.test_db_dir),
            patch('core.branch.METADATA_DB_PATH', self.test_metadata_path),
            
            patch('core.thought_store.DB_DIR', self.test_db_dir),
            
            patch('core.decision_log.DB_DIR', self.test_db_dir),
            patch('core.decision_log.METADATA_DB_PATH', self.test_metadata_path),
            
            patch('core.dead_ideas.DB_DIR', self.test_db_dir),
            patch('core.dead_ideas.METADATA_DB_PATH', self.test_metadata_path)
        ]
        for p in self.patchers:
            p.start()

        # Mock embedding engine to output deterministic test vectors
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed.side_effect = lambda text: [0.1] * 768 if "basic" in text.lower() else [0.9] * 768
        
        # Instantiate stores and modules using patched configurations
        self.store = ThoughtStore()
        self.branch_manager = BranchManager()
        self.time_machine = TimeMachine(self.store)
        self.duplicate_detector = DuplicateDetector(self.store)
        self.decision_logger = DecisionLogger(self.mock_embedder)
        self.dead_ideas_tracker = DeadIdeasTracker(self.mock_embedder)

    def tearDown(self):
        # Stop all patchers
        for p in self.patchers:
            p.stop()
            
        # Clean references and force garbage collection to release database file locks on Windows
        self.store = None
        self.branch_manager = None
        self.time_machine = None
        self.duplicate_detector = None
        self.decision_logger = None
        self.dead_ideas_tracker = None
        gc.collect()
        
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except PermissionError:
            pass # Windows file locking delays cleanup, isolation is guaranteed by unique folder name

    def test_branch_lifecycle_and_merging(self):
        # Initial branch should be main
        self.assertEqual(self.branch_manager.get_active_branch(), "main")
        
        # Create new branch
        self.branch_manager.create_branch("dev")
        self.assertTrue(self.branch_manager.branch_exists("dev"))
        self.assertIn("dev", self.branch_manager.list_branches())
        
        # Switch branch
        self.branch_manager.switch_branch("dev")
        self.assertEqual(self.branch_manager.get_active_branch(), "dev")
        
        # Ingest a chunk in branch 'dev'
        mock_emb = [0.1] * 768
        chunk = EmbeddedChunk(
            chunk_id="dev_chunk_1",
            text="Branch dev code note.",
            embedding=mock_emb,
            source="vscode",
            timestamp=datetime(2026, 7, 1),
            collection_name="thoughts_dev_2026_07",
            metadata={
                "source": "vscode",
                "timestamp": datetime(2026, 7, 1).isoformat(),
                "timestamp_epoch": datetime(2026, 7, 1).timestamp()
            }
        )
        self.store.store_chunk(chunk)
        
        # Check source collection exists
        colls_dev = self.store.list_collections(branch="dev")
        self.assertIn("thoughts_dev_2026_07", colls_dev)
        
        # Merge branch dev into main
        self.branch_manager.merge_branch("dev", "main")
        
        # Chunks should now exist in thoughts_main_2026_07
        colls_main = self.store.list_collections(branch="main")
        self.assertIn("thoughts_main_2026_07", colls_main)
        
        # Query main branch and verify chunk is retrieved
        results = self.store.query_across_time(mock_emb, n_results=1, branch="main")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Branch dev code note.")

    def test_time_machine_filters_and_comparison(self):
        mock_emb = [0.1] * 768
        
        # Ingest chunks with June and July timestamps
        chunks = [
            EmbeddedChunk(
                chunk_id="chunk_june",
                text="June note about vectors.",
                embedding=mock_emb,
                source="vscode",
                timestamp=datetime(2026, 6, 15),
                collection_name="thoughts_main_2026_06",
                metadata={
                    "timestamp": datetime(2026, 6, 15).isoformat(),
                    "timestamp_epoch": datetime(2026, 6, 15).timestamp()
                }
            ),
            EmbeddedChunk(
                chunk_id="chunk_july",
                text="July note about agents.",
                embedding=mock_emb,
                source="vscode",
                timestamp=datetime(2026, 7, 10),
                collection_name="thoughts_main_2026_07",
                metadata={
                    "timestamp": datetime(2026, 7, 10).isoformat(),
                    "timestamp_epoch": datetime(2026, 7, 10).timestamp()
                }
            )
        ]
        self.store.store_chunks_batch(chunks)
        
        # Query time machine as of June 30
        results_june = self.time_machine.recall_as_of(
            topic="notes",
            query_embedding=mock_emb,
            as_of_date=datetime(2026, 6, 30),
            n_results=5
        )
        self.assertEqual(len(results_june), 1)
        self.assertEqual(results_june[0]["text"], "June note about vectors.")
        
        # Query time machine as of July 15
        results_july = self.time_machine.recall_as_of(
            topic="notes",
            query_embedding=mock_emb,
            as_of_date=datetime(2026, 7, 15),
            n_results=5
        )
        self.assertEqual(len(results_july), 2)
        
        # Compare June and July understanding
        comparison = self.time_machine.compare_understanding(
            topic="notes",
            query_embedding=mock_emb,
            date_a=datetime(2026, 6, 30),
            date_b=datetime(2026, 7, 15)
        )
        self.assertEqual(comparison["snapshots_count_a"], 1)
        self.assertEqual(comparison["snapshots_count_b"], 2)
        # New learnings should include July note
        self.assertEqual(len(comparison["new_learnings_since_a"]), 1)
        self.assertEqual(comparison["new_learnings_since_a"][0]["text"], "July note about agents.")

    def test_duplicate_detection(self):
        mock_emb = [0.1] * 768
        chunk = EmbeddedChunk(
            chunk_id="chunk_orig",
            text="How to design transformers.",
            embedding=mock_emb,
            source="obsidian",
            timestamp=datetime(2026, 7, 1),
            collection_name="thoughts_main_2026_07",
            metadata={
                "timestamp": datetime(2026, 7, 1).isoformat(),
                "timestamp_epoch": datetime(2026, 7, 1).timestamp()
            }
        )
        self.store.store_chunk(chunk)
        
        # Run duplicate check with exact vector (distance = 0.0)
        res_dup = self.duplicate_detector.check_duplicate("Transformer design query", mock_emb)
        self.assertTrue(res_dup["is_duplicate"])
        self.assertEqual(res_dup["matched_chunk"]["text"], "How to design transformers.")

    def test_decision_logger(self):
        # Log a decision
        decision_id = self.decision_logger.log_decision(
            title="Chose SQLite over Redis",
            chosen="SQLite",
            alternatives=["Redis", "Postgres"],
            reasoning="SQLite runs locally with zero installation dependencies, making deployment simpler.",
            assumptions="Local memory is sufficient",
            tags=["database", "storage"]
        )
        
        # Retrieve decision
        data = self.decision_logger.get_decision(decision_id)
        self.assertEqual(data["title"], "Chose SQLite over Redis")
        self.assertEqual(data["chosen"], "SQLite")
        self.assertIn("database", data["tags"])
        
        # Update outcome
        self.decision_logger.update_outcome(decision_id, "Worked out perfectly, extremely lightweight.")
        updated_data = self.decision_logger.get_decision(decision_id)
        self.assertEqual(updated_data["outcome"], "Worked out perfectly, extremely lightweight.")
        
        # Semantic search decision
        search_res = self.decision_logger.search_decisions("Chose SQLite database", n_results=1)
        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["title"], "Chose SQLite over Redis")

    def test_dead_ideas_tracker(self):
        # Bury an idea
        idea_id = self.dead_ideas_tracker.bury_idea(
            title="Voice-Activated IDE Connector",
            description="Controls code formatting using direct voice commands.",
            reason_abandoned="Speech-to-text latency was too high for coding productivity.",
            resurrection_triggers=["Local speech engine latency drops below 50ms"]
        )
        
        # Graveyard lists buried ideas
        graveyard = self.dead_ideas_tracker.list_graveyard()
        self.assertEqual(len(graveyard), 1)
        self.assertEqual(graveyard[0]["title"], "Voice-Activated IDE Connector")
        
        # Check resurrection candidates (using mock close vector)
        mock_query_vector = [0.1] * 768
        candidates = self.dead_ideas_tracker.check_resurrections(mock_query_vector, similarity_threshold=0.70)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["title"], "Voice-Activated IDE Connector")
        
        # Mark as resurrected
        self.dead_ideas_tracker.resurrect_idea(idea_id)
        graveyard_after = self.dead_ideas_tracker.list_graveyard()
        self.assertEqual(len(graveyard_after), 0)
