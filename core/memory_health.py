from datetime import datetime, timedelta
from typing import Dict, Any, List
import sqlite3
from core.thought_store import ThoughtStore
from core.forgetting_curve import ForgettingCurveTracker
from core.config import METADATA_DB_PATH

class MemoryHealthEngine:
    def __init__(self, store: ThoughtStore, spacing_tracker: ForgettingCurveTracker):
        self.store = store
        self.spacing_tracker = spacing_tracker
        self.metadata_db_path = METADATA_DB_PATH

    def calculate_health_report(self, branch: str = "main") -> Dict[str, Any]:
        """
        Calculates a composite memory health index score (0-100) combining:
        1. Activity (35%): Frequency of recently ingested thoughts (last 30 days).
        2. Diversity (35%): Number of unique topics/tags in memory.
        3. Spacing (30%): Ratio of topics kept at high retention (>40%) on forgetting curve.
        """
        # 1. Calculate Activity Score (35 pts max)
        # Query total chunks ingested in the last 30 days across monthly collections
        collections = self.store.list_collections(branch=branch)
        recent_chunks_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        cutoff_epoch = cutoff_date.timestamp()

        for name in collections:
            coll_date = self.store._parse_collection_date(name)
            # Only search collections representing current or recent months
            if coll_date and coll_date >= datetime(cutoff_date.year, cutoff_date.month, 1):
                try:
                    coll = self.store.client.get_collection(name=name)
                    count = coll.count()
                    if count > 0:
                        res = coll.get(where={"timestamp_epoch": {"$gte": cutoff_epoch}})
                        if res and res["ids"]:
                            recent_chunks_count += len(res["ids"])
                except Exception:
                    pass

        # Activity rating: 10 chunks or more gets full score
        activity_score = min(35.0, recent_chunks_count * 3.5)

        # 2. Calculate Diversity Score (35 pts max)
        # Unique topics are derived from the spaced repetition database list
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT topic FROM spaced_repetition")
            topics = [row[0] for row in cursor.fetchall()]
        except Exception:
            topics = []
        conn.close()

        # Diversity rating: 5 or more unique topics gets full score
        diversity_score = min(35.0, len(topics) * 7.0)

        # 3. Calculate Spacing Score (30 pts max)
        # Ratio of topics with current retention strength >= 0.40
        retained_count = 0
        total_tracked_topics = len(topics)

        for topic in topics:
            strength = self.spacing_tracker.get_memory_strength(topic)
            if strength >= 0.40:
                retained_count += 1

        if total_tracked_topics > 0:
            spacing_ratio = retained_count / total_tracked_topics
            spacing_score = spacing_ratio * 30.0
        else:
            spacing_score = 30.0 # Default to max score if no topics are tracked yet

        # Aggregate Score
        total_health_score = round(activity_score + diversity_score + spacing_score, 1)

        return {
            "health_score": total_health_score,
            "metrics": {
                "activity": {
                    "score": round(activity_score, 1),
                    "max": 35.0,
                    "recent_chunks_count": recent_chunks_count
                },
                "diversity": {
                    "score": round(diversity_score, 1),
                    "max": 35.0,
                    "unique_topics_count": len(topics)
                },
                "spacing": {
                    "score": round(spacing_score, 1),
                    "max": 30.0,
                    "retained_topics_ratio": round(retained_count / total_tracked_topics, 2) if total_tracked_topics > 0 else 1.0
                }
            },
            "interpretation": self._interpret_health_score(total_health_score)
        }

    def _interpret_health_score(self, score: float) -> str:
        if score >= 85.0:
            return "excellent: Active learning and robust spaced repetition review habits."
        elif score >= 60.0:
            return "good: Steady activity but some topics are fading from memory."
        elif score >= 35.0:
            return "fair: Low recent ingestion or review spacing needs attention."
        else:
            return "poor: Highly inactive memory vault. Resume indexing notes."
