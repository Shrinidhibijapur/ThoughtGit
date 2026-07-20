import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

class LearningVelocityEngine:
    def __init__(self, store: ThoughtStore, diff_engine: SemanticDiffEngine):
        self.store = store
        self.diff_engine = diff_engine

    def calculate_velocity(
        self,
        topic: str,
        query_embedding: List[float],
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Calculates:
        1. Volume velocity (number of chunks added per month/period).
        2. Conceptual velocity (how fast the semantic centroid moves over time).
        """
        # Run semantic diff to obtain snapshots and drift events
        analysis = self.diff_engine.analyze_drift(
            topic=topic,
            query_embedding=query_embedding,
            branch=branch
        )
        
        snapshots = analysis.get("snapshots", [])
        drift_events = analysis.get("drift_events", [])
        
        if not snapshots:
            return {
                "topic": topic,
                "volume_velocity": 0.0,
                "conceptual_velocity": 0.0,
                "status": "No memory data available",
                "total_chunks": 0
            }

        # 1. Calculate Volume Velocity (average chunks per snapshot period)
        total_chunks = sum(s["chunks_count"] for s in snapshots)
        volume_velocity = total_chunks / len(snapshots)
        
        # Determine volume trend (comparing last period chunk count to previous average)
        if len(snapshots) > 1:
            last_count = snapshots[-1]["chunks_count"]
            prev_avg = sum(s["chunks_count"] for s in snapshots[:-1]) / (len(snapshots) - 1)
            volume_trend = "increasing" if last_count > prev_avg else ("stable" if last_count == prev_avg else "decreasing")
        else:
            volume_trend = "stable"

        # 2. Calculate Conceptual Velocity (average cosine distance between periods)
        if drift_events:
            distances = [e["distance"] for e in drift_events]
            conceptual_velocity = float(np.mean(distances))
        else:
            conceptual_velocity = 0.0

        # 3. Classify learning status based on velocity intersection
        # High conceptual velocity (> 0.25) = evolving
        # High volume (> 2.0) with low conceptual velocity = stuck in loop
        if volume_velocity >= 2.0:
            if conceptual_velocity > 0.25:
                status = "rapidly_evolving"
            else:
                status = "stagnant_volume" # writing a lot, but ideas are not changing
        else:
            if conceptual_velocity > 0.25:
                status = "efficient_learning" # few notes, but significant shifts
            else:
                status = "stable_reinforcement"

        return {
            "topic": topic,
            "total_chunks": total_chunks,
            "periods_tracked": len(snapshots),
            "volume_velocity": volume_velocity,
            "volume_trend": volume_trend,
            "conceptual_velocity": conceptual_velocity,
            "status": status
        }
