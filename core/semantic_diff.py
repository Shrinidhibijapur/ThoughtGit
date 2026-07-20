import numpy as np
import warnings
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import HDBSCAN
from core.config import DRIFT_THRESHOLD, SIMILARITY_THRESHOLD
from core.thought_store import ThoughtStore

# Silence scikit-learn HDBSCAN copy FutureWarning warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SemanticDiffEngine:
    def __init__(self, store: ThoughtStore):
        self.store = store

    def _cosine_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Computes cosine distance between two vectors."""
        dot = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 1.0
        similarity = dot / (norm_u * norm_v)
        return float(1.0 - similarity)

    def _compute_centroid(self, embeddings: List[List[float]]) -> np.ndarray:
        """Computes the normalized centroid of a list of embeddings."""
        arr = np.array(embeddings)
        mean_vector = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
        return mean_vector

    def analyze_drift(
        self,
        topic: str,
        query_embedding: List[float],
        min_cluster_size: int = 2,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Retrieves all similar thoughts for a topic, groups them by exact timestamp
        to represent each note-saving event, tracks drift between consecutive events,
        and formats timestamps to Indian Standard Time (GMT+5:30).
        """
        # Step 1: Retrieve all semantically similar chunks across all collections
        chunks = self.store.get_all_chunks_for_diff(
            query_embedding=query_embedding,
            topic=topic,
            threshold=0.40,
            branch=branch
        )
        
        if not chunks:
            return {
                "topic": topic,
                "snapshots": [],
                "drift_events": []
            }

        # Step 2: Group chunks by their exact timestamp (representing one saved thought event)
        timestamp_groups: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            ts = chunk.get("timestamp") or chunk.get("metadata", {}).get("timestamp", "")
            group_key = ts if ts else chunk["collection"]
            timestamp_groups.setdefault(group_key, []).append(chunk)

        # Sort the group keys chronologically
        def get_group_time(group_key):
            if group_key:
                try:
                    return datetime.fromisoformat(group_key.replace("Z", "+00:00"))
                except ValueError:
                    pass
            return datetime.min

        sorted_group_keys = sorted(timestamp_groups.keys(), key=get_group_time)

        IST = timezone(timedelta(hours=5, minutes=30))
        snapshots = []
        centroids = {}

        # Step 3: Format snapshots and compute centroids for each saved event
        for idx, key in enumerate(sorted_group_keys):
            group_chunks = timestamp_groups[key]
            embeddings = [c["embedding"] for c in group_chunks if c.get("embedding") is not None]
            
            if not embeddings:
                continue
                
            centroid = self._compute_centroid(embeddings)
            group_id = f"group_{idx}"
            centroids[group_id] = centroid
            
            # Format fallback friendly time label, e.g. thoughts_main_2026_07 -> "2026-07"
            coll_name = group_chunks[0]["collection"]
            parts = coll_name.split("_")
            fallback_label = f"{parts[-2]}-{parts[-1]}" if len(parts) >= 3 else coll_name
            
            # Format time label to Indian Standard Time (UTC +5:30)
            time_label = fallback_label
            if key and not key.startswith("thoughts_"):
                try:
                    dt = datetime.fromisoformat(key.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ist_dt = dt.astimezone(IST)
                    time_label = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            
            avg_len = sum(len(c["text"]) for c in group_chunks) / len(group_chunks)
            snapshots.append({
                "collection": group_chunks[0]["collection"],
                "time_label": time_label,
                "chunks_count": len(group_chunks),
                "avg_text_length": avg_len,
                "sample_texts": [c["text"] for c in group_chunks[:3]]
            })

        # Step 4 & 5: Compute drift between successive saved events
        drift_events = []
        for i in range(len(sorted_group_keys) - 1):
            snap_a = snapshots[i]
            snap_b = snapshots[i+1]
            
            centroid_a = centroids[f"group_{i}"]
            centroid_b = centroids[f"group_{i+1}"]
            
            distance = self._cosine_distance(centroid_a, centroid_b)
            
            len_a = snap_a["avg_text_length"]
            len_b = snap_b["avg_text_length"]
            
            # Classification logic
            if distance > 0.6:
                drift_type = "major_shift"
                summary = "Completely different direction or context shift."
            elif distance > 0.4:
                drift_type = "changed_direction"
                summary = "Significant revision of core concepts."
            elif distance > DRIFT_THRESHOLD:
                if len_b > len_a * 1.15:
                    drift_type = "deepened"
                    summary = "Understanding deepened with more detailed descriptions."
                else:
                    drift_type = "refined"
                    summary = "Understanding refined with more precise concepts."
            else:
                drift_type = "reinforced"
                summary = "Understanding reinforced and stable."
                
            drift_events.append({
                "from_period": snap_a["time_label"],
                "to_period": snap_b["time_label"],
                "distance": distance,
                "drift_type": drift_type,
                "summary": summary
            })

        return {
            "topic": topic,
            "snapshots": snapshots,
            "drift_events": drift_events
        }
