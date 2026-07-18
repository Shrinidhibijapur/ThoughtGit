import numpy as np
import warnings
from datetime import datetime
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
        Retrieves all similar thoughts across time buckets, clusters each bucket
        using HDBSCAN to find the conceptual centroid, tracks centroid drift over time,
        and classifies each drift event.
        """
        # Step 1: Retrieve all semantically similar chunks across all collections
        chunks = self.store.get_all_chunks_for_diff(
            query_embedding=query_embedding,
            threshold=0.40,
            branch=branch
        )
        
        if not chunks:
            return {
                "topic": topic,
                "snapshots": [],
                "drift_events": []
            }

        # Step 2: Group chunks by time bucket (month / collection)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            coll_name = chunk["collection"]
            buckets.setdefault(coll_name, []).append(chunk)

        # Sort collections chronologically
        sorted_coll_names = self.store.list_collections(branch=branch)
        active_colls = [c for c in sorted_coll_names if c in buckets]

        snapshots = []
        centroids = {}

        # Step 3: Compute centroid for each collection bucket
        for coll_name in active_colls:
            coll_chunks = buckets[coll_name]
            embeddings = [c["embedding"] for c in coll_chunks if c.get("embedding") is not None]
            
            if not embeddings:
                continue
                
            # Run HDBSCAN if we have enough data, otherwise fallback to simple mean
            if len(embeddings) >= min_cluster_size:
                try:
                    # Convert to numpy array
                    X = np.array(embeddings)
                    
                    # HDBSCAN clustering using cosine metric
                    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine")
                    clusterer.fit(X)
                    labels = clusterer.labels_
                    
                    # Find dominant cluster (exclude noise -1)
                    unique_labels = set(labels)
                    unique_labels.discard(-1)
                    
                    if unique_labels:
                        # Find the label with the most points
                        label_counts = {l: np.count_nonzero(labels == l) for l in unique_labels}
                        dominant_label = max(label_counts, key=label_counts.get)
                        
                        # Get embeddings belonging to dominant cluster
                        dominant_indices = np.where(labels == dominant_label)[0]
                        dominant_embeddings = [embeddings[idx] for idx in dominant_indices]
                        centroid = self._compute_centroid(dominant_embeddings)
                        
                        # Store info
                        clustered_chunks = [coll_chunks[idx] for idx in dominant_indices]
                    else:
                        # Fallback: all points are noise, compute mean of all points
                        centroid = self._compute_centroid(embeddings)
                        clustered_chunks = coll_chunks
                except Exception:
                    # Fallback on any failure
                    centroid = self._compute_centroid(embeddings)
                    clustered_chunks = coll_chunks
            else:
                centroid = self._compute_centroid(embeddings)
                clustered_chunks = coll_chunks

            # Record snapshot details
            avg_len = sum(len(c["text"]) for c in coll_chunks) / len(coll_chunks)
            centroids[coll_name] = centroid
            
            # Format friendly time label, e.g. thoughts_main_2026_07 -> "2026-07"
            parts = coll_name.split("_")
            time_label = f"{parts[-2]}-{parts[-1]}" if len(parts) >= 3 else coll_name
            
            snapshots.append({
                "collection": coll_name,
                "time_label": time_label,
                "chunks_count": len(coll_chunks),
                "avg_text_length": avg_len,
                "sample_texts": [c["text"] for c in clustered_chunks[:3]]
            })

        # Step 4 & 5: Compute drift between successive centroids and classify drift events
        drift_events = []
        for i in range(len(active_colls) - 1):
            coll_a = active_colls[i]
            coll_b = active_colls[i+1]
            
            centroid_a = centroids[coll_a]
            centroid_b = centroids[coll_b]
            
            distance = self._cosine_distance(centroid_a, centroid_b)
            
            # Retrieve metadata for length calculations
            snap_a = next(s for s in snapshots if s["collection"] == coll_a)
            snap_b = next(s for s in snapshots if s["collection"] == coll_b)
            
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
                # Compare average length: increase of > 15% counts as deepened
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
