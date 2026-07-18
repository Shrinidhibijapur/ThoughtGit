from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from core.thought_store import ThoughtStore
from core.config import SIMILARITY_THRESHOLD

class TimeMachine:
    def __init__(self, store: ThoughtStore):
        self.store = store

    def recall_as_of(
        self,
        topic: str,
        query_embedding: List[float],
        as_of_date: datetime,
        n_results: int = 5,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Reconstructs what the system understood about a topic up to a specific date,
        excluding any future collections or chunks.
        """
        collections = self.store.list_collections(branch=branch)
        merged_results = []
        
        # Max collection date is the year and month of as_of_date
        as_of_cutoff_epoch = as_of_date.timestamp()
        
        for name in collections:
            coll_date = self.store._parse_collection_date(name)
            if coll_date:
                # Exclude future collections entirely
                if coll_date.year > as_of_date.year or (
                    coll_date.year == as_of_date.year and coll_date.month > as_of_date.month
                ):
                    continue
                    
            collection = self.store.client.get_collection(name=name)
            count = collection.count()
            if count == 0:
                continue
                
            # Query collection with metadata filter: timestamp_epoch <= as_of_date timestamp
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                where={"timestamp_epoch": {"$lte": as_of_cutoff_epoch}}
            )
            
            if results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    merged_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "collection": name,
                        "similarity": 1.0 - results["distances"][0][i]
                    })
                    
        merged_results.sort(key=lambda x: x["distance"])
        return merged_results[:n_results]

    def _cosine_similarity(self, u: List[float], v: List[float]) -> float:
        """Helper to compute cosine similarity between two lists."""
        arr_u = np.array(u)
        arr_v = np.array(v)
        dot = np.dot(arr_u, arr_v)
        norm_u = np.linalg.norm(arr_u)
        norm_v = np.linalg.norm(arr_v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(dot / (norm_u * norm_v))

    def compare_understanding(
        self,
        topic: str,
        query_embedding: List[float],
        date_a: datetime,
        date_b: datetime,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Compares understanding of a topic at two different points in time side-by-side.
        Identifies what new knowledge/concepts were learned between date_a and date_b.
        """
        if date_a > date_b:
            date_a, date_b = date_b, date_a
            
        # Recall items as of both dates (retrieve up to 50 for complete comparison)
        snapshots_a = self.recall_as_of(topic, query_embedding, date_a, n_results=50, branch=branch)
        snapshots_b = self.recall_as_of(topic, query_embedding, date_b, n_results=50, branch=branch)
        
        # Identify chunks in B that are not present in A
        ids_a = {item["id"] for item in snapshots_a}
        
        new_learnings = []
        for item in snapshots_b:
            if item["id"] not in ids_a:
                # To verify it's a conceptually new thought, check similarity to all items in A.
                # If it's too similar, it's just a duplicate/revision, not a "new learning".
                # For this, retrieve chunk embeddings. Since recall doesn't fetch embeddings by default,
                # we match based on text/ID or metadata.
                new_learnings.append({
                    "id": item["id"],
                    "text": item["text"],
                    "timestamp": item["metadata"].get("timestamp"),
                    "collection": item["collection"]
                })
                
        return {
            "topic": topic,
            "date_a": date_a.strftime("%Y-%m-%d"),
            "date_b": date_b.strftime("%Y-%m-%d"),
            "snapshots_count_a": len(snapshots_a),
            "snapshots_count_b": len(snapshots_b),
            "learnings_at_date_a": [item["text"] for item in snapshots_a[:5]],
            "learnings_at_date_b": [item["text"] for item in snapshots_b[:5]],
            "new_learnings_since_a": new_learnings
        }
