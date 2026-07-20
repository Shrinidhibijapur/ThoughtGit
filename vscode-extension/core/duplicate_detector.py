from typing import List, Dict, Any, Optional
from core.thought_store import ThoughtStore
from core.config import DUPLICATE_THRESHOLD

class DuplicateDetector:
    def __init__(self, store: ThoughtStore):
        self.store = store
        self.threshold = DUPLICATE_THRESHOLD

    def check_duplicate(
        self,
        text: str,
        embedding: List[float],
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Queries the database for the single most similar chunk.
        If the cosine distance is below DUPLICATE_THRESHOLD (0.15),
        flags it as a duplicate concept.
        """
        results = self.store.query_across_time(
            query_embedding=embedding,
            n_results=1,
            branch=branch
        )
        
        if results:
            best_match = results[0]
            distance = best_match["distance"]
            similarity = best_match["similarity"]
            
            is_duplicate = distance < self.threshold
            
            return {
                "is_duplicate": is_duplicate,
                "similarity": similarity,
                "distance": distance,
                "matched_chunk": {
                    "id": best_match["id"],
                    "text": best_match["text"],
                    "collection": best_match["collection"],
                    "timestamp": best_match["metadata"].get("timestamp")
                } if is_duplicate else None
            }
            
        return {
            "is_duplicate": False,
            "similarity": 0.0,
            "distance": 2.0,
            "matched_chunk": None
        }
