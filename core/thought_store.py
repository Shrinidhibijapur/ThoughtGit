import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from core.config import DB_DIR, SIMILARITY_THRESHOLD
from core.models import EmbeddedChunk

class ThoughtStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_DIR)

    def _parse_collection_date(self, collection_name: str) -> Optional[datetime]:
        """Parses the year and month from collection name format: thoughts_{branch}_{YYYY}_{MM}"""
        parts = collection_name.split("_")
        if len(parts) >= 3:
            try:
                year = int(parts[-2])
                month = int(parts[-1])
                return datetime(year, month, 1)
            except ValueError:
                return None
        return None

    def _get_branch_from_collection(self, collection_name: str) -> str:
        """Extracts branch name from collection name: thoughts_{branch}_{YYYY}_{MM}"""
        parts = collection_name.split("_")
        if len(parts) >= 3 and parts[0] == "thoughts":
            # Extract everything between 'thoughts' and '{YYYY}_{MM}'
            return "_".join(parts[1:-2])
        return ""

    def store_chunk(self, chunk: EmbeddedChunk):
        """Stores a single EmbeddedChunk into its corresponding collection."""
        collection = self.client.get_or_create_collection(
            name=chunk.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        collection.add(
            ids=[chunk.chunk_id],
            embeddings=[chunk.embedding],
            documents=[chunk.text],
            metadatas=[chunk.metadata]
        )

    def store_chunks_batch(self, chunks: List[EmbeddedChunk]):
        """Stores a batch of EmbeddedChunks, grouping them to minimize database calls."""
        if not chunks:
            return
            
        # Group by collection name
        grouped: Dict[str, List[EmbeddedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.collection_name, []).append(chunk)
            
        for coll_name, coll_chunks in grouped.items():
            collection = self.client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            ids = [c.chunk_id for c in coll_chunks]
            embeddings = [c.embedding for c in coll_chunks]
            documents = [c.text for c in coll_chunks]
            metadatas = [c.metadata for c in coll_chunks]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def list_collections(self, branch: Optional[str] = None) -> List[str]:
        """Lists all thought collections, sorted chronologically."""
        collections = self.client.list_collections()
        thought_colls = []
        for coll in collections:
            name = coll.name
            if name.startswith("thoughts_"):
                if branch is None or self._get_branch_from_collection(name) == branch:
                    thought_colls.append(name)
                    
        # Sort chronologically based on parsed date
        def get_sort_key(name: str):
            dt = self._parse_collection_date(name)
            return dt if dt else datetime.min
            
        thought_colls.sort(key=get_sort_key)
        return thought_colls

    def query_across_time(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Queries time-sliced collections for the given branch,
        merges results, and returns the top elements sorted by distance.
        """
        collections = self.list_collections(branch=branch)
        merged_results = []
        
        for name in collections:
            coll_date = self._parse_collection_date(name)
            if coll_date:
                if since and coll_date < datetime(since.year, since.month, 1):
                    continue
                if until and coll_date > datetime(until.year, until.month, 1):
                    continue
                    
            collection = self.client.get_collection(name=name)
            
            # Query the collection
            # To ensure we get enough candidates for merging, retrieve n_results from each collection
            count = collection.count()
            if count == 0:
                continue
                
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count)
            )
            
            # Parse results
            if results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][i]
                    text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    merged_results.append({
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "distance": distance,
                        "collection": name,
                        "similarity": 1.0 - distance
                    })
                    
        # Sort by distance (smaller distance first)
        merged_results.sort(key=lambda x: x["distance"])
        return merged_results[:n_results]

    def get_all_chunks_for_diff(
        self,
        query_embedding: List[float],
        topic: Optional[str] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all chunks matching the branch and query embedding with similarity > threshold,
        grouped/tagged by their chronological collection name.
        """
        collections = self.list_collections(branch=branch)
        matched_chunks = []
        
        # Max cosine distance for similarity threshold
        max_distance = 1.0 - threshold
        
        for name in collections:
            collection = self.client.get_collection(name=name)
            count = collection.count()
            if count == 0:
                continue
                
            results = None
            if topic:
                try:
                    # Try filtering strictly by topic hint first
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(100, count),
                        where={"topic_hint": topic},
                        include=["documents", "metadatas", "distances", "embeddings"]
                    )
                except Exception:
                    results = None
                    
            # Fallback to general vector search if no metadata matches or filter is empty
            if not results or not results["ids"] or not results["ids"][0]:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(100, count),
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
            
            if results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    distance = results["distances"][0][i]
                    if distance <= max_distance:
                        matched_chunks.append({
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "embedding": results["embeddings"][0][i] if results.get("embeddings") else None,
                            "distance": distance,
                            "collection": name,
                            "timestamp": results["metadatas"][0][i].get("timestamp", "")
                        })
                        
        return matched_chunks
