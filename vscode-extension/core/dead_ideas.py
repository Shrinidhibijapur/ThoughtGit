import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from core.config import METADATA_DB_PATH, DB_DIR
from core.embedder import EmbeddingEngine

class DeadIdeasTracker:
    def __init__(self, embedder: EmbeddingEngine):
        self.metadata_db_path = METADATA_DB_PATH
        self.chroma_db_dir = DB_DIR
        self.embedder = embedder
        self._init_db()

    def _init_db(self):
        """Initializes the dead_ideas table in the metadata SQLite database."""
        import os
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                reason_abandoned TEXT,
                resurrection_triggers TEXT,
                status TEXT DEFAULT 'buried',
                created_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def bury_idea(
        self,
        title: str,
        description: str,
        reason_abandoned: str,
        resurrection_triggers: List[str]
    ) -> int:
        """
        Stores an abandoned project/idea in SQLite,
        embeds the concept, and indexes it in ChromaDB for resurrection checks.
        """
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dead_ideas (title, description, reason_abandoned, resurrection_triggers, status, created_at)
            VALUES (?, ?, ?, ?, 'buried', ?)
            """,
            (
                title,
                description,
                reason_abandoned,
                json.dumps(resurrection_triggers),
                datetime.utcnow().isoformat()
            )
        )
        idea_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Build semantic representation for vector lookup
        summary = (
            f"Buried Idea: {title}\n"
            f"Description: {description}\n"
            f"Why Abandoned: {reason_abandoned}\n"
            f"Resurrection Triggers: {', '.join(resurrection_triggers)}"
        )
        
        try:
            embedding = self.embedder.embed(summary)
            
            client = chromadb.PersistentClient(path=self.chroma_db_dir)
            collection = client.get_or_create_collection(
                name="dead_ideas_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            collection.add(
                ids=[str(idea_id)],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[{"title": title, "created_at": datetime.utcnow().isoformat()}]
            )
        except Exception as e:
            print(f"ChromaDB storage failed for dead idea vector: {e}")

        return idea_id

    def resurrect_idea(self, idea_id: int):
        """Marks a buried idea as active (resurrected) in SQLite."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE dead_ideas SET status = 'resurrected' WHERE id = ?",
            (idea_id,)
        )
        conn.commit()
        conn.close()

    def get_idea(self, idea_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a dead idea by ID."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dead_ideas WHERE id = ?", (idea_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            res = dict(row)
            res["resurrection_triggers"] = json.loads(res["resurrection_triggers"])
            return res
        return None

    def list_graveyard(self) -> List[Dict[str, Any]]:
        """Returns all ideas with status 'buried' ordered by created_at descending."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dead_ideas WHERE status = 'buried' ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            res = dict(r)
            res["resurrection_triggers"] = json.loads(res["resurrection_triggers"])
            results.append(res)
        return results

    def check_resurrections(
        self,
        current_context_embedding: List[float],
        similarity_threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Compares current text context with buried ideas in vector space.
        Returns matching dead ideas with similarity above threshold to trigger warnings.
        """
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        try:
            collection = client.get_collection(name="dead_ideas_embeddings")
        except Exception:
            return []
            
        count = collection.count()
        if count == 0:
            return []
            
        results = collection.query(
            query_embeddings=[current_context_embedding],
            n_results=min(5, count)
        )
        
        candidates = []
        max_distance = 1.0 - similarity_threshold
        
        if results and results["ids"]:
            for idx, str_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][idx]
                if distance <= max_distance:
                    idea_id = int(str_id)
                    idea = self.get_idea(idea_id)
                    if idea and idea["status"] == "buried":
                        idea["distance"] = distance
                        idea["similarity"] = 1.0 - distance
                        candidates.append(idea)
                        
        return candidates
