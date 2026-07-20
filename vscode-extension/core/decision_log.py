import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from core.config import METADATA_DB_PATH, DB_DIR
from core.embedder import EmbeddingEngine

class DecisionLogger:
    def __init__(self, embedder: EmbeddingEngine):
        self.metadata_db_path = METADATA_DB_PATH
        self.chroma_db_dir = DB_DIR
        self.embedder = embedder
        self._init_db()

    def _init_db(self):
        """Initializes the decisions table in the metadata SQLite database."""
        import os
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                chosen TEXT,
                alternatives TEXT,
                reasoning TEXT,
                assumptions TEXT,
                tags TEXT,
                outcome TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def log_decision(
        self,
        title: str,
        chosen: str,
        alternatives: List[str],
        reasoning: str,
        assumptions: str,
        tags: List[str]
    ) -> int:
        """
        Logs a decision in SQLite, generates a semantic summary,
        embeds it, and stores the vector in ChromaDB for semantic search.
        """
        # Save to SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO decisions (title, chosen, alternatives, reasoning, assumptions, tags, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                chosen,
                json.dumps(alternatives),
                reasoning,
                assumptions,
                ",".join(tags),
                None, # Outcome filled in later
                datetime.utcnow().isoformat()
            )
        )
        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Generate summary and store vector in ChromaDB
        summary = (
            f"Decision: {title}\n"
            f"Chosen Option: {chosen}\n"
            f"Rejected Alternatives: {', '.join(alternatives)}\n"
            f"Reasoning: {reasoning}\n"
            f"Assumptions: {assumptions}\n"
            f"Tags: {', '.join(tags)}"
        )
        
        try:
            embedding = self.embedder.embed(summary)
            
            client = chromadb.PersistentClient(path=self.chroma_db_dir)
            collection = client.get_or_create_collection(
                name="decisions_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            collection.add(
                ids=[str(decision_id)],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[{"title": title, "created_at": datetime.utcnow().isoformat()}]
            )
        except Exception as e:
            # Print error and proceed (database state is still preserved in SQLite)
            print(f"ChromaDB storage failed for decision vector: {e}")

        return decision_id

    def update_outcome(self, decision_id: int, outcome: str):
        """Updates the outcome of a past decision in SQLite."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE decisions SET outcome = ? WHERE id = ?",
            (outcome, decision_id)
        )
        conn.commit()
        conn.close()

    def get_decision(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a decision by ID."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            res = dict(row)
            res["alternatives"] = json.loads(res["alternatives"])
            res["tags"] = res["tags"].split(",") if res["tags"] else []
            return res
        return None

    def list_decisions(self) -> List[Dict[str, Any]]:
        """Returns all decisions ordered by created_at descending."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM decisions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            res = dict(r)
            res["alternatives"] = json.loads(res["alternatives"])
            res["tags"] = res["tags"].split(",") if res["tags"] else []
            results.append(res)
        return results

    def search_decisions(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds query, searches decisions collection in ChromaDB,
        and retrieves matched records from SQLite, returning them ordered by relevance.
        """
        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        try:
            collection = client.get_collection(name="decisions_embeddings")
        except Exception:
            # Collection doesn't exist yet, return empty list
            return []
            
        embedding = self.embedder.embed(query)
        count = collection.count()
        if count == 0:
            return []
            
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count)
        )
        
        decisions = []
        if results and results["ids"]:
            for idx, str_id in enumerate(results["ids"][0]):
                decision_id = int(str_id)
                data = self.get_decision(decision_id)
                if data:
                    data["distance"] = results["distances"][0][idx]
                    data["similarity"] = 1.0 - data["distance"]
                    decisions.append(data)
                    
        return decisions
