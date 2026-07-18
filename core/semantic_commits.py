import sqlite3
import ollama
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.config import METADATA_DB_PATH

class SemanticCommitLogger:
    def __init__(self, model: str = "mistral"):
        self.metadata_db_path = METADATA_DB_PATH
        self.model = model
        self._init_db()

    def _init_db(self):
        """Initializes the semantic_commits table in metadata database."""
        import os
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                message TEXT,
                timestamp TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _check_ollama_status(self) -> bool:
        """Helper to verify if local Ollama is active."""
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=1)
            return True
        except Exception:
            return False

    def create_commit(self, topic: str, drift_event: Dict[str, Any]) -> str:
        """
        Generates a natural language commit message summarizing conceptual shifts,
        saves the commit record in SQLite, and returns the message.
        """
        from_period = drift_event.get("from_period", "A")
        to_period = drift_event.get("to_period", "B")
        drift_type = drift_event.get("drift_type", "evolution")
        summary = drift_event.get("summary", "")
        
        # Build prompt for commit message summarization
        prompt = (
            f"You are a developer logging a knowledge change commit. "
            f"Write a short, professional, Git-style commit message (max 80 chars) summarizing "
            f"this conceptual shift in the topic '{topic}':\n"
            f"Transition period: {from_period} -> {to_period}\n"
            f"Drift type: {drift_type}\n"
            f"Description: {summary}\n\n"
            f"Commit message should look like: 'Shift in [topic]: [brief summary of change]'"
        )

        commit_message = ""
        
        if self._check_ollama_status():
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                commit_message = response.message.content.strip().replace('"', '')
            except Exception as e:
                commit_message = self._get_fallback_message(topic, drift_event, f"Ollama Error: {e}")
        else:
            commit_message = self._get_fallback_message(topic, drift_event, None)

        # Force message size limits
        if len(commit_message) > 120:
            commit_message = commit_message[:117] + "..."

        # Save commit in SQLite
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO semantic_commits (topic, message, timestamp) VALUES (?, ?, ?)",
            (topic, commit_message, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        return commit_message

    def _get_fallback_message(self, topic: str, drift_event: Dict[str, Any], error: Optional[str]) -> str:
        """Generates a clean automated fallback commit message."""
        from_period = drift_event.get("from_period", "")
        to_period = drift_event.get("to_period", "")
        drift_type = drift_event.get("drift_type", "").upper()
        
        prefix = f"[{error}] " if error else ""
        return f"{prefix}Shift in {topic}: {drift_type} from {from_period} to {to_period}."

    def get_commit_history(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns chronological list of semantic commits, optionally filtered by topic."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if topic:
            cursor.execute(
                "SELECT * FROM semantic_commits WHERE topic = ? ORDER BY timestamp DESC",
                (topic.strip().lower(),)
            )
        else:
            cursor.execute("SELECT * FROM semantic_commits ORDER BY timestamp DESC")
            
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
