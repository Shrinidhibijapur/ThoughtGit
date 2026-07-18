import sqlite3
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.config import METADATA_DB_PATH

class ForgettingCurveTracker:
    def __init__(self):
        self.metadata_db_path = METADATA_DB_PATH
        self._init_db()

    def _init_db(self):
        """Initializes the spaced_repetition SQLite table."""
        import os
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spaced_repetition (
                topic TEXT PRIMARY KEY,
                last_accessed TEXT,
                stability REAL DEFAULT 1.0
            )
            """
        )
        conn.commit()
        conn.close()

    def record_access(self, topic: str):
        """Records access/review of a topic, updating its retention stability."""
        topic = topic.strip().lower()
        if not topic:
            return

        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_accessed, stability FROM spaced_repetition WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        
        now_str = datetime.utcnow().isoformat()
        
        if row:
            last_accessed_str, stability = row
            last_accessed = datetime.fromisoformat(last_accessed_str)
            
            # Days since last review
            delta = datetime.utcnow() - last_accessed
            t = max(0.1, delta.total_seconds() / 86400.0) # in days
            
            # Retention strength R = e^(-t/S)
            retention = math.exp(-t / stability)
            
            # Spacing effect: if reviewed when retention is still good, stability doubles.
            # If reviewed after forgetting, stability resets/stabilizes at 1.0 day.
            if retention >= 0.4:
                new_stability = stability * 2.0
            else:
                new_stability = 1.0
                
            cursor.execute(
                "UPDATE spaced_repetition SET last_accessed = ?, stability = ? WHERE topic = ?",
                (now_str, new_stability, topic)
            )
        else:
            # First insertion: stability of 1.0 day
            cursor.execute(
                "INSERT INTO spaced_repetition (topic, last_accessed, stability) VALUES (?, ?, ?)",
                (topic, now_str, 1.0)
            )
            
        conn.commit()
        conn.close()

    def get_memory_strength(self, topic: str) -> float:
        """Returns the current memory strength score R (0.0 to 1.0) for a topic."""
        topic = topic.strip().lower()
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_accessed, stability FROM spaced_repetition WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return 1.0 # Default to fully remembered
            
        last_accessed_str, stability = row
        last_accessed = datetime.fromisoformat(last_accessed_str)
        
        delta = datetime.utcnow() - last_accessed
        t = delta.total_seconds() / 86400.0 # days
        
        # R = e^(-t/S)
        return float(math.exp(-t / stability))

    def get_review_schedule(self) -> List[Dict[str, Any]]:
        """Lists all topics needing spaced repetition review (memory strength R < 0.4)."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT topic, last_accessed, stability FROM spaced_repetition")
        rows = cursor.fetchall()
        conn.close()
        
        review_list = []
        for topic, last_accessed_str, stability in rows:
            last_accessed = datetime.fromisoformat(last_accessed_str)
            delta = datetime.utcnow() - last_accessed
            t = delta.total_seconds() / 86400.0 # days
            
            # Calculate retention
            retention = float(math.exp(-t / stability))
            
            # If retention drops below 40% threshold, flag for review
            if retention < 0.4:
                review_list.append({
                    "topic": topic,
                    "last_accessed": last_accessed_str,
                    "retention_strength": retention,
                    "stability_days": stability,
                    "days_since_access": t
                })
                
        # Sort so the most forgotten topic appears first
        review_list.sort(key=lambda x: x["retention_strength"])
        return review_list
