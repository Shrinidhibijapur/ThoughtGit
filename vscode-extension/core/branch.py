import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from core.config import BASE_DIR, METADATA_DB_PATH, DB_DIR

class BranchManager:
    def __init__(self):
        self.config_path = os.path.join(BASE_DIR, "config.json")
        self.metadata_db_path = METADATA_DB_PATH
        self.chroma_db_dir = DB_DIR
        self._init_metadata_db()
        self._init_config()

    def _init_metadata_db(self):
        """Initializes the metadata SQLite database and creates branch tables."""
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS branches (
                name TEXT PRIMARY KEY,
                created_at TEXT
            )
            """
        )
        # Ensure 'main' branch exists
        cursor.execute("SELECT name FROM branches WHERE name = 'main'")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO branches (name, created_at) VALUES (?, ?)",
                ("main", datetime.utcnow().isoformat())
            )
        conn.commit()
        conn.close()

    def _init_config(self):
        """Ensures the global config.json exists with active_branch set."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump({"active_branch": "main"}, f)

    def get_active_branch(self) -> str:
        """Reads the currently active branch from config.json."""
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
                return data.get("active_branch", "main")
        except (FileNotFoundError, json.JSONDecodeError):
            return "main"

    def switch_branch(self, name: str):
        """Switches the active branch to the specified name (must exist)."""
        if not self.branch_exists(name):
            raise ValueError(f"Branch '{name}' does not exist. Create it first.")
            
        with open(self.config_path, "w") as f:
            json.dump({"active_branch": name}, f)

    def create_branch(self, name: str):
        """Creates a new branch namespace."""
        name = name.strip().lower()
        if not name or " " in name or "_" in name:
            raise ValueError("Branch name must be alphanumeric and cannot contain spaces or underscores.")
            
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO branches (name, created_at) VALUES (?, ?)",
                (name, datetime.utcnow().isoformat())
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Branch '{name}' already exists.")
        finally:
            conn.close()

    def branch_exists(self, name: str) -> bool:
        """Checks if a branch exists in metadata database."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM branches WHERE name = ?", (name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def list_branches(self) -> List[str]:
        """Lists all branches in the repository."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM branches ORDER BY created_at ASC")
        branches = [row[0] for row in cursor.fetchall()]
        conn.close()
        return branches

    def merge_branch(self, source_branch: str, target_branch: str):
        """
        Merges the source branch into the target branch.
        Copies all chunks from source branch collections into target branch collections.
        """
        if not self.branch_exists(source_branch):
            raise ValueError(f"Source branch '{source_branch}' does not exist.")
        if not self.branch_exists(target_branch):
            raise ValueError(f"Target branch '{target_branch}' does not exist.")
        if source_branch == target_branch:
            raise ValueError("Cannot merge a branch into itself.")

        client = chromadb.PersistentClient(path=self.chroma_db_dir)
        collections = client.list_collections()
        
        # Source collection naming pattern: thoughts_{source_branch}_{YYYY}_{MM}
        source_prefix = f"thoughts_{source_branch}_"
        
        for coll in collections:
            name = coll.name
            if name.startswith(source_prefix):
                # Extract year and month suffix, e.g. "2026_07"
                suffix = name[len(source_prefix):]
                target_coll_name = f"thoughts_{target_branch}_{suffix}"
                
                # Fetch all elements from source collection (including embeddings)
                source_coll = client.get_collection(name=name)
                count = source_coll.count()
                if count == 0:
                    continue
                    
                data = source_coll.get(include=["documents", "metadatas", "embeddings"])
                
                if data and data["ids"]:
                    target_coll = client.get_or_create_collection(
                        name=target_coll_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    
                    # Update target collection name in metadata dictionary
                    updated_metadatas = []
                    for m in data["metadatas"]:
                        updated_meta = m.copy() if m else {}
                        updated_meta["branch"] = target_branch
                        updated_metadatas.append(updated_meta)
                        
                    target_coll.add(
                        ids=data["ids"],
                        embeddings=data["embeddings"],
                        documents=data["documents"],
                        metadatas=updated_metadatas
                    )
