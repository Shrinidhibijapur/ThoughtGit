from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class RawThought(BaseModel):
    content: str
    source: str  # e.g., "vscode", "obsidian", "cli"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddedChunk(BaseModel):
    chunk_id: str
    text: str
    embedding: List[float]
    source: str
    timestamp: datetime
    collection_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
