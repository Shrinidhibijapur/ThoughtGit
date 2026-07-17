from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core.models import RawThought
from core.embedder import EmbeddingEngine
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

app = FastAPI(title="ThoughtGit API Server")

# Global instances of the core engines
engine = EmbeddingEngine()
store = ThoughtStore()
diff_engine = SemanticDiffEngine(store)

class IngestRequest(BaseModel):
    content: str
    source: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = Field(default_factory=dict)

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Backend service health check."""
    return {"status": "healthy"}

@app.post("/ingest")
def ingest_thought(request: IngestRequest) -> Dict[str, Any]:
    """Ingests a new thought, chunks it, embeds it, and stores it in the vector DB."""
    try:
        thought = RawThought(
            content=request.content,
            source=request.source,
            timestamp=request.timestamp,
            metadata=request.metadata
        )
        
        # Embed and store chunks
        chunks = engine.embed_thought(thought)
        if chunks:
            store.store_chunks_batch(chunks)
            
        return {
            "status": "success",
            "chunks_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recall")
def recall_context(
    query: str = Query(..., description="Query string to find relevant memory context"),
    n_results: int = Query(5, description="Number of results to retrieve"),
    since: Optional[datetime] = Query(None, description="Start date filter"),
    until: Optional[datetime] = Query(None, description="End date filter"),
    branch: str = Query("main", description="Target repository branch namespace")
) -> List[Dict[str, Any]]:
    """Retrieves past thoughts matching the query, sorted by relevance distance."""
    try:
        query_vector = engine.embed(query)
        results = store.query_across_time(
            query_embedding=query_vector,
            n_results=n_results,
            since=since,
            until=until,
            branch=branch
        )
        
        # Clean results (remove raw float embeddings for JSON response size)
        cleaned_results = []
        for r in results:
            cleaned = r.copy()
            cleaned.pop("embedding", None) # Ensure embedding isn't returned
            cleaned_results.append(cleaned)
            
        return cleaned_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diff")
def compute_semantic_diff(
    topic: str = Query(..., description="The concept topic to analyze drift for"),
    min_cluster_size: int = Query(2, description="Minimum size of cluster for HDBSCAN"),
    branch: str = Query("main", description="Target repository branch namespace")
) -> Dict[str, Any]:
    """Computes semantic drift snapshots and centroid shift events for a topic."""
    try:
        query_vector = engine.embed(topic)
        analysis = diff_engine.analyze_drift(
            topic=topic,
            query_embedding=query_vector,
            min_cluster_size=min_cluster_size,
            branch=branch
        )
        
        # Clean embedding arrays from snapshots in response
        if "snapshots" in analysis:
            for s in analysis["snapshots"]:
                s.pop("embedding", None)
                
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
