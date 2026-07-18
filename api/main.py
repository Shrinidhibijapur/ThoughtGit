from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field

from core.models import RawThought
from core.embedder import EmbeddingEngine
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

# Extended Feature Imports
from core.branch import BranchManager
from core.time_machine import TimeMachine
from core.duplicate_detector import DuplicateDetector
from core.decision_log import DecisionLogger
from core.dead_ideas import DeadIdeasTracker
from core.learning_velocity import LearningVelocityEngine
from core.forgetting_curve import ForgettingCurveTracker
from core.ai_mentor import AIMentor
from core.semantic_commits import SemanticCommitLogger
from core.memory_health import MemoryHealthEngine

app = FastAPI(title="ThoughtGit API Server")

# Global instances of core and extended engines
engine = EmbeddingEngine()
store = ThoughtStore()
diff_engine = SemanticDiffEngine(store)

branch_manager = BranchManager()
time_machine = TimeMachine(store)
duplicate_detector = DuplicateDetector(store)
decision_logger = DecisionLogger(engine)
dead_ideas_tracker = DeadIdeasTracker(engine)
velocity_engine = LearningVelocityEngine(store, diff_engine)
forgetting_tracker = ForgettingCurveTracker()
ai_mentor = AIMentor(store)
commit_logger = SemanticCommitLogger()
health_engine = MemoryHealthEngine(store, forgetting_tracker)

class IngestRequest(BaseModel):
    content: str
    source: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ----------------------------------------------------
# 1. Foundation Endpoints (Part 1)
# ----------------------------------------------------
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
            cleaned.pop("embedding", None)
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

# ----------------------------------------------------
# 2. Branch Management Endpoints
# ----------------------------------------------------
@app.get("/branch")
def get_branches() -> Dict[str, Any]:
    return {
        "active_branch": branch_manager.get_active_branch(),
        "branches": branch_manager.list_branches()
    }

@app.post("/branch")
def create_branch(name: str = Body(..., embed=True)) -> Dict[str, str]:
    try:
        branch_manager.create_branch(name)
        return {"status": "success", "message": f"Branch '{name}' created."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/branch/switch")
def switch_branch(name: str = Body(..., embed=True)) -> Dict[str, str]:
    try:
        branch_manager.switch_branch(name)
        return {"status": "success", "message": f"Switched to branch '{name}'."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/branch/merge")
def merge_branch(
    source: str = Body(..., description="Source branch to copy from"),
    target: str = Body(..., description="Target branch to write to")
) -> Dict[str, str]:
    try:
        branch_manager.merge_branch(source, target)
        return {"status": "success", "message": f"Merged branch '{source}' into '{target}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 3. Time Machine Endpoints
# ----------------------------------------------------
@app.get("/timemachine")
def timemachine_recall(
    topic: str = Query(...),
    as_of: datetime = Query(...),
    compare_to: Optional[datetime] = Query(None),
    branch: str = Query("main")
) -> Dict[str, Any]:
    try:
        query_vector = engine.embed(topic)
        if compare_to:
            # Side-by-side comparison mode
            return time_machine.compare_understanding(
                topic=topic,
                query_embedding=query_vector,
                date_a=as_of,
                date_b=compare_to,
                branch=branch
            )
        else:
            # Single point recall mode
            results = time_machine.recall_as_of(
                topic=topic,
                query_embedding=query_vector,
                as_of_date=as_of,
                branch=branch
            )
            for r in results:
                r.pop("embedding", None)
            return {"topic": topic, "as_of": as_of.isoformat(), "snapshots": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 4. Duplicate Check Endpoint
# ----------------------------------------------------
@app.post("/check_duplicate")
def check_duplicate(payload: Dict[str, str] = Body(...)) -> Dict[str, Any]:
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing parameter 'text'")
    try:
        query_vector = engine.embed(text)
        # Check defaults to active configuration branch
        branch = branch_manager.get_active_branch()
        return duplicate_detector.check_duplicate(text, query_vector, branch=branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 5. Decision Logging Endpoints
# ----------------------------------------------------
class DecisionLogRequest(BaseModel):
    title: str
    chosen: str
    alternatives: List[str]
    reasoning: str
    assumptions: str
    tags: List[str]

@app.post("/decisions")
def log_decision(request: DecisionLogRequest) -> Dict[str, Any]:
    try:
        decision_id = decision_logger.log_decision(
            title=request.title,
            chosen=request.chosen,
            alternatives=request.alternatives,
            reasoning=request.reasoning,
            assumptions=request.assumptions,
            tags=request.tags
        )
        return {"status": "success", "decision_id": decision_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decisions")
def list_or_search_decisions(
    query: Optional[str] = Query(None, description="Semantic search query")
) -> List[Dict[str, Any]]:
    try:
        if query:
            return decision_logger.search_decisions(query)
        else:
            return decision_logger.list_decisions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisions/{decision_id}/outcome")
def update_decision_outcome(decision_id: int, payload: Dict[str, str] = Body(...)) -> Dict[str, str]:
    outcome = payload.get("outcome", "")
    if not outcome:
        raise HTTPException(status_code=400, detail="Missing body parameter 'outcome'")
    try:
        decision_logger.update_outcome(decision_id, outcome)
        return {"status": "success", "message": "Decision outcome updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 6. Dead Ideas Graveyard Endpoints
# ----------------------------------------------------
class BuryIdeaRequest(BaseModel):
    title: str
    description: str
    reason_abandoned: str
    resurrection_triggers: List[str]

@app.post("/graveyard")
def bury_idea(request: BuryIdeaRequest) -> Dict[str, Any]:
    try:
        idea_id = dead_ideas_tracker.bury_idea(
            title=request.title,
            description=request.description,
            reason_abandoned=request.reason_abandoned,
            resurrection_triggers=request.resurrection_triggers
        )
        return {"status": "success", "idea_id": idea_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graveyard")
def list_graveyard() -> List[Dict[str, Any]]:
    return dead_ideas_tracker.list_graveyard()

@app.post("/graveyard/{idea_id}/resurrect")
def resurrect_idea(idea_id: int) -> Dict[str, str]:
    try:
        dead_ideas_tracker.resurrect_idea(idea_id)
        return {"status": "success", "message": f"Idea {idea_id} resurrected."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graveyard/check")
def check_resurrections(
    current_context: str = Body(..., embed=True),
    threshold: float = Query(0.70)
) -> List[Dict[str, Any]]:
    try:
        context_vector = engine.embed(current_context)
        return dead_ideas_tracker.check_resurrections(context_vector, similarity_threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 7. Semantic Commits Endpoint
# ----------------------------------------------------
@app.get("/commits")
def get_semantic_commits(topic: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    return commit_logger.get_commit_history(topic)

@app.post("/commits")
def create_semantic_commit(
    topic: str = Body(...),
    drift_event: Dict[str, Any] = Body(...)
) -> Dict[str, str]:
    try:
        message = commit_logger.create_commit(topic, drift_event)
        return {"status": "success", "commit_message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 8. Learning Velocity Endpoint
# ----------------------------------------------------
@app.get("/velocity")
def get_learning_velocity(
    topic: str = Query(...),
    branch: str = Query("main")
) -> Dict[str, Any]:
    try:
        query_vector = engine.embed(topic)
        return velocity_engine.calculate_velocity(topic, query_vector, branch=branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 9. AI Mentor Endpoints
# ----------------------------------------------------
@app.post("/mentor/suggest")
def get_mentor_suggestion(
    current_context: str = Body(..., embed=True),
    branch: str = Query("main")
) -> Dict[str, str]:
    try:
        context_vector = engine.embed(current_context)
        return ai_mentor.get_mentor_suggestion(current_context, context_vector, branch=branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mentor/review")
def record_spaced_repetition_review(topic: str = Body(..., embed=True)) -> Dict[str, str]:
    try:
        forgetting_tracker.record_access(topic)
        return {"status": "success", "message": f"Review recorded for topic '{topic}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 10. Spaced Repetition (Forgetting Curve) Endpoints
# ----------------------------------------------------
@app.get("/forgotten")
def get_review_schedule() -> List[Dict[str, Any]]:
    return forgetting_tracker.get_review_schedule()

# ----------------------------------------------------
# 11. Memory Health Report Endpoint
# ----------------------------------------------------
@app.get("/health_report")
def get_memory_health(branch: str = Query("main")) -> Dict[str, Any]:
    try:
        return health_engine.calculate_health_report(branch=branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
