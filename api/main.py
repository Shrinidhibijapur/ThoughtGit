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

# ----------------------------------------------------
# 12. Interactive Project Map & Codebase Flow Endpoints
# ----------------------------------------------------
import os
import re

@app.get("/project_map")
def get_project_map(branch: str = Query("main")) -> Dict[str, Any]:
    """Computes pairwise similarity links between all topics in ChromaDB."""
    try:
        collections = store.list_collections(branch=branch)
        all_chunks = []
        for name in collections:
            coll = store.client.get_collection(name=name)
            count = coll.count()
            if count > 0:
                results = coll.query(
                    query_embeddings=[[0.0]*768],
                    n_results=count,
                    include=["documents", "metadatas", "embeddings"]
                )
                if results and results["ids"]:
                    for i in range(len(results["ids"][0])):
                        all_chunks.append({
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "embedding": results["embeddings"][0][i]
                        })
        
        # Group by topic_hint
        topic_groups = {}
        for chunk in all_chunks:
            meta = chunk["metadata"]
            topic = meta.get("topic_hint", "")
            if topic:
                topic_groups.setdefault(topic, []).append(chunk)
                
        nodes = []
        centroids = {}
        for topic, chunks in topic_groups.items():
            embeddings = [c["embedding"] for c in chunks if c["embedding"] is not None]
            if not embeddings:
                continue
            centroid = diff_engine._compute_centroid(embeddings)
            centroids[topic] = centroid
            
            latest_ts = ""
            for c in chunks:
                ts = c["metadata"].get("timestamp", "")
                if ts and (not latest_ts or ts > latest_ts):
                    latest_ts = ts
            
            if latest_ts and not latest_ts.endswith("Z") and "+" not in latest_ts:
                latest_ts += "Z"
                    
            nodes.append({
                "id": topic,
                "label": topic,
                "size": len(chunks),
                "latest_timestamp": latest_ts
            })
            
        links = []
        topics_list = list(centroids.keys())
        for i in range(len(topics_list)):
            for j in range(i + 1, len(topics_list)):
                ta = topics_list[i]
                tb = topics_list[j]
                dist = diff_engine._cosine_distance(centroids[ta], centroids[tb])
                similarity = 1.0 - dist
                if similarity >= 0.40:
                    links.append({
                        "source": ta,
                        "target": tb,
                        "value": float(similarity)
                    })
                    
        # Sort nodes and links for deterministic stable serialization
        nodes.sort(key=lambda x: x["id"])
        links.sort(key=lambda x: (x["source"], x["target"]))
        
        return {"nodes": nodes, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/codebase_flow")
def get_codebase_flow(workspace_dir: str = Query(...)) -> Dict[str, Any]:
    """Generates a Mermaid dependency flowchart mapping folders and cross-folder module imports."""
    try:
        workspace_dir = os.path.abspath(workspace_dir)
        if not os.path.exists(workspace_dir):
            raise HTTPException(status_code=400, detail="Workspace directory does not exist")
            
        file_paths = []
        exclude_dirs = {"venv", ".venv", "env", "node_modules", "build", "dist", ".git", "obsidian-plugin"}
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith((".py", ".ts", ".js")):
                    file_paths.append(os.path.join(root, file))
                    
        # Helper to get parent folder name
        def get_parent_folder(fp):
            rel = os.path.relpath(fp, workspace_dir).replace("\\", "/")
            parts = rel.split("/")
            if len(parts) > 1:
                return parts[0]
            return "root" # root folder level files

        links = []
        
        # Build node maps
        rel_paths = {}
        for fp in file_paths:
            rel = os.path.relpath(fp, workspace_dir).replace("\\", "/")
            rel_paths[fp] = rel
            
        # Parse imports
        for fp in file_paths:
            source_folder = get_parent_folder(fp)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue
                
            imported_names = []
            if fp.endswith(".py"):
                matches = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", content, re.MULTILINE)
                for m in matches:
                    parts = m.split(".")
                    imported_names.append(parts)
            elif fp.endswith((".ts", ".js")):
                matches = re.findall(r"from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\)", content)
                for m in matches:
                    path_str = m[0] if m[0] else m[1]
                    if path_str.startswith("."):
                        imported_names.append(path_str)
                        
            for imp in imported_names:
                resolved_rel = None
                if fp.endswith(".py") and isinstance(imp, list):
                    test_paths = [
                        os.path.join(workspace_dir, *imp) + ".py",
                        os.path.join(os.path.dirname(fp), *imp) + ".py"
                    ]
                    for tp in test_paths:
                        if os.path.exists(tp):
                            resolved_rel = os.path.relpath(tp, workspace_dir).replace("\\", "/")
                            break
                elif fp.endswith((".ts", ".js")) and isinstance(imp, str):
                    base_dir = os.path.dirname(fp)
                    test_paths = [
                        os.path.abspath(os.path.join(base_dir, imp)),
                        os.path.abspath(os.path.join(base_dir, imp) + ".ts"),
                        os.path.abspath(os.path.join(base_dir, imp) + ".js")
                    ]
                    for tp in test_paths:
                        if os.path.exists(tp):
                            resolved_rel = os.path.relpath(tp, workspace_dir).replace("\\", "/")
                            break
                        elif os.path.isdir(tp):
                            for idx in ["index.ts", "index.js"]:
                                if os.path.exists(os.path.join(tp, idx)):
                                    resolved_rel = os.path.relpath(os.path.join(tp, idx), workspace_dir).replace("\\", "/")
                                    break
                                    
                if resolved_rel:
                    resolved_abs = os.path.join(workspace_dir, resolved_rel)
                    target_folder = get_parent_folder(resolved_abs)
                    if source_folder != target_folder:
                        links.append({"source": source_folder, "target": target_folder})
                        
        # Extract unique folder nodes from scanned files
        folders = set(get_parent_folder(fp) for fp in file_paths)
        nodes = [{"id": f, "label": f + "/"} for f in folders]
        
        # Generate Mermaid String
        mermaid_lines = ["graph TD"]
        mermaid_lines.append("    classDef folderNode fill:#0b132b,stroke:#00f2fe,stroke-width:2.5px,color:#ffffff,rx:10px,font-weight:bold;")
        
        added_links = set()
        for link in links:
            pair = (link["source"], link["target"])
            if pair not in added_links:
                added_links.add(pair)
                src_id = f"folder_{pair[0]}"
                tgt_id = f"folder_{pair[1]}"
                mermaid_lines.append(f'    {src_id}["📂 {pair[0]}/"] --> {tgt_id}["📂 {pair[1]}/"]')
                
        for folder in folders:
            f_id = f"folder_{folder}"
            mermaid_lines.append(f'    class {f_id} folderNode;')
            
        mermaid_str = "\n".join(mermaid_lines)
        return {"mermaid_code": mermaid_str, "nodes": nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
