import sys
import os
from datetime import datetime
import shutil
import numpy as np

# Make sure core package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import RawThought
from core.embedder import EmbeddingEngine
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

def check_ollama_status() -> bool:
    """Checks if the local Ollama service is running and accessible."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False

def patch_ollama_with_mock():
    """Patches the ollama client to return simulated semantic vectors for offline verification."""
    print(">>> WARNING: Local Ollama service not detected.")
    print(">>> Activating offline Mock Embedding Engine fallback to verify the pipeline...")
    
    import ollama
    
    def mock_embed(model, input):
        # Handle input being a single string or list of strings
        texts = [input] if isinstance(input, str) else input
        embeddings = []
        
        for text in texts:
            # Deterministic vector based on content to simulate drift
            text_lower = text.lower()
            if "hybrid search" in text_lower or "query expansion" in text_lower or "rerankers" in text_lower:
                # March: Advanced concepts shift
                v = [0.707, 0.707] + [0.0] * 766
            elif "database collections" in text_lower or "similarity searches" in text_lower or "hallucination" in text_lower:
                # February: Stable / slight refine
                v = [0.999, 0.001] + [0.0] * 766
            else:
                # January: Basic RAG
                v = [1.0] + [0.0] * 767
            
            # Ensure unit norm
            v = list(np.array(v) / np.linalg.norm(v))
            embeddings.append(v)
            
        return {"embeddings": embeddings}
        
    ollama.embed = mock_embed

def run_verification():
    print("===========================================")
    print("ThoughtGit Core Pipeline Verification Utility")
    print("===========================================\n")
    
    # Check if we need to run in mock mode
    if not check_ollama_status():
        patch_ollama_with_mock()
    else:
        print(">>> Local Ollama service detected. Running with actual embeddings...")
    
    # We will use temporary directories to avoid polluting production databases
    test_db_dir = "./scratch_thoughtgit_db"
    test_cache_path = "./scratch_thoughtgit_cache.db"
    
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir, ignore_errors=True)
    if os.path.exists(test_cache_path):
        os.remove(test_cache_path)
        
    print("Initializing engine components...")
    
    # Patch config variables before importing store and embedder
    import core.config as config
    config.DB_DIR = test_db_dir
    config.CACHE_DB_PATH = test_cache_path
    
    # Re-initialize engine components with patched paths
    engine = EmbeddingEngine()
    store = ThoughtStore()
    diff_engine = SemanticDiffEngine(store)
    
    print("Creating synthetic thoughts across three months...")
    
    # Define synthetic thoughts
    thoughts_data = [
        # January 2026: Basic RAG notes
        (
            "Basic RAG searches document embeddings and appends to prompt context. Standard retrieval utilizes semantic vector lookup using cosine similarity.",
            datetime(2026, 1, 15),
            "vscode"
        ),
        (
            "Vector embeddings represent text in high-dimensional space for semantic search retrieval in basic RAG systems.",
            datetime(2026, 1, 28),
            "obsidian"
        ),
        
        # February 2026: Reinforcing the same understanding
        (
            "Retrieving relevant context from a vector database helps reduce hallucination in large language model responses.",
            datetime(2026, 2, 10),
            "vscode"
        ),
        (
            "We use database collections to index text chunks and run similarity searches based on query embeddings.",
            datetime(2026, 2, 22),
            "cli"
        ),
        
        # March 2026: Conceptual shift to advanced RAG (reranking, hybrid search)
        (
            "Production RAG requires hybrid search combining BM25 keyword matching and dense vector search, followed by cross-encoder rerankers to filter low-quality context.",
            datetime(2026, 3, 5),
            "vscode"
        ),
        (
            "Advanced retrieval systems use query expansion, sub-queries, and hierarchical summarization of documents to improve retrieval precision and context efficiency.",
            datetime(2026, 3, 20),
            "obsidian"
        )
    ]
    
    # Ingest thoughts
    print("\nIngesting thoughts...")
    for text, ts, src in thoughts_data:
        thought = RawThought(content=text, source=src, timestamp=ts)
        print(f"[{ts.strftime('%Y-%m-%d')}] Ingesting thought from {src} ({len(text)} chars)...")
        embedded_chunks = engine.embed_thought(thought)
        store.store_chunks_batch(embedded_chunks)
        
    print("\nListing time-bucket collections in store:")
    colls = store.list_collections(branch="main")
    for coll in colls:
        print(f" - {coll}")
        
    print("\nRunning Semantic Recall query for topic 'RAG'...")
    # Get embedding of query
    query_vector = engine.embed("RAG and vector search systems")
    results = store.query_across_time(query_vector, n_results=3)
    for idx, r in enumerate(results):
        print(f" {idx+1}. [{r['collection']}] (Similarity: {r['similarity']:.3f}) {r['text'][:70]}...")
        
    print("\nComputing Semantic Diff for topic 'RAG' across time buckets...")
    analysis = diff_engine.analyze_drift(
        topic="RAG",
        query_embedding=query_vector,
        min_cluster_size=2
    )
    
    print("\n--- Snapshots ---")
    for s in analysis["snapshots"]:
        print(f"Period: {s['time_label']} | Chunks: {s['chunks_count']} | Avg text length: {s['avg_text_length']:.1f}")
        print(f"  Sample: {s['sample_texts'][0][:80]}...")
        
    print("\n--- Drift Events ---")
    if not analysis["drift_events"]:
        print("  No drift events detected (stable understanding).")
    for e in analysis["drift_events"]:
        print(f"Transition: {e['from_period']} -> {e['to_period']}")
        print(f"  Cosine Distance: {e['distance']:.3f}")
        print(f"  Drift Type: {e['drift_type'].upper()}")
        print(f"  Summary: {e['summary']}")
        print()
        
    # Cleanup scratch DB files after execution
    shutil.rmtree(test_db_dir, ignore_errors=True)
    if os.path.exists(test_cache_path):
        os.remove(test_cache_path)
    print("Verification completed successfully and scratch data cleaned up.")

if __name__ == "__main__":
    run_verification()
