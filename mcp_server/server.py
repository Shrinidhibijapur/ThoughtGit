import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from core.models import RawThought
from core.embedder import EmbeddingEngine
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

# Create the FastMCP instance
mcp = FastMCP("ThoughtGit")

# Instantiate core engines
engine = EmbeddingEngine()
store = ThoughtStore()
diff_engine = SemanticDiffEngine(store)

@mcp.tool()
def recall_context(topic: str, n_results: int = 5) -> str:
    """Retrieve what the user has written about a topic, ordered by relevance."""
    try:
        query_vector = engine.embed(topic)
        results = store.query_across_time(query_vector, n_results=n_results)
        if not results:
            return f"No memories found about '{topic}'."
        
        formatted = [f"Memories found for '{topic}':\n"]
        for idx, r in enumerate(results):
            formatted.append(
                f"{idx+1}. [{r['collection']}] (Similarity: {r['similarity']:.3f})\n"
                f"Source: {r['metadata'].get('source', 'unknown')} | Date: {r['metadata'].get('timestamp', 'unknown')}\n"
                f"Content: {r['text']}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error recalling context: {str(e)}"

@mcp.tool()
def diff_thinking(topic: str, min_cluster_size: int = 2) -> str:
    """Show how the understanding of a topic has evolved over time in the timeline."""
    try:
        query_vector = engine.embed(topic)
        analysis = diff_engine.analyze_drift(topic, query_vector, min_cluster_size=min_cluster_size)
        
        if not analysis["snapshots"]:
            return f"No snapshots found for topic '{topic}'."
            
        lines = [f"Conceptual evolution timeline for '{topic}':\n"]
        lines.append("--- Monthly Snapshots ---")
        for s in analysis["snapshots"]:
            lines.append(
                f"Period: {s['time_label']} | Chunks: {s['chunks_count']} | "
                f"Avg text length: {s['avg_text_length']:.1f}\n"
                f"  Sample: {s['sample_texts'][0][:100]}...\n"
            )
            
        lines.append("--- Concept Shift / Drift Events ---")
        if not analysis["drift_events"]:
            lines.append("No concept shifts detected (stable understanding).")
        for e in analysis["drift_events"]:
            lines.append(
                f"Transition: {e['from_period']} -> {e['to_period']}\n"
                f"  Distance: {e['distance']:.3f} | Shift Category: {e['drift_type'].upper()}\n"
                f"  Summary: {e['summary']}\n"
            )
            
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing thinking drift: {str(e)}"

@mcp.tool()
def ingest_thought(content: str, topic_hint: str = "general") -> str:
    """Store a new thought from the conversation directly into memory."""
    try:
        # Save thought under MCP source with topic_hint inside metadata
        thought = RawThought(
            content=content,
            source="mcp",
            metadata={"topic_hint": topic_hint}
        )
        chunks = engine.embed_thought(thought)
        if chunks:
            store.store_chunks_batch(chunks)
            return f"Success! Ingested thought ({len(chunks)} chunks) into collection '{chunks[0].collection_name}'."
        return "No chunks generated from thought content."
    except Exception as e:
        return f"Error ingesting thought: {str(e)}"

@mcp.tool()
def list_topics() -> str:
    """List all chronological bucket collections currently stored in memory."""
    try:
        colls = store.list_collections(branch="main")
        if not colls:
            return "No memory collections found in databases."
        return "Stored collections:\n" + "\n".join(f" - {c}" for c in colls)
    except Exception as e:
        return f"Error listing topics: {str(e)}"

@mcp.tool()
def check_duplicates(text: str) -> str:
    """Check if a thought matches previously written/solved ideas."""
    try:
        from core.config import DUPLICATE_THRESHOLD
        query_vector = engine.embed(text)
        results = store.query_across_time(query_vector, n_results=1)
        if results and results[0]["distance"] < DUPLICATE_THRESHOLD:
            return (
                f"DUPLICATE FOUND (Distance: {results[0]['distance']:.3f} < threshold {DUPLICATE_THRESHOLD}):\n"
                f"Location: {results[0]['collection']}\n"
                f"Content: {results[0]['text']}"
            )
        return "No duplicate concepts found (the idea is unique)."
    except Exception as e:
        return f"Error checking duplicates: {str(e)}"

if __name__ == "__main__":
    # FastMCP uses standard IO transport by default if executed as main module
    mcp.run()
