import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from core.models import RawThought
from core.embedder import EmbeddingEngine
from core.thought_store import ThoughtStore
from core.semantic_diff import SemanticDiffEngine

# Extended Feature Imports
from core.time_machine import TimeMachine
from core.ai_mentor import AIMentor
from core.forgetting_curve import ForgettingCurveTracker
from core.memory_health import MemoryHealthEngine

# Create the FastMCP instance
mcp = FastMCP("ThoughtGit")

# Instantiate core and extended engines
engine = EmbeddingEngine()
store = ThoughtStore()
diff_engine = SemanticDiffEngine(store)

time_machine = TimeMachine(store)
ai_mentor = AIMentor(store)
forgetting_tracker = ForgettingCurveTracker()
health_engine = MemoryHealthEngine(store, forgetting_tracker)

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

@mcp.tool()
def get_timeline_summary() -> str:
    """Show an overview of recent intellectual activity and memory health metrics."""
    try:
        report = health_engine.calculate_health_report()
        res = [
            f"=== ThoughtGit Memory Health Summary ===",
            f"Memory Health Score: {report['health_score']}/100",
            f"Interpretation: {report['interpretation']}\n",
            f"--- Breakdown Metrics ---",
            f"1. Ingestion Activity (last 30 days): {report['metrics']['activity']['score']}/35.0 pts ({report['metrics']['activity']['recent_chunks_count']} new entries)",
            f"2. Topic Diversity: {report['metrics']['diversity']['score']}/35.0 pts ({report['metrics']['diversity']['unique_topics_count']} topics tracked)",
            f"3. Spacing Review Ratio: {report['metrics']['spacing']['score']}/30.0 pts ({int(report['metrics']['spacing']['retained_topics_ratio']*100)}% review retention rate)"
        ]
        return "\n".join(res)
    except Exception as e:
        return f"Error calculating timeline summary: {str(e)}"

@mcp.tool()
def recall_as_of_date(topic: str, as_of: str, compare_to: str = "") -> str:
    """
    Retrieve what you knew about a topic at a specific date (Format: YYYY-MM-DD),
    optionally comparing it side-by-side with your understanding at another date.
    """
    try:
        # Parse dates
        try:
            date_a = datetime.strptime(as_of.strip(), "%Y-%m-%d")
        except ValueError:
            return "Error: Invalid 'as_of' date format. Please use YYYY-MM-DD."

        query_vector = engine.embed(topic)

        if compare_to.strip():
            try:
                date_b = datetime.strptime(compare_to.strip(), "%Y-%m-%d")
            except ValueError:
                return "Error: Invalid 'compare_to' date format. Please use YYYY-MM-DD."

            comp = time_machine.compare_understanding(topic, query_vector, date_a, date_b)
            lines = [
                f"=== Side-by-side Understanding comparison for '{topic}' ===",
                f"Date A: {comp['date_a']} ({comp['snapshots_count_a']} notes)",
                f"Date B: {comp['date_b']} ({comp['snapshots_count_b']} notes)\n",
                f"--- Understanding at Date A ---",
                "\n".join(f"- {txt}" for txt in comp['learnings_at_date_a']) if comp['learnings_at_date_a'] else "- No notes recorded.",
                f"\n--- Understanding at Date B ---",
                "\n".join(f"- {txt}" for txt in comp['learnings_at_date_b']) if comp['learnings_at_date_b'] else "- No notes recorded.",
                f"\n--- New Learnings acquired between Date A and Date B ---"
            ]
            if comp['new_learnings_since_a']:
                for item in comp['new_learnings_since_a']:
                    lines.append(f"- [{item['timestamp'][:10]}] {item['text']}")
            else:
                lines.append("- No new conceptual learnings identified in this interval.")
            return "\n".join(lines)
        else:
            snaps = time_machine.recall_as_of(topic, query_vector, date_a)
            if not snaps:
                return f"No memories found about '{topic}' up to date {as_of}."
            lines = [f"Memories found for '{topic}' as of date {as_of}:\n"]
            for idx, r in enumerate(snaps):
                lines.append(
                    f"{idx+1}. [{r['collection']}] (Similarity: {r['similarity']:.3f})\n"
                    f"Content: {r['text']}\n"
                )
            return "\n".join(lines)
    except Exception as e:
        return f"Error running time machine: {str(e)}"

@mcp.tool()
def get_mentor_suggestion(context: str) -> str:
    """Get proactive mentor advice, linking your current task context to relevant past insights."""
    try:
        context_vector = engine.embed(context)
        advice = ai_mentor.get_mentor_suggestion(context, context_vector)
        res = [
            f"=== AI Mentor Proactive Suggestion ===",
            f"Insight: {advice['insight']}",
            f"Reasoning: {advice['reason']}",
            f"Past Reference: {advice['past_reference']}",
            f"Suggested Action: {advice['action']}"
        ]
        return "\n".join(res)
    except Exception as e:
        return f"Error generating mentor advice: {str(e)}"

if __name__ == "__main__":
    mcp.run()
