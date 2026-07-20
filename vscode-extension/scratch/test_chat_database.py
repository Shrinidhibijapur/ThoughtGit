import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.thought_store import ThoughtStore
from core.embedder import EmbeddingEngine
from core.models import RawThought

def main():
    print("==================================================")
    # Print ASCII header
    print("      Ingesting Synthetic Chat Database Notes")
    print("==================================================")
    
    store = ThoughtStore()
    embedder = EmbeddingEngine()
    
    # 3 Notes tracing the decision change from SQL (Postgres) to NoSQL (ScyllaDB)
    notes = [
        # Month 1: Initial SQL Setup
        (
            "We are designing our real-time chat application backend. We will start with standard PostgreSQL database storage to index user messages. A relational database gives us strict ACID guarantees and lets us easily join users and groups tables.",
            datetime(2026, 5, 10)
        ),
        # Month 2: Scale reinforcement
        (
            "Our chat application has scaled to 1,000 active users. PostgreSQL is working well for messages. We set up indexes on group_id and created_at to keep query latency low when loading channels history.",
            datetime(2026, 6, 12)
        ),
        # Month 3: Drift / Migration to ScyllaDB NoSQL
        (
            "PostgreSQL message table size is growing too fast. We are shifting to ScyllaDB (distributed wide-column NoSQL database) for message storage because it provides masterless replication, horizontal scaling, and sub-millisecond writes for telemetry and active logs.",
            datetime(2026, 7, 15)
        )
    ]
    
    topic = "chat-database"
    print(f"Ingesting {len(notes)} notes about topic '{topic}'...")
    
    for idx, (content, date) in enumerate(notes):
        thought = RawThought(
            content=content,
            source="manual_test",
            timestamp=date,
            metadata={"topic_hint": topic}
        )
        
        # Split & embed chunks
        embedded_chunks = embedder.embed_thought(thought)
        
        # Store in ChromaDB
        store.store_chunks_batch(embedded_chunks)
        print(f"  [{date.strftime('%Y-%m-%d')}] Ingested note {idx+1}")
        
    print("\n[OK] Ingestion Complete! Ready to test on dashboard.")
    print("Next step: Start your server and search for 'chat-database' in the dashboard!")

if __name__ == "__main__":
    main()
