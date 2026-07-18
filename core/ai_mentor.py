import ollama
from typing import List, Dict, Any, Optional
from core.thought_store import ThoughtStore

class AIMentor:
    def __init__(self, store: ThoughtStore, model: str = "mistral"):
        self.store = store
        self.model = model

    def _check_ollama_status(self) -> bool:
        """Helper to verify if local Ollama is active."""
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=1)
            return True
        except Exception:
            return False

    def get_mentor_suggestion(
        self,
        current_context: str,
        query_embedding: List[float],
        branch: str = "main"
    ) -> Dict[str, str]:
        """
        Retrieves relevant past memories, builds a prompt, asks the local
        Ollama LLM to generate insights, and returns a structured card.
        """
        # Fetch up to 3 similar past thoughts
        past_memories = self.store.query_across_time(
            query_embedding=query_embedding,
            n_results=3,
            branch=branch
        )
        
        # Fallback if no past memories exist
        if not past_memories:
            return {
                "insight": "Start building your memory base.",
                "reason": "You don't have enough thoughts indexed in this branch.",
                "past_reference": "N/A",
                "action": "Write more code comments or Markdown notes, and they will automatically index."
            }

        # Build context from memories
        memories_text = "\n".join(
            f"- [{m['collection']}]: {m['text']}" for m in past_memories
        )

        prompt = (
            f"You are an AI Technical Mentor helping a developer. "
            f"Analyze their current writing context and their past thoughts, "
            f"and generate a short, high-value proactive suggestion.\n\n"
            f"Developer's Current Context:\n\"{current_context}\"\n\n"
            f"Relevant Past Thinking (From Memory Database):\n{memories_text}\n\n"
            f"Provide your advice formatted exactly as follows:\n"
            f"INSIGHT: <one sentence highlight of a connection or reflection>\n"
            f"REASON: <one sentence explaining why this is relevant to their current focus>\n"
            f"PAST REFERENCE: <mention which topics or collections from their history this links to>\n"
            f"ACTION: <suggested immediate next step or coding experiment>"
        )

        # Call local Ollama if online, otherwise return a fallback mock
        if self._check_ollama_status():
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.message.content
                return self._parse_mentor_response(content)
            except Exception as e:
                # Fallback on Ollama exception
                return self._get_fallback_suggestion(current_context, past_memories, f"Ollama Error: {e}")
        else:
            return self._get_fallback_suggestion(current_context, past_memories, "Ollama Offline Fallback")

    def _parse_mentor_response(self, text: str) -> Dict[str, str]:
        """Parses the structured text from LLM response into dictionary."""
        lines = text.split("\n")
        result = {
            "insight": "",
            "reason": "",
            "past_reference": "",
            "action": ""
        }
        
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("insight:"):
                result["insight"] = line[8:].strip()
            elif line_lower.startswith("reason:"):
                result["reason"] = line[7:].strip()
            elif line_lower.startswith("past reference:"):
                result["past_reference"] = line[15:].strip()
            elif line_lower.startswith("action:"):
                result["action"] = line[7:].strip()
                
        # Handle unformatted responses gracefully
        if not result["insight"]:
            result["insight"] = "Connection found between current task and past notes."
            result["reason"] = "Your current focus shares semantic concepts with previous writing."
            result["past_reference"] = "Past database search results"
            result["action"] = "Review your historical vector database queries."
            
        return result

    def _get_fallback_suggestion(
        self,
        current_context: str,
        past_memories: List[Dict[str, Any]],
        reason_flag: str
    ) -> Dict[str, str]:
        """Generates a clean deterministic fallback suggestion for offline mode."""
        best_match = past_memories[0]["text"]
        collection = past_memories[0]["collection"]
        
        return {
            "insight": f"Identified similarity to records in '{collection}'.",
            "reason": f"({reason_flag}) Your current focus is closely related to your past thoughts on: \"{best_match[:60]}...\"",
            "past_reference": f"Collection '{collection}' in your local memory.",
            "action": f"Verify whether the logic inside notes in '{collection}' already contains solutions for your current work."
        }
