import hashlib
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple
import ollama
from core.config import EMBEDDING_MODEL, CACHE_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from core.models import RawThought, EmbeddedChunk

class EmbeddingEngine:
    def __init__(self):
        self.model = EMBEDDING_MODEL
        self.cache_path = CACHE_DB_PATH
        self._init_cache_db()

    def _init_cache_db(self):
        """Initializes the SQLite cache database for embeddings."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                text_content TEXT,
                embedding TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _get_hash(self, text: str) -> str:
        """Computes SHA-256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached_embeddings(self, hashes: List[str]) -> Dict[str, List[float]]:
        """Retrieves cached embeddings for a list of hashes."""
        if not hashes:
            return {}
        
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        # SQL IN clause placeholders
        placeholders = ",".join("?" for _ in hashes)
        cursor.execute(
            f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders})",
            hashes
        )
        
        results = {}
        for text_hash, embedding_str in cursor.fetchall():
            results[text_hash] = json.loads(embedding_str)
            
        conn.close()
        return results

    def _save_to_cache(self, items: List[Tuple[str, str, List[float]]]):
        """Saves text and embedding to cache database."""
        if not items:
            return
        
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO embedding_cache (text_hash, text_content, embedding)
            VALUES (?, ?, ?)
            """,
            [(h, txt, json.dumps(emb)) for h, txt, emb in items]
        )
        conn.commit()
        conn.close()

    def chunk_text(self, text: str, source: str) -> List[str]:
        """
        Splits text into overlapping chunks at word boundaries.
        Each chunk is up to CHUNK_SIZE words, with CHUNK_OVERLAP words overlapping.
        """
        words = text.split()
        if not words:
            return []
        
        if len(words) <= CHUNK_SIZE:
            return [text]
        
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk_words = words[i : i + CHUNK_SIZE]
            chunks.append(" ".join(chunk_words))
            # Stop if we reached or passed the end of words
            if i + CHUNK_SIZE >= len(words):
                break
                
        return chunks

    def embed(self, text: str) -> List[float]:
        """Gets embedding for a single text, checks cache first."""
        text_hash = self._get_hash(text)
        cached = self._get_cached_embeddings([text_hash])
        if text_hash in cached:
            return cached[text_hash]
        
        # If not cached, get from Ollama
        response = ollama.embed(model=self.model, input=text)
        embedding = self._extract_embedding_from_response(response)
        
        # Save to cache
        self._save_to_cache([(text_hash, text, embedding)])
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Gets embeddings for a batch of texts, checking cache first."""
        if not texts:
            return []
            
        hashes = [self._get_hash(txt) for txt in texts]
        cached_map = self._get_cached_embeddings(hashes)
        
        # Determine which texts need embedding
        to_embed_indices = []
        to_embed_texts = []
        for i, text_hash in enumerate(hashes):
            if text_hash not in cached_map:
                to_embed_indices.append(i)
                to_embed_texts.append(texts[i])
                
        # Call Ollama for uncached texts
        if to_embed_texts:
            response = ollama.embed(model=self.model, input=to_embed_texts)
            new_embeddings = self._extract_embeddings_from_response(response)
            
            # Cache the new ones
            to_cache = []
            for i, emb in zip(to_embed_indices, new_embeddings):
                text_hash = hashes[i]
                txt = texts[i]
                cached_map[text_hash] = emb
                to_cache.append((text_hash, txt, emb))
            self._save_to_cache(to_cache)
            
        return [cached_map[h] for h in hashes]

    def _extract_embedding_from_response(self, response: Any) -> List[float]:
        """Extracts a single embedding vector from the Ollama response object/dict."""
        # Ollama SDK handles returns differently across versions.
        # It could return a dict, or a custom class with an embeddings attribute or dict interface.
        if isinstance(response, dict):
            embs = response.get("embeddings")
            if embs:
                return embs[0]
            # Fallback to single embedding key
            emb = response.get("embedding")
            if emb:
                return emb
        
        # If response is an object (newer ollama SDK releases)
        if hasattr(response, "embeddings"):
            return response.embeddings[0]
        if hasattr(response, "embedding"):
            return response.embedding
            
        raise ValueError(f"Unexpected response format from Ollama: {response}")

    def _extract_embeddings_from_response(self, response: Any) -> List[List[float]]:
        """Extracts list of embedding vectors from Ollama response."""
        if isinstance(response, dict):
            embs = response.get("embeddings")
            if embs:
                return embs
            emb = response.get("embedding")
            if emb:
                return [emb]
                
        if hasattr(response, "embeddings"):
            return response.embeddings
        if hasattr(response, "embedding"):
            return [response.embedding]
            
        raise ValueError(f"Unexpected response format from Ollama: {response}")

    def embed_thought(self, thought: RawThought) -> List[EmbeddedChunk]:
        """Splits, embeds, and structures a RawThought into list of EmbeddedChunks."""
        chunks = self.chunk_text(thought.content, thought.source)
        if not chunks:
            return []
            
        embeddings = self.embed_batch(chunks)
        
        # Format partition collections naming: thoughts_{branch}_{YYYY}_{MM}
        branch = thought.metadata.get("branch", "main")
        year_month = thought.timestamp.strftime("%Y_%m")
        collection_name = f"thoughts_{branch}_{year_month}"
        
        embedded_chunks = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_hash = self._get_hash(f"{thought.timestamp.isoformat()}_{thought.source}_{idx}")
            chunk_id = f"chunk_{chunk_hash}"
            
            # Carry over metadata and set source-specific tags
            metadata = thought.metadata.copy()
            metadata.update({
                "source": thought.source,
                "timestamp": thought.timestamp.isoformat(),
                "chunk_index": str(idx),
                "total_chunks": str(len(chunks))
            })
            
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    source=thought.source,
                    timestamp=thought.timestamp,
                    collection_name=collection_name,
                    metadata=metadata
                )
            )
            
        return embedded_chunks
