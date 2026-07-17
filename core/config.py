import os

# Embedding Configurations
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 400       # words per chunk
CHUNK_OVERLAP = 50     # word overlap between chunks

# Algorithmic Thresholds
DRIFT_THRESHOLD = 0.25      # Cosine distance above which we identify drift
DUPLICATE_THRESHOLD = 0.15  # Cosine distance below which notes are duplicate
SIMILARITY_THRESHOLD = 0.70 # Cosine similarity threshold for relevance queries

# Storage Paths
BASE_DIR = os.path.expanduser("~/.thoughtgit")
DB_DIR = os.path.join(BASE_DIR, "chroma")
CACHE_DB_PATH = os.path.join(BASE_DIR, "embed_cache.db")
METADATA_DB_PATH = os.path.join(BASE_DIR, "metadata.db")

# Ensure base directories exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
