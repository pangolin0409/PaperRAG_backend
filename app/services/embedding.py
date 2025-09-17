from sentence_transformers import SentenceTransformer
from typing import List

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts."""
    return model.encode(texts).tolist()
