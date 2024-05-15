from typing import List
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, CACHE_DIR

class ModelWrapper:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = CACHE_DIR):
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True)
