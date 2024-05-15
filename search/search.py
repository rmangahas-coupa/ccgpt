import logging
from typing import List, Tuple

import faiss
import numpy as np

from config import TOP_K
from embeddings.model import ModelWrapper

logger = logging.getLogger(__name__)

def find_relevant_pages(question: str, embeddings: np.ndarray, ids: List[str], model_wrapper: ModelWrapper, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    question_embedding = model_wrapper.generate_embeddings([question])[0]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    scores, indices = index.search(np.expand_dims(question_embedding, axis=0), top_k)
    top_indices = [ids[i] for i in indices[0].tolist()]
    top_pages = list(zip(top_indices, scores[0].tolist()))
    return top_pages
