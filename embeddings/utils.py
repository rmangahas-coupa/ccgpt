"""This module contains utility functions for embeddings."""

import logging
import os
from typing import List, Tuple

import faiss
import numpy as np

from config import EMBEDDINGS_FILE
from embeddings.model import ModelWrapper
from utils.text_processing import clean_html

logger = logging.getLogger(__name__)


def generate_embeddings(texts: List[str],
                        model_wrapper: ModelWrapper) -> List[List[float]]:
    """Generate embeddings for the given list of texts."""
    cleaned_texts = [clean_html(text) for text in texts]
    return model_wrapper.generate_embeddings(cleaned_texts)


def save_embeddings(embeddings: List[List[float]], ids: List[str],
                    file_path: str) -> None:
    """Save the generated embeddings and their corresponding IDs to a file."""
    embeddings_dict = {'ids': ids, 'embeddings': embeddings}
    np.save(file_path, embeddings_dict)


def load_embeddings(file_path: str) -> Tuple[List[str], np.ndarray]:
    """Load the embeddings and their corresponding IDs from a file."""
    if not os.path.exists(file_path):
        logger.error(f"Embeddings file not found: {file_path}")
        return [], np.empty((0, 0))
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    return embeddings_dict['ids'], embeddings_dict['embeddings']


def create_faiss_index(embeddings: np.ndarray, file_path: str) -> None:
    """Create a FAISS index from the given embeddings and save it to a file."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, file_path)


def load_faiss_index(file_path: str) -> faiss.Index:
    """Load a FAISS index from a file."""
    return faiss.read_index(file_path)
