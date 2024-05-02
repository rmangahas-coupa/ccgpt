# model_utils.py
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from constants import MODEL_NAME, MAX_LENGTH
from config import logging
from sentence_transformers import SentenceTransformer

class ModelWrapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

