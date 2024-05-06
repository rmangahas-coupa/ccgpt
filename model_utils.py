import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from constants import MODEL_NAME, MAX_LENGTH
from config import logging

# Initialize tokenizer and model globally within the module
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_embeddings(texts, batch_size=32):
    logging.info("Generating embeddings...")
    num_texts = len(texts)
    embeddings = []
    progress_step = max(1, num_texts // 100)  # Update progress every 1% of the total texts

    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            outputs = model(**encoded_input)
            last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
            pooled_output = torch.mean(last_hidden_state, dim=1)  # (batch_size, hidden_size)
        embeddings.append(pooled_output.cpu().numpy())

        # Log progress
        if i % progress_step == 0:
            progress = min(100, i * 100 // num_texts)
            logging.info(f"Progress: {progress}%")

    embeddings = np.concatenate(embeddings, axis=0)
    logging.info("Embedding generation complete.")
    return embeddings
