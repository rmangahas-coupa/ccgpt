import os
import time
import openai
from datetime import datetime, timedelta
from constants import TIMESTAMP_FILE, EMBEDDINGS_FILE, GPT_MODEL, TOP_K
from api_utils import get_all_page_ids, get_page_content, get_all_spaces
from text_processing import clean_html
from model_utils import generate_embeddings
from config import logging, openai_client
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import logging, openai_client

def update_timestamp():
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(str(time.time()))

def need_update(force_update=False):
    if force_update:
        return True
    if not os.path.exists(TIMESTAMP_FILE):
        return True
    with open(TIMESTAMP_FILE, 'r') as f:
        last_update = f.read().strip()
    last_update_date = datetime.fromtimestamp(float(last_update))
    if datetime.now() - last_update_date > timedelta(days=7):
        return True
    return False

def download_and_embed_confluence():
    all_texts = []
    space_keys = get_all_spaces()  # Assume this function fetches all space keys

    for space_key in space_keys:
        page_ids = get_all_page_ids([space_key])
        for page_id in page_ids:
            content = get_page_content(page_id)
            cleaned_content = clean_html(content)
            all_texts.append(cleaned_content)
            print(f"Processed content from page ID {page_id} in space {space_key}")

    # Generate embeddings from all fetched and cleaned texts
    if all_texts:
        embeddings = generate_embeddings(all_texts, batch_size=32)
        np.save(EMBEDDINGS_FILE, {'embeddings': embeddings})
        print("Saved embeddings for the entire Confluence.")
    else:
        print("No content available to generate embeddings.")

def refresh_embeddings(space_keys, cache=None, max_workers=5, batch_size=32):
    logging.info("Refreshing embeddings.")
    all_page_ids = get_all_page_ids(space_keys)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_page_content, page_id): page_id for page_id in all_page_ids}
        texts_to_embed = []
        for future in as_completed(futures):
            content = future.result()
            if content:  # Only process content that exists
                cleaned_content = clean_html(content)
                texts_to_embed.append(cleaned_content)
            else:
                logging.debug(f"Skipping embedding generation for missing content at page ID {futures[future]}")

    if texts_to_embed:  # Check if there are texts to process
        embeddings = generate_embeddings(texts_to_embed, batch_size=batch_size)
        embeddings_dict = {'ids': all_page_ids, 'embeddings': embeddings}
        np.save(EMBEDDINGS_FILE, embeddings_dict)
        update_timestamp()
    else:
        logging.info("No valid content to generate embeddings.")

def load_embeddings(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Embeddings file not found: {file_path}")
        return None
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    return embeddings_dict

def find_relevant_pages(question, embeddings_dict, top_k=TOP_K):
    logging.info(f"Finding relevant pages for question: '{question}'")
    question_embedding = generate_embeddings([question])[0]
    dim = embeddings_dict['embeddings'].shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_dict['embeddings'])
    scores, indices = index.search(np.expand_dims(question_embedding, axis=0), top_k)
    top_indices = [embeddings_dict['ids'][i] for i in indices[0].tolist()]
    top_pages = list(zip(top_indices, scores[0].tolist()))
    return top_pages

def get_and_clean_contents(page_ids, max_workers=5):
    texts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_page_content, page_id): page_id for page_id in page_ids}
        for future in as_completed(futures):
            page_id = futures[future]
            try:
                content = future.result()
                cleaned_content = clean_html(content)
                texts.append(cleaned_content)
            except Exception as exc:
                logging.error(f"Failed to clean content for page ID {page_id}: {exc}")
    return texts

import openai

def ask_gpt(question, texts):
    context = "\n\n".join([f"Document {idx + 1}: {text}" for idx, text in enumerate(texts)])
    prompt = f"Relevant internal documentation:\n\n{context}\n\nBased on the internal documentation, answer the question: {question}"

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable and experienced member of Coupa's technical support team, dedicated to providing comprehensive and accurate assistance. Your primary goal is to leverage the provided documentation to address user queries effectively. However, you should also utilize your own broader knowledge and reasoning abilities to fill in any gaps or provide additional context where necessary.\n\nWhen responding, please follow these guidelines:\n\nBase your answers primarily on the relevant information from the provided documentation, quoting or referencing specific sections when applicable.\n\nIf the documentation alone is insufficient to fully address the query, supplement it with your own knowledge and insights, clearly distinguishing this supplementary information from the documentation.\n\nMaintain an objective and factual tone, avoiding speculation or unfounded assertions.\n\nIf you cannot find sufficient information to provide a satisfactory answer, communicate this transparently, and request additional clarification or details from the user.\n\nFor each response, include a confidence level (e.g., high, medium, low) to indicate your degree of certainty in the accuracy and completeness of the information provided.\n\nEncourage follow-up questions from the user if there is any ambiguity or if additional clarification is needed.\n\nRemain professional, patient, and courteous throughout the interaction, acknowledging the potential complexity of technical issues and the user's need for clear and understandable explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content
