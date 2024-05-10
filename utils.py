import os
import time
import openai
from datetime import datetime, timedelta
from constants import TIMESTAMP_FILE, EMBEDDINGS_FILE, GPT_MODEL, TOP_K
from api_utils import get_all_page_ids, get_page_content
from text_processing import clean_html
from model_utils import ModelWrapper  # Corrected import statement
from config import logging, openai_client
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    all_page_ids = []
    space_keys = get_all_spaces()  # Assume this function fetches all space keys

    for space_key in space_keys:
        page_ids = get_all_page_ids([space_key])
        for page_id in page_ids:
            content = get_page_content(page_id)
            cleaned_content = clean_html(content)
            all_texts.append(cleaned_content)
            all_page_ids.append(page_id)

    if all_texts:
        model_wrapper = ModelWrapper()
        embeddings = model_wrapper.generate_embeddings(all_texts)
        embeddings_dict = {'ids': all_page_ids, 'embeddings': embeddings}
        np.save(EMBEDDINGS_FILE, embeddings_dict)
        print("Saved embeddings for the entire Confluence.")
    else:
        print("No content available to generate embeddings.")

def refresh_embeddings(space_keys, model_name='all-MiniLM-L6-v2', batch_size=32):
    logging.info("Refreshing embeddings.")
    all_page_ids = get_all_page_ids(space_keys)
    model_wrapper = ModelWrapper(model_name)

    texts_to_embed = []
    for page_id in all_page_ids:
        content = get_page_content(page_id)
        if content:
            cleaned_content = clean_html(content)
            texts_to_embed.append(cleaned_content)

    if texts_to_embed:
        embeddings = model_wrapper.generate_embeddings(texts_to_embed)
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

def find_relevant_pages(question, embeddings_dict, model_name, top_k=TOP_K):
    logging.info(f"Finding relevant pages for question: '{question}'")
    model_wrapper = ModelWrapper(model_name)
    question_embedding = model_wrapper.generate_embeddings([question])[0]
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

def ask_gpt(question, relevant_texts, openai_model_name="gpt-3.5-turbo"):
    # Limit the number of documents and/or their length
    context = "\n\n".join([f"Document {idx + 1}: {text}" for idx, text in enumerate(relevant_texts[:7])])
    prompt = f"Relevant internal documentation:\n\n{context}\n\nBased on the internal documentation, answer the question: {question}"

    try:
        response = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system", "content": "You are a knowledgeable and experienced member of the Coupa Support team, dedicated to providing comprehensive and accurate assistance. Primarily base your answers on the information provided, using external sources to fill in the gaps.."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to execute OpenAI API call: {str(e)}")
        return "Error in processing the request. Please check the logs."

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for finding nearest neighbors
    index.add(embeddings)  # Adding the embeddings to the index
    faiss.write_index(index, 'path_to_faiss_index')  # Save the index to disk

def load_faiss_index(path_to_index):
    return faiss.read_index(path_to_index)

def search(query, model_wrapper, index, k=5):
    query_embedding = model_wrapper.generate_embeddings([query])[0]  # Get the embedding for the query
    distances, indices = index.search(np.array([query_embedding]), k)  # Retrieve the top k nearest neighbors
    return distances, indices

