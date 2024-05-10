import argparse
import numpy as np
from api_utils import get_all_page_ids, get_page_content
from constants import EMBEDDINGS_FILE
from text_processing import clean_html
from config import logging
from utils import need_update, refresh_embeddings, load_embeddings, find_relevant_pages, get_and_clean_contents, ask_gpt, download_and_embed_confluence, create_faiss_index, load_faiss_index, search
from model_utils import ModelWrapper

def main(question=None, force_refresh=False, full_embed=False, model_name='all-MiniLM-L6-v2'):
    logging.info("Starting main function with model: {}".format(model_name))

    if full_embed:
        logging.info("Generating embeddings for the entire Confluence content...")
        download_and_embed_confluence(model_name)
        return

    space_keys = ['CSAUS','CCS']
    if force_refresh or need_update(force_refresh):
        logging.info("Refreshing embeddings...")
        refresh_embeddings(space_keys, model_name)

    if question:
        embeddings_dict = load_embeddings(EMBEDDINGS_FILE)  # Use load_embeddings function
        if not embeddings_dict or 'embeddings' not in embeddings_dict or 'ids' not in embeddings_dict:
            logging.error("Embeddings are incorrectly formatted or empty.")
            return

        create_faiss_index(embeddings_dict['embeddings'])  # Create and save the index
        index = load_faiss_index('path_to_faiss_index')

        top_pages = find_relevant_pages(question, embeddings_dict, model_name)
        top_page_ids = [page[0] for page in top_pages]
        relevant_texts = get_and_clean_contents(top_page_ids)

        model_wrapper = ModelWrapper(model_name)
        answer = ask_gpt(question, relevant_texts)

        logging.info("Top relevant page IDs: {}".format(top_page_ids))
        print("Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the knowledge base assistant.')
    parser.add_argument('question', nargs='?', default=None, help='A question to ask the model')
    parser.add_argument('--force-refresh', action='store_true', help='Force regeneration of embeddings')
    parser.add_argument('--full-embed', action='store_true', help='Generate embeddings for the entire Confluence')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Specify the model to use')

    args = parser.parse_args()
    main(args.question, args.force_refresh, args.full_embed, args.model)

