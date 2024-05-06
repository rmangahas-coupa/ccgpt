import argparse
from api_utils import get_all_page_ids, get_page_content
from text_processing import clean_html
from constants import EMBEDDINGS_FILE
from config import logging
from utils import need_update, refresh_embeddings, load_embeddings, find_relevant_pages, get_and_clean_contents, ask_gpt, download_and_embed_confluence

def main(question=None, force_refresh=False, full_embed=False):
    if full_embed:
        print("Generating embeddings for the entire Confluence content...")
        download_and_embed_confluence()
        return
    space_keys = ['CSAUS']
    if force_refresh or need_update(force_refresh):
        refresh_embeddings(space_keys)
    if question:
        embeddings_dict = load_embeddings(EMBEDDINGS_FILE)
        if embeddings_dict is None:
            logging.error("Failed to load embeddings.")
            return
        top_pages = find_relevant_pages(question, embeddings_dict)
        top_page_ids = [page[0] for page in top_pages]
        relevant_texts = get_and_clean_contents(top_page_ids)
        answer = ask_gpt(question, relevant_texts)
        print("Top relevant page IDs:", top_page_ids)
        print("Answer:", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the knowledge base assistant.')
    parser.add_argument('question', nargs='?', default=None, help='A question to ask the model')
    parser.add_argument('--force-refresh', action='store_true', help='Force regeneration of embeddings')
    parser.add_argument('--full-embed', action='store_true', help='Generate embeddings for the entire Confluence')
    args = parser.parse_args()
    main(args.question, args.force_refresh, args.full_embed)
