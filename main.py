import argparse
from typing import List
import asyncio
from api.confluence import get_all_page_ids, get_page_content, get_all_spaces
from config import logging, EMBEDDINGS_FILE, MODEL_NAME
from embeddings.model import ModelWrapper
from embeddings.utils import generate_embeddings, save_embeddings, load_embeddings, create_faiss_index
from search.search import find_relevant_pages
from utils.gpt import ask_gpt
from utils.text_processing import clean_html
from utils.timestamp import need_update, update_timestamp

logger = logging.getLogger(__name__)

async def download_and_embed_confluence() -> None:
    all_texts = []
    all_page_ids = []
    space_keys = await get_all_spaces()

    for space_key in space_keys:
        page_ids = await get_all_page_ids([space_key])
        for page_id in page_ids:
            content = await get_page_content(page_id)
            cleaned_content = clean_html(content)
            all_texts.append(cleaned_content)
            all_page_ids.append(page_id)

    if all_texts:
        model_wrapper = ModelWrapper()
        embeddings = generate_embeddings(all_texts, model_wrapper)
        save_embeddings(embeddings, all_page_ids, EMBEDDINGS_FILE)
        logger.info("Saved embeddings for the entire Confluence.")
    else:
        logger.info("No content available to generate embeddings.")

async def refresh_embeddings(space_keys: List[str]) -> None:
    logger.info("Refreshing embeddings.")
    all_page_ids = await get_all_page_ids(space_keys)
    model_wrapper = ModelWrapper()

    texts_to_embed = []
    for page_id in all_page_ids:
        content = await get_page_content(page_id)
        if content:
            cleaned_content = clean_html(content)
            texts_to_embed.append(cleaned_content)

    if texts_to_embed:
        embeddings = generate_embeddings(texts_to_embed, model_wrapper)
        save_embeddings(embeddings, all_page_ids, EMBEDDINGS_FILE)
        update_timestamp()
    else:
        logger.info("No valid content to generate embeddings.")

async def main_async(question: str = None, force_refresh: bool = False, full_embed: bool = False) -> None:
    logger.info(f"Starting main function with model: {MODEL_NAME}")

    if full_embed:
        logger.info("Generating embeddings for the entire Confluence content...")
        await download_and_embed_confluence()
        return

    space_keys = ['CSAUS']
    if force_refresh or need_update():
        logger.info("Refreshing embeddings...")
        await refresh_embeddings(space_keys)

    if question:
        ids, embeddings = load_embeddings(EMBEDDINGS_FILE)
        if ids and embeddings.any():
            model_wrapper = ModelWrapper()
            top_pages = find_relevant_pages(question, embeddings, ids, model_wrapper)
            top_page_ids = [page[0] for page in top_pages]
            relevant_texts = [clean_html(await get_page_content(page_id)) for page_id in top_page_ids] 
            answer = ask_gpt(question, relevant_texts)

            logger.info(f"Top relevant page IDs: {top_page_ids}")
            print("Answer:", answer)
        else:
            logger.error("Embeddings are empty.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the knowledge base assistant.')
    parser.add_argument('question', nargs='?', default=None, help='A question to ask the model')
    parser.add_argument('--force-refresh', action='store_true', help='Force regeneration of embeddings')
    parser.add_argument('--full-embed', action='store_true', help='Generate embeddings for the entire Confluence')

    args = parser.parse_args()
    asyncio.run(main_async(args.question, args.force_refresh, args.full_embed))

