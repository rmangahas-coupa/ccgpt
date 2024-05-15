"""This module contains functions for interacting with the Confluence API."""

import logging
import random
import asyncio
from typing import List

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, AsyncRetrying

from config import (
    AUTH, BASE_URL, MAX_RETRIES, LAST_RETRY_DELAY_MILLIS,
    MAX_RETRY_DELAY_MILLIS, JITTER_MULTIPLIER_RANGE
)

logger = logging.getLogger(__name__)


async def fetch_with_retry(url: str) -> dict:
    """Fetch data from the given URL with retries and rate limiting."""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=LAST_RETRY_DELAY_MILLIS / 1000,
            max=MAX_RETRY_DELAY_MILLIS / 1000
        )
    ):
        with attempt:
            async with aiohttp.ClientSession() as session:
                logger.info(f"Fetching URL: {url}")
                async with session.get(url,
                                       auth=aiohttp.BasicAuth(*AUTH),
                                       timeout=60) as response:
                    logger.info(
                        f"Received response for URL: {url} with status: {response.status}"
                    )
                    if response.status == 429:
                        retry_delay_millis = int(
                            response.headers.get('Retry-After', 0)) * 1000
                        jitter_multiplier = random.uniform(
                            *JITTER_MULTIPLIER_RANGE)
                        sleep_time = retry_delay_millis / 1000 * jitter_multiplier
                        logger.warning(
                            f"Rate limited. Retrying after {sleep_time:.2f} seconds."
                        )
                        await asyncio.sleep(sleep_time)
                        raise Exception("Rate limited. Retrying...")
                    response.raise_for_status()
                    return await response.json()


async def get_all_spaces() -> List[str]:
    """Retrieve all spaces from the Confluence API."""
    logger.info("Fetching all spaces.")
    url = f"{BASE_URL}/space"
    spaces = []
    start = 0
    limit = 50

    while True:
        logger.info(f"Fetching spaces with start={start} and limit={limit}")
        data = await fetch_with_retry(f"{url}?start={start}&limit={limit}")
        spaces.extend(data['results'])

        if 'next' in data['_links']:
            start += limit
        else:
            break

    logger.info(f"Found spaces: {spaces}")
    return [space['key'] for space in spaces]


async def get_all_page_ids(space_keys: List[str]) -> List[str]:
    """Retrieve all page IDs from the given space keys."""
    logger.info("Fetching all page IDs.")
    all_page_ids = []
    limit = 50
    for space_key in space_keys:
        start = 0
        while True:
            url = f"{BASE_URL}/content/?spaceKey={space_key}&start={start}&limit={limit}"
            logger.info(
                f"Fetching page IDs for space key: {space_key} with start={start} and limit={limit}"
            )
            json_data = await fetch_with_retry(url)
            page_results = json_data.get('results', [])
            page_ids = [page['id'] for page in page_results]
            all_page_ids.extend(page_ids)
            if len(page_results) < limit:
                break
            start += limit
    logger.info(f"Found page IDs: {all_page_ids}")
    return all_page_ids


async def get_page_content(page_id: str) -> str:
    """Retrieve the content of a page with the given page ID."""
    logger.info(f"Fetching content for page ID: {page_id}")
    url = f"{BASE_URL}/content/{page_id}?expand=body.storage"
    try:
        json_content = await fetch_with_retry(url)
        content = json_content.get('body', {}).get('storage', {}).get('value', "")
        logger.info(f"Content fetched for page ID: {page_id}")
        return content
    except Exception as e:
        logger.error(f"Failed to retrieve content for page ID {page_id}: {str(e)}")
        return ""
