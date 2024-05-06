import requests
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import AUTH, BASE_URL

# Constants for retry
MAX_RETRIES = 4
LAST_RETRY_DELAY_MILLIS = 5000
MAX_RETRY_DELAY_MILLIS = 30000
JITTER_MULTIPLIER_RANGE = (0.7, 1.3)

def fetch_with_retry(url):
    retry_count = 0
    retry_delay_millis = LAST_RETRY_DELAY_MILLIS
    
    while retry_count <= MAX_RETRIES:
        response = requests.get(url, auth=AUTH)
        if response.ok:
            return response  # Return the full response object
        else:
            # handle retries similarly as before
            if 'Retry-After' in response.headers:
                retry_delay_millis = int(response.headers['Retry-After']) * 1000
            elif response.status_code == 429:
                retry_delay_millis = min(2 * retry_delay_millis, MAX_RETRY_DELAY_MILLIS)
            
            if retry_delay_millis > 0 and retry_count < MAX_RETRIES:
                sleep_time = retry_delay_millis / 1000 * random.uniform(*JITTER_MULTIPLIER_RANGE)
                time.sleep(sleep_time)
                retry_count += 1
            else:
                response.raise_for_status()

def get_all_spaces():
    url = f"{BASE_URL}/space"
    spaces = []
    start = 0
    limit = 50  # Adjust based on what the API supports

    while True:
        response = requests.get(f"{url}?start={start}&limit={limit}", auth=AUTH)
        data = response.json()
        spaces.extend(data['results'])

        # Check if there's more data to fetch
        if 'next' in data['_links']:
            start += limit
        else:
            break

    return [space['key'] for space in spaces]  # Return a list of space keys

def get_all_page_ids(space_keys):
    all_page_ids = []
    limit = 50  # Number of page IDs to retrieve per request
    for space_key in space_keys:
        start = 0
        while True:
            url = f"{BASE_URL}/content/?spaceKey={space_key}&start={start}&limit={limit}"
            response = fetch_with_retry(url)
            if response.status_code == 200:
                json_data = response.json()
                page_results = json_data.get('results', [])
                page_ids = [page['id'] for page in page_results]
                all_page_ids.extend(page_ids)
                if len(page_results) < limit:
                    break  # No more pages to fetch
                start += limit
            else:
                logging.error(f"Failed to fetch page IDs for space key {space_key}")
                break
    return all_page_ids

def get_page_content(page_id):
    url = f"{BASE_URL}/content/{page_id}?expand=body.storage"
    try:
        response = fetch_with_retry(url)
        if response.status_code == 200:
            json_content = response.json()
            return json_content.get('body', {}).get('storage', {}).get('value', "")
        else:
            return None
    except Exception as e:
        logging.error(f"Failed to retrieve content for page ID {page_id}: {str(e)}")
        return None

