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
    urls = [f"{BASE_URL}/content/?spaceKey={key}&start=0&limit=50" for key in space_keys]

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(fetch_with_retry, url): url for url in urls}
        for future in as_completed(future_to_url):
            response = future.result()
            json_data = response.json()
            logging.debug(f"Data received for URL: {future_to_url[future]} is {json_data}")  # Log the JSON data
            if json_data and 'results' in json_data and json_data['results']:
                page_ids = [page['id'] for page in json_data['results']]
                all_page_ids.extend(page_ids)
            else:
                logging.warning(f"No results found or failed request for URL: {future_to_url[future]}")
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

