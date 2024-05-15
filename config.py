import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUTH = (os.getenv('CONFLUENCE_USER'), os.getenv('CONFLUENCE_API_TOKEN'))
BASE_URL = os.getenv('BASE_URL', 'https://coupadev.atlassian.net/wiki/rest/api')
CACHE_DIR = os.getenv('CACHE_DIR', './cache')
EMBEDDINGS_FILE = os.getenv('EMBEDDINGS_FILE', 'embeddings.npy')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-3.5-turbo')
JITTER_MULTIPLIER_RANGE = (float(os.getenv('JITTER_MULTIPLIER_MIN', 0.7)), float(os.getenv('JITTER_MULTIPLIER_MAX', 1.3)))
LAST_RETRY_DELAY_MILLIS = int(os.getenv('LAST_RETRY_DELAY_MILLIS', 5000))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 4))
MAX_RETRY_DELAY_MILLIS = int(os.getenv('MAX_RETRY_DELAY_MILLIS', 30000))
MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
TIMESTAMP_FILE = os.getenv('TIMESTAMP_FILE', 'last_update.txt')
TOP_K = int(os.getenv('TOP_K', 7))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

