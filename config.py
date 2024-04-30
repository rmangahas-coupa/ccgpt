import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = 'https://coupadev.atlassian.net/wiki/rest/api'
MODEL_NAME = "roberta-base"
GPT_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_FILE = 'embeddings.npy'
TIMESTAMP_FILE = 'last_update.txt'
TOP_K = 20

# Load environment variables
load_dotenv()
AUTH = (os.getenv('CONFLUENCE_USER'), os.getenv('CONFLUENCE_API_TOKEN'))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

