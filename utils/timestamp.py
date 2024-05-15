import logging
import os
import time
from datetime import datetime, timedelta

from config import TIMESTAMP_FILE

logger = logging.getLogger(__name__)

def update_timestamp() -> None:
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(str(time.time()))

def need_update() -> bool:
    if not os.path.exists(TIMESTAMP_FILE):
        return True
    with open(TIMESTAMP_FILE, 'r') as f:
        last_update = f.read().strip()
    last_update_date = datetime.fromtimestamp(float(last_update))
    if datetime.now() - last_update_date > timedelta(days=7):
        return True
    return False
