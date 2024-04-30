import html2text
import re

def clean_html(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_tables = True
    h.ignore_emphasis = True
    text = h.handle(html_content)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

