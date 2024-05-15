"""This module contains utility functions for processing text."""

import html2text
import re


def clean_html(html_content: str) -> str:
    """Clean the given HTML content by removing unwanted elements."""
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_tables = False
    h.ignore_emphasis = False
    text = h.handle(html_content)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
