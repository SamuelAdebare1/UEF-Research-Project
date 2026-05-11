"""Shared utilities: PDF text extraction and sentence splitting."""
import re
from pdfminer.high_level import extract_text


def load_text(pdf_path: str) -> str:
    return extract_text(pdf_path)


def split_sentences(text: str) -> list[str]:
    # Split on period/exclamation/question followed by whitespace or end
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip().replace('\n', ' ') for s in raw if len(s.strip()) > 20]
    return sentences
