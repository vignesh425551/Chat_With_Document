import hashlib
import re
from typing import List

from .caching import get_tokenizer


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def normalize_text_for_hash(text: str) -> str:
    # Remove null bytes and non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())
    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def keyword_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q.intersection(t)) / max(1, len(q))


def count_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text or ""))

