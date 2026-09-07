from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9@.\s-]", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9@.]+", normalize(text)) if tok not in ENGLISH_STOP_WORDS]


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    return 2 * precision * recall / (precision + recall)


def extract_question_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}\b", text)
    cleaned: list[str] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" ,.;:()[]{}\"'")
        if len(candidate) < 3:
            continue
        lowered = candidate.lower()
        if lowered in {"what", "which", "who", "when", "where", "why", "how", "were", "was", "is", "are", "the"}:
            continue
        if lowered not in seen:
            seen.add(lowered)
            cleaned.append(candidate)
    return cleaned
