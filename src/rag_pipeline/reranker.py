from __future__ import annotations

import os
import re

from ._llm_client import make_llm_call
from .schema import Document


_BRIDGE_PROMPT = """\
Question: {question}

Document title: {title}
Document text: {text}

This document is one of two documents needed to answer the question above.
Identify the single key entity or concept from this document that would help find the second document needed to answer the question.
Reply with ONLY the entity or short phrase (under 10 words), nothing else."""

_RELEVANCE_PROMPT = """\
You are a relevance judge for a multi-hop question answering system.

Question: {question}

Document title: {title}
Document text: {text}

Rate how useful this document is for answering the question above.
Reply with a single integer from 0 to 10, where:
  0  = completely irrelevant
  5  = partially relevant (mentions related concepts but not the answer)
  10 = highly relevant (contains key facts needed to answer)

Reply with ONLY the integer, nothing else."""


class LLMReranker:
    """RankRAG-style pointwise LLM reranker."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 8):
        self.model = model
        self.max_tokens = max_tokens
        self._disabled = False

    def extract_bridge_entity(self, question: str, doc: Document) -> str:
        if self._disabled:
            return doc.title
        prompt = _BRIDGE_PROMPT.format(question=question, title=doc.title, text=doc.text[:600])
        try:
            return make_llm_call(prompt, self.model, max_tokens=32)
        except Exception as exc:
            if _is_auth_error(exc):
                print(f"[LLMReranker] Auth error — disabling for this run.")
                self._disabled = True
            return doc.title

    def rerank(self, question: str, candidates: list[Document], top_k: int) -> list[str]:
        if self._disabled:
            return [doc.doc_id for doc in candidates[:top_k]]
        scored: list[tuple[float, str]] = []
        for doc in candidates:
            scored.append((self._score(question, doc), doc.doc_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc_id for _, doc_id in scored[:top_k]]

    def _score(self, question: str, doc: Document) -> float:
        if self._disabled:
            return 0.0
        prompt = _RELEVANCE_PROMPT.format(question=question, title=doc.title, text=doc.text[:800])
        try:
            raw = make_llm_call(prompt, self.model, max_tokens=self.max_tokens)
            match = re.search(r"\d+", raw)
            return float(match.group()) if match else 0.0
        except Exception as exc:
            if _is_auth_error(exc):
                print(f"[LLMReranker] Auth error — disabling for this run.")
                self._disabled = True
            return 0.0


def _is_auth_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "401" in s or "authentication" in s or "unauthorized" in s or "invalid" in s
