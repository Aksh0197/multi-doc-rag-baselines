from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ._llm_client import make_llm_call
from .schema import Document


_FAITHFULNESS_PROMPT = """\
You are evaluating a RAG system's answer for faithfulness to retrieved documents.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Retrieved Documents:
{docs}

Evaluate the predicted answer on two dimensions:

1. faithfulness_score (0-10): How well is the predicted answer grounded in the retrieved documents?
   0 = completely hallucinated (no support in docs)
   5 = partially supported (some claims in docs, some not)
   10 = fully grounded (every claim traceable to a retrieved doc)

2. hallucinated_claims: List specific phrases or facts in the predicted answer that are NOT found in any retrieved document. Empty list if fully faithful.

3. failure_mode: Classify why the answer is wrong (if it is):
   "correct"            - predicted answer matches gold answer
   "retrieval_failure"  - needed facts were absent from retrieved docs
   "generation_failure" - facts were present in retrieved docs but answer is still wrong (hallucination)
   "partial"            - some relevant facts retrieved but answer incomplete or slightly off

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "faithfulness_score": <integer 0-10>,
  "hallucinated_claims": [<string>, ...],
  "failure_mode": "correct" | "retrieval_failure" | "generation_failure" | "partial"
}}"""


@dataclass
class FaithfulnessResult:
    faithfulness_score: float = 0.0
    hallucinated_claims: list[str] = field(default_factory=list)
    failure_mode: str = "unknown"
    raw_response: str = ""


class FaithfulnessChecker:
    """RAGChecker-style faithfulness evaluator."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_doc_chars: int = 600):
        self.model = model
        self.max_doc_chars = max_doc_chars
        self._disabled = False

    def check(
        self,
        question: str,
        predicted_answer: str,
        retrieved_docs: list[Document],
        gold_answer: str | None = None,
    ) -> FaithfulnessResult:
        if self._disabled:
            return FaithfulnessResult(failure_mode="unknown")
        if not predicted_answer.strip():
            return FaithfulnessResult(faithfulness_score=0.0, failure_mode="retrieval_failure")

        docs_text = "\n\n".join(
            f"[Doc {i+1}] Title: {doc.title}\n{doc.text[:self.max_doc_chars]}"
            for i, doc in enumerate(retrieved_docs)
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            gold_answer=gold_answer or "unknown",
            predicted_answer=predicted_answer,
            docs=docs_text,
        )
        try:
            raw = make_llm_call(prompt, self.model, max_tokens=256)
            return self._parse(raw)
        except Exception as exc:
            if _is_auth_error(exc):
                print(f"[FaithfulnessChecker] Auth error — disabling for this run.")
                self._disabled = True
            else:
                print(f"[FaithfulnessChecker] API error: {exc}")
            return FaithfulnessResult(failure_mode="unknown", raw_response=str(exc))

    def _parse(self, raw: str) -> FaithfulnessResult:
        try:
            cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(cleaned)
            return FaithfulnessResult(
                faithfulness_score=float(data.get("faithfulness_score", 0)) / 10.0,
                hallucinated_claims=data.get("hallucinated_claims", []),
                failure_mode=data.get("failure_mode", "unknown"),
                raw_response=raw,
            )
        except Exception:
            return FaithfulnessResult(failure_mode="unknown", raw_response=raw)


def _is_auth_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "401" in s or "authentication" in s or "unauthorized" in s


def deterministic_failure_mode(answer_em: float, multi_doc_hit: float) -> str:
    """Classify failure mode without LLM calls using retrieval + answer signals."""
    if answer_em == 1.0:
        return "correct"
    if multi_doc_hit == 1.0:
        return "generation_failure"
    return "retrieval_failure"
