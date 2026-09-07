from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ._llm_client import make_llm_call
from .schema import Document


_ANSWER_PROMPT = """\
Answer the question using only the documents provided. Be as concise as possible — give the answer in a few words, not a full sentence.

Question: {question}

{docs}

Answer:"""


_AGENT_PROMPT = """\
You are one agent in a multi-agent debate. Read the single document below and answer the question.

Question: {question}

Document title: {title}
Document text: {doc_text}

Respond ONLY with valid JSON (no markdown):
{{"answer": "<your answer in 1-5 words>", "confidence": "<low|medium|high>", "evidence": "<the key sentence from the document that supports your answer>"}}"""


_ARBITRATOR_PROMPT = """\
You are the final arbitrator in a multi-agent debate. {n} agents each read one document and proposed an answer to the question below.

Question: {question}

Agent proposals:
{proposals}

Pick the single most likely correct answer. Prefer answers that:
- Are supported by multiple agents
- Come from agents with "high" confidence
- Are specific (names, places, dates) rather than vague

Respond with only the final answer in as few words as possible. No explanation.
Answer:"""


@dataclass
class GenerationResult:
    answer: str
    method: str  # "llm_direct" | "llm_debate" | "fallback"
    agent_proposals: list[dict] | None = None


class LLMGenerator:
    """Direct answer generation from retrieved documents."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 64, max_doc_chars: int = 800):
        self.model = model
        self.max_tokens = max_tokens
        self.max_doc_chars = max_doc_chars
        self._disabled = False

    def generate(self, question: str, docs: list[Document]) -> GenerationResult:
        if self._disabled:
            return GenerationResult(answer="", method="fallback")
        docs_text = "\n\n".join(
            f"[Doc {i + 1}] Title: {doc.title}\n{doc.text[:self.max_doc_chars]}"
            for i, doc in enumerate(docs)
        )
        prompt = _ANSWER_PROMPT.format(question=question, docs=docs_text)
        try:
            answer = make_llm_call(prompt, self.model, max_tokens=self.max_tokens)
            return GenerationResult(answer=answer, method="llm_direct")
        except Exception as exc:
            if _is_auth_error(exc):
                print(f"[LLMGenerator] Auth error — disabling for this run.")
                self._disabled = True
            else:
                print(f"[LLMGenerator] API error: {exc}")
            return GenerationResult(answer="", method="fallback")


class LLMDebate:
    """MADAM-RAG style multi-agent debate: each agent reads one doc, arbitrator synthesises."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 128, max_doc_chars: int = 600):
        self.model = model
        self.max_tokens = max_tokens
        self.max_doc_chars = max_doc_chars
        self._disabled = False

    def debate(self, question: str, docs: list[Document]) -> GenerationResult:
        if self._disabled:
            return GenerationResult(answer="", method="fallback", agent_proposals=[])
        proposals: list[dict] = []
        for doc in docs:
            proposal = self._agent_turn(question, doc)
            if proposal:
                proposals.append(proposal)

        if not proposals:
            return GenerationResult(answer="", method="fallback", agent_proposals=[])

        final_answer = self._arbitrate(question, proposals)
        return GenerationResult(answer=final_answer, method="llm_debate", agent_proposals=proposals)

    def _agent_turn(self, question: str, doc: Document) -> dict | None:
        prompt = _AGENT_PROMPT.format(
            question=question,
            title=doc.title,
            doc_text=doc.text[:self.max_doc_chars],
        )
        try:
            raw = make_llm_call(prompt, self.model, max_tokens=self.max_tokens)
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(raw)
            return {"doc_id": doc.doc_id, "title": doc.title, **data}
        except Exception as exc:
            if _is_auth_error(exc):
                print(f"[LLMDebate] Auth error — disabling for this run.")
                self._disabled = True
            else:
                print(f"[LLMDebate] Agent error on doc '{doc.title}': {exc}")
            return None

    def _arbitrate(self, question: str, proposals: list[dict]) -> str:
        proposals_text = "\n".join(
            f"Agent {i + 1} (doc: '{p.get('title', '?')}', confidence: {p.get('confidence', '?')}): "
            f"answer='{p.get('answer', '')}', evidence='{p.get('evidence', '')}'"
            for i, p in enumerate(proposals)
        )
        prompt = _ARBITRATOR_PROMPT.format(n=len(proposals), question=question, proposals=proposals_text)
        try:
            return make_llm_call(prompt, self.model, max_tokens=64)
        except Exception as exc:
            print(f"[LLMDebate] Arbitrator error: {exc}")
            from collections import Counter
            votes = Counter(p.get("answer", "") for p in proposals if p.get("answer"))
            return votes.most_common(1)[0][0] if votes else ""


def _is_auth_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "401" in s or "authentication" in s or "unauthorized" in s
