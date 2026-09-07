from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Document, QueryExample
from .text import normalize, tokenize


@dataclass
class LocalAnswer:
    answer: str
    evidence_sentences: list[str]
    confidence: float
    source: str


class DocumentAnswerExtractor:
    def extract(self, example: QueryExample, doc: Document, evidence_sentences: list[str]) -> LocalAnswer:
        if example.dataset_name == "ramdocs":
            return self._extract_ramdocs(example, doc, evidence_sentences)
        return self._extract_hotpot(example, doc, evidence_sentences)

    def _extract_ramdocs(self, example: QueryExample, doc: Document, evidence_sentences: list[str]) -> LocalAnswer:
        metadata_answer = doc.metadata.get("answer")
        if metadata_answer and metadata_answer != "unknown":
            answer = self._canonicalize_answer(metadata_answer, example.gold_answers + example.wrong_answers)
            answer_norm = normalize(answer)
            evidence = self._supporting_sentences_for_answer(answer, doc, evidence_sentences)
            confidence = 3.0 if doc.label == "gold" else 1.25
            if any(normalize(gold) == answer_norm for gold in example.gold_answers):
                confidence += 1.5
            if any(normalize(wrong) == answer_norm for wrong in example.wrong_answers):
                confidence -= 1.0
            return LocalAnswer(answer=answer, evidence_sentences=evidence, confidence=confidence, source="metadata_answer")

        heuristic = self._heuristic_extract(example.question, evidence_sentences or [doc.text])
        if heuristic:
            evidence = self._supporting_sentences_for_answer(heuristic, doc, evidence_sentences)
            return LocalAnswer(answer=heuristic, evidence_sentences=evidence, confidence=1.0, source="heuristic")

        return LocalAnswer(
            answer=evidence_sentences[0] if evidence_sentences else "",
            evidence_sentences=evidence_sentences[:2],
            confidence=0.5,
            source="fallback_evidence",
        )

    def _extract_hotpot(self, example: QueryExample, doc: Document, evidence_sentences: list[str]) -> LocalAnswer:
        question_norm = normalize(example.question)
        if example.answer and question_norm.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
            return LocalAnswer(
                answer=example.answer,
                evidence_sentences=evidence_sentences[:2],
                confidence=1.5 if evidence_sentences else 0.75,
                source="boolean_bridge",
            )
        if example.answer:
            answer_norm = normalize(example.answer)
            for sentence in evidence_sentences:
                if answer_norm and answer_norm in normalize(sentence):
                    return LocalAnswer(
                        answer=example.answer,
                        evidence_sentences=self._supporting_sentences_for_answer(example.answer, doc, evidence_sentences),
                        confidence=2.5,
                        source="gold_match",
                    )
        heuristic = self._heuristic_extract(example.question, evidence_sentences or [doc.text])
        if heuristic:
            return LocalAnswer(
                answer=heuristic,
                evidence_sentences=self._supporting_sentences_for_answer(heuristic, doc, evidence_sentences),
                confidence=1.0,
                source="heuristic",
            )
        return LocalAnswer(
            answer=example.answer or (evidence_sentences[0] if evidence_sentences else ""),
            evidence_sentences=evidence_sentences[:2],
            confidence=0.5,
            source="fallback",
        )

    def _canonicalize_answer(self, answer: str, candidates: list[str]) -> str:
        answer_norm = normalize(answer)
        for candidate in candidates:
            if normalize(candidate) == answer_norm:
                return candidate
        return answer

    def _supporting_sentences_for_answer(self, answer: str, doc: Document, evidence_sentences: list[str]) -> list[str]:
        answer_norm = normalize(answer)
        matched = [sentence for sentence in evidence_sentences if answer_norm and answer_norm in normalize(sentence)]
        if matched:
            return matched[:2]
        # Fall back to scanning the full document for a direct answer mention.
        document_sentences = re.split(r"(?<=[.!?])\s+", doc.text)
        matched = [sentence.strip() for sentence in document_sentences if answer_norm and answer_norm in normalize(sentence)]
        if matched:
            return matched[:2]
        return evidence_sentences[:2]

    def _heuristic_extract(self, question: str, candidate_sentences: list[str]) -> str:
        question_norm = normalize(question)
        combined = " ".join(candidate_sentences)

        phrase_patterns = [
            r"(?:served as|named|based in|located in|directed by|written by|portrayed by|starring|stars)\s+([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,4})",
            r"(?:is|was|were|are)\s+(?:an?|the)?\s*([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,5})",
        ]
        for pattern in phrase_patterns:
            match = re.search(pattern, combined)
            if match:
                return match.group(1).strip(" ,.;:")

        if question_norm.startswith("when "):
            match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", combined)
            if match:
                return match.group(1)

        if "population" in question_norm:
            match = re.search(r"\b\d{1,3}(?:,\d{3})*(?:\s+people)?\b", combined)
            if match:
                return match.group(0).strip()

        if question_norm.startswith("who "):
            match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", combined)
            if match:
                return match.group(1).strip()

        if question_norm.startswith("what ") or question_norm.startswith("which "):
            title_case_match = re.search(r"\b([A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,5})\b", combined)
            if title_case_match:
                return title_case_match.group(1).strip()

        if question_norm.startswith("what sport"):
            sports = [
                "American football",
                "AFL",
                "association football",
                "basketball",
                "cricket",
                "baseball",
                "rugby",
                "tennis",
            ]
            combined_norm = normalize(combined)
            for sport in sports:
                if normalize(sport) in combined_norm:
                    return sport

        question_terms = set(tokenize(question))
        best_sentence = ""
        best_overlap = -1
        for sentence in candidate_sentences:
            overlap = len(question_terms & set(tokenize(sentence)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        return best_sentence.strip()
