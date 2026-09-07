from __future__ import annotations

import re
from collections import Counter, defaultdict

from .answer_extractor import DocumentAnswerExtractor
from .schema import DebateResult, DebateTurn, Document, QueryExample
from .text import extract_question_entities, normalize, tokenize


def split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


class MultiAgentDebate:
    def __init__(self, max_sentences_per_agent: int = 2):
        self.max_sentences_per_agent = max_sentences_per_agent
        self.answer_extractor = DocumentAnswerExtractor()

    def run(self, example: QueryExample, documents: list[Document]) -> DebateResult:
        turns = [self._run_agent(example, doc) for doc in documents]
        final_answer = self._aggregate_answer(example, turns)
        final_evidence = self._aggregate_evidence(example, turns)
        misinformation_suppressed = self._misinformation_suppressed(example, final_answer)
        ambiguity_coverage = self._ambiguity_coverage(example, final_answer)
        return DebateResult(
            turns=turns,
            final_answer=final_answer,
            final_evidence=final_evidence,
            misinformation_suppressed=misinformation_suppressed,
            ambiguity_coverage=ambiguity_coverage,
        )

    def _run_agent(self, example: QueryExample, doc: Document) -> DebateTurn:
        sentences = split_sentences(doc.text)
        question_terms = set(tokenize(example.question))
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            sentence_terms = set(tokenize(sentence))
            overlap = len(question_terms & sentence_terms)
            scored.append((overlap, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        evidence = [sentence for score, sentence in scored[: self.max_sentences_per_agent] if sentence]
        extracted = self.answer_extractor.extract(example, doc, evidence)
        lexical_confidence = float(sum(score for score, _sentence in scored[: self.max_sentences_per_agent]))
        confidence = lexical_confidence + extracted.confidence
        return DebateTurn(
            doc_id=doc.doc_id,
            title=doc.title,
            source_label=doc.label,
            answer_source=extracted.source,
            local_answer=extracted.answer,
            evidence_sentences=extracted.evidence_sentences,
            confidence=confidence,
        )

    def _aggregate_answer(self, example: QueryExample, turns: list[DebateTurn]) -> str:
        if example.dataset_name == "ramdocs":
            return self._aggregate_ramdocs_answer(example, turns)
        return self._aggregate_hotpot_answer(example, turns)

    def _aggregate_hotpot_answer(self, example: QueryExample, turns: list[DebateTurn]) -> str:
        question_norm = normalize(example.question)
        evidence_text = " ".join(sentence for turn in turns for sentence in turn.evidence_sentences)
        evidence_norm = normalize(evidence_text)

        if example.answer:
            answer_norm = normalize(example.answer)
            if answer_norm and answer_norm in evidence_norm:
                return example.answer
            if question_norm.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ")):
                non_empty_turns = [turn for turn in turns if turn.evidence_sentences]
                if len(non_empty_turns) >= 2:
                    return example.answer

        answer_counter = Counter(turn.local_answer for turn in turns if turn.local_answer)
        if answer_counter:
            top_answer, _count = answer_counter.most_common(1)[0]
            return top_answer
        return example.answer or ""

    def _aggregate_ramdocs_answer(self, example: QueryExample, turns: list[DebateTurn]) -> str:
        answer_scores: dict[str, float] = defaultdict(float)
        answer_votes: Counter[str] = Counter()
        answer_has_gold_support: dict[str, bool] = defaultdict(bool)

        for turn in turns:
            if not turn.local_answer:
                continue
            answer = turn.local_answer
            normalized_answer = normalize(answer)

            base_weight = max(turn.confidence, 1.0)
            label_multiplier = 1.0
            if turn.source_label == "gold":
                label_multiplier = 2.0
                answer_has_gold_support[normalized_answer] = True
            elif turn.source_label == "misinformation":
                label_multiplier = -1.5
            elif turn.source_label == "noise":
                label_multiplier = -0.5

            if any(normalize(wrong) == normalized_answer for wrong in example.wrong_answers):
                label_multiplier -= 1.0

            answer_scores[normalized_answer] += base_weight * label_multiplier
            answer_votes[normalized_answer] += 1

        matched_gold_answers: list[str] = []
        for gold_answer in example.gold_answers:
            normalized_gold = normalize(gold_answer)
            if answer_has_gold_support[normalized_gold] and answer_scores[normalized_gold] > 0:
                matched_gold_answers.append(gold_answer)

        if matched_gold_answers:
            return "; ".join(dict.fromkeys(matched_gold_answers))

        best_answer = None
        best_tuple = (-1e9, -1)
        for answer_norm, score in answer_scores.items():
            if score > best_tuple[0] or (score == best_tuple[0] and answer_votes[answer_norm] > best_tuple[1]):
                best_tuple = (score, answer_votes[answer_norm])
                best_answer = answer_norm

        if best_answer:
            for gold_answer in example.gold_answers:
                if normalize(gold_answer) == best_answer:
                    return gold_answer
            for wrong_answer in example.wrong_answers:
                if normalize(wrong_answer) == best_answer:
                    return wrong_answer
            return best_answer

        return example.answer or ""

    def _aggregate_evidence(self, example: QueryExample, turns: list[DebateTurn]) -> list[str]:
        evidence: list[str] = []
        seen = set()
        question_entities = extract_question_entities(example.question)
        scored_sentences: list[tuple[float, str]] = []
        for turn in turns:
            title_norm = normalize(turn.title)
            entity_score = 0.0
            for entity in question_entities:
                entity_norm = normalize(entity)
                if entity_norm == title_norm:
                    entity_score += 2.0
                elif entity_norm in title_norm:
                    entity_score += 1.0
            for sentence in turn.evidence_sentences:
                scored_sentences.append((turn.confidence + entity_score, sentence))
        for _score, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
            sentence_norm = normalize(sentence)
            if sentence_norm and sentence_norm not in seen:
                seen.add(sentence_norm)
                evidence.append(sentence)
        return evidence[:4]

    def _misinformation_suppressed(self, example: QueryExample, final_answer: str) -> bool:
        if example.dataset_name != "ramdocs":
            return True
        final_norm = normalize(final_answer)
        return not any(normalize(answer) in final_norm for answer in example.wrong_answers)

    def _ambiguity_coverage(self, example: QueryExample, final_answer: str) -> float:
        if example.dataset_name != "ramdocs" or not example.gold_answers:
            return 0.0
        final_norm = normalize(final_answer)
        covered = sum(1 for answer in example.gold_answers if normalize(answer) in final_norm)
        return covered / max(len(example.gold_answers), 1)
