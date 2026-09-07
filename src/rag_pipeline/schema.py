from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    label: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryExample:
    query_id: str
    question: str
    documents: list[Document]
    gold_doc_ids: list[str]
    answer: str | None = None
    answer_aliases: list[str] = field(default_factory=list)
    supporting_facts: list[str] = field(default_factory=list)
    wrong_answers: list[str] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    dataset_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    initial_ranked_ids: list[str]
    reranked_ids: list[str]
    selected_ids: list[str]


@dataclass
class DebateTurn:
    doc_id: str
    title: str
    source_label: str | None
    answer_source: str
    local_answer: str
    evidence_sentences: list[str]
    confidence: float


@dataclass
class DebateResult:
    turns: list[DebateTurn]
    final_answer: str
    final_evidence: list[str]
    misinformation_suppressed: bool = True
    ambiguity_coverage: float = 0.0


@dataclass
class QueryTrace:
    dataset_name: str
    method: str
    query_id: str
    question: str
    gold_answer: str | None
    predicted_answer: str
    gold_doc_ids: list[str]
    initial_ranked_ids: list[str]
    reranked_ids: list[str]
    selected_ids: list[str]
    final_evidence: list[str]
    agent_turns: list[dict]
    answer_exact_match: float
    answer_f1: float
    supporting_fact_f1: float
    joint_f1: float
    misinformation_suppressed: float
    ambiguity_coverage: float
    faithfulness_score: float = 0.0
    failure_mode: str = "unknown"
    hallucinated_claims: list = field(default_factory=list)
