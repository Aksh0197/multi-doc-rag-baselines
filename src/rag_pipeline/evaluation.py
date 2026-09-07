from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .faithfulness import FaithfulnessResult, deterministic_failure_mode
from .schema import Document, QueryExample
from .text import normalize, token_f1


@dataclass
class QueryMetrics:
    dataset_name: str
    method: str
    query_id: str
    recall_at_k: dict[int, float]
    mrr: float
    multi_doc_hit_rate: float
    answer_exact_match: float
    answer_f1: float
    supporting_fact_f1: float
    joint_exact_match: float
    joint_f1: float
    misinformation_suppressed: float
    ambiguity_coverage: float
    faithfulness_score: float = 0.0
    failure_mode: str = "unknown"
    hallucination_rate: float = 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & set(gold_ids)) / max(len(gold_ids), 1)


def reciprocal_rank(ranked_ids: list[str], gold_ids: list[str]) -> float:
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / idx
    return 0.0


def multi_doc_hit_rate(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    if len(gold_ids) < 2:
        return float(any(doc_id in ranked_ids[:k] for doc_id in gold_ids))
    return float(set(gold_ids).issubset(set(ranked_ids[:k])))


def answer_exact_match(prediction: str, aliases: list[str]) -> float:
    normalized = normalize(prediction)
    return float(any(normalized == normalize(alias) for alias in aliases if alias))


def answer_f1_score(prediction: str, aliases: list[str]) -> float:
    if not aliases:
        return 0.0
    return max(token_f1(prediction, alias) for alias in aliases if alias)


def supporting_fact_f1_score(predicted_facts: list[str], gold_facts: list[str]) -> float:
    if not predicted_facts or not gold_facts:
        return 0.0
    prediction_text = " ".join(predicted_facts)
    gold_text = " ".join(gold_facts)
    return token_f1(prediction_text, gold_text)


def joint_exact_match(answer_em: float, supporting_fact_f1: float, sp_threshold: float = 0.5) -> float:
    """answer EM AND supporting facts F1 above threshold (HotpotQA-style joint metric)."""
    return float(answer_em == 1.0 and supporting_fact_f1 >= sp_threshold)


def joint_f1(answer_f1: float, supporting_fact_f1: float) -> float:
    """Multiplicative joint F1: penalises missing either answer or supporting facts."""
    return answer_f1 * supporting_fact_f1


def evaluate_query(
    example: QueryExample,
    method: str,
    ranked_ids: list[str],
    k_values: list[int],
    prediction: str,
    predicted_supporting_facts: list[str] | None = None,
    misinformation_suppressed: bool = True,
    ambiguity_coverage: float = 0.0,
    faithfulness_result: FaithfulnessResult | None = None,
    selected_docs: list[Document] | None = None,
) -> QueryMetrics:
    answer_em = answer_exact_match(prediction, example.answer_aliases)
    a_f1 = answer_f1_score(prediction, example.answer_aliases)
    support_f1 = supporting_fact_f1_score(predicted_supporting_facts or [], example.supporting_facts)
    mdh = multi_doc_hit_rate(ranked_ids, example.gold_doc_ids, min(k_values))

    # Always compute deterministic failure mode as baseline
    det_fail_mode = deterministic_failure_mode(answer_em, mdh)

    if faithfulness_result is not None and faithfulness_result.failure_mode != "unknown":
        faith_score = faithfulness_result.faithfulness_score
        fail_mode = faithfulness_result.failure_mode
        halluc_rate = float(len(faithfulness_result.hallucinated_claims) > 0)
    else:
        faith_score = faithfulness_result.faithfulness_score if faithfulness_result else 0.0
        fail_mode = det_fail_mode
        halluc_rate = 0.0

    return QueryMetrics(
        dataset_name=example.dataset_name,
        method=method,
        query_id=example.query_id,
        recall_at_k={k: recall_at_k(ranked_ids, example.gold_doc_ids, k) for k in k_values},
        mrr=reciprocal_rank(ranked_ids, example.gold_doc_ids),
        multi_doc_hit_rate=mdh,
        answer_exact_match=answer_em,
        answer_f1=a_f1,
        supporting_fact_f1=support_f1,
        joint_exact_match=joint_exact_match(answer_em, support_f1),
        joint_f1=joint_f1(a_f1, support_f1),
        misinformation_suppressed=float(misinformation_suppressed),
        ambiguity_coverage=ambiguity_coverage,
        faithfulness_score=faith_score,
        failure_mode=fail_mode,
        hallucination_rate=halluc_rate,
    )


def summarize_metrics(records: list[QueryMetrics], k_values: list[int]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "dataset_name": record.dataset_name,
            "method": record.method,
            "query_id": record.query_id,
            "mrr": record.mrr,
            "multi_doc_hit_rate": record.multi_doc_hit_rate,
            "answer_exact_match": record.answer_exact_match,
            "answer_f1": record.answer_f1,
            "supporting_fact_f1": record.supporting_fact_f1,
            "joint_exact_match": record.joint_exact_match,
            "joint_f1": record.joint_f1,
            "misinformation_suppressed": record.misinformation_suppressed,
            "ambiguity_coverage": record.ambiguity_coverage,
            "faithfulness_score": record.faithfulness_score,
            "hallucination_rate": record.hallucination_rate,
            "failure_mode_retrieval": float(record.failure_mode == "retrieval_failure"),
            "failure_mode_generation": float(record.failure_mode == "generation_failure"),
            "failure_mode_correct": float(record.failure_mode == "correct"),
        }
        for k, value in record.recall_at_k.items():
            row[f"recall_at_{k}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    group_cols = ["dataset_name", "method"]
    metric_cols = [col for col in frame.columns if col not in {"query_id", *group_cols}]
    return frame.groupby(group_cols, as_index=False)[metric_cols].mean()


def records_to_frame(records: list[QueryMetrics]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = asdict(record)
        for key, value in record.recall_at_k.items():
            row[f"recall_at_{key}"] = value
        row.pop("recall_at_k", None)
        rows.append(row)
    return pd.DataFrame(rows)
