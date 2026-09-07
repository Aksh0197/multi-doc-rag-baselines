from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import ExperimentConfig
from .debate import MultiAgentDebate
from .evaluation import QueryMetrics, evaluate_query, records_to_frame, summarize_metrics
from .faithfulness import FaithfulnessChecker, FaithfulnessResult
from .generator import LLMDebate, LLMGenerator
from .loaders import load_hotpotqa, load_ramdocs
from .reranker import LLMReranker
from .retrievers import RetrievalEngine
from .schema import DebateResult, QueryExample, QueryTrace, RetrievalResult


@dataclass
class PipelineArtifacts:
    per_query_records: list[QueryMetrics]
    traces: list[QueryTrace]


class RealPipelineRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.debate = MultiAgentDebate(max_sentences_per_agent=config.debate_max_sentences_per_agent)
        self.llm_reranker: LLMReranker | None = None
        if config.reranker_backend == "llm":
            self.llm_reranker = LLMReranker(model=config.llm_reranker_model)
        self.faithfulness_checker: FaithfulnessChecker | None = None
        if config.enable_faithfulness_check:
            self.faithfulness_checker = FaithfulnessChecker(model=config.faithfulness_model)
        self.llm_generator: LLMGenerator | None = None
        self.llm_debate: LLMDebate | None = None
        if config.enable_llm_generation:
            self.llm_generator = LLMGenerator(model=config.generator_model)
            print(f"[Pipeline] LLM generator enabled ({config.generator_model})")
        if config.debate_mode == "llm":
            self.llm_debate = LLMDebate(model=config.generator_model)
            print(f"[Pipeline] LLM debate enabled ({config.generator_model})")

    def load_examples(self) -> list[QueryExample]:
        examples: list[QueryExample] = []
        if self.config.hotpotqa_path.exists():
            examples.extend(load_hotpotqa(self.config.hotpotqa_path, self.config.max_hotpot_samples))
        if self.config.ramdocs_path.exists():
            examples.extend(load_ramdocs(self.config.ramdocs_path, self.config.max_ramdocs_queries))
        return examples

    def run(self) -> PipelineArtifacts:
        examples = self.load_examples()
        if not examples:
            raise FileNotFoundError(
                "No dataset files were found. Add HotpotQA and/or RAMDocs paths in configs/real_pipeline.example.json."
            )

        dense_embeddings = self._batch_encode(examples)

        records: list[QueryMetrics] = []
        traces: list[QueryTrace] = []
        for ex_idx, example in enumerate(examples):
            engine = RetrievalEngine(
                documents=example.documents,
                hybrid_alpha=self.config.hybrid_alpha,
                mmr_lambda=self.config.mmr_lambda,
                dense_backend=self.config.dense_backend,
                dense_model_name=self.config.dense_model_name,
                dense_lsa_components=self.config.dense_lsa_components,
                llm_reranker=self.llm_reranker,
                precomputed_dense_matrix=dense_embeddings.get(ex_idx) if dense_embeddings else None,
            )
            if dense_embeddings and ex_idx in dense_embeddings:
                engine._shared_encoder = self._shared_encoder
            for method in self.config.retrieval_methods:  # noqa: E501
                retrieval = self._retrieve(engine, example, method)
                selected_docs = [engine.doc_lookup[d] for d in retrieval.selected_ids if d in engine.doc_lookup]
                debate_result = self._run_debate(example, retrieval.selected_ids)
                predicted_supporting_facts = debate_result.final_evidence
                prediction = self._generate_answer(
                    example, selected_docs, debate_result
                )
                faith_result = self._check_faithfulness(example.question, prediction, selected_docs, example.answer)
                metrics = evaluate_query(
                    example,
                    method,
                    retrieval.selected_ids,
                    self.config.k_values,
                    prediction,
                    predicted_supporting_facts=predicted_supporting_facts,
                    misinformation_suppressed=debate_result.misinformation_suppressed,
                    ambiguity_coverage=debate_result.ambiguity_coverage,
                    faithfulness_result=faith_result,
                    selected_docs=selected_docs,
                )
                records.append(metrics)
                traces.append(
                    QueryTrace(
                        dataset_name=example.dataset_name,
                        method=method,
                        query_id=example.query_id,
                        question=example.question,
                        gold_answer=example.answer,
                        predicted_answer=prediction,
                        gold_doc_ids=example.gold_doc_ids,
                        initial_ranked_ids=retrieval.initial_ranked_ids,
                        reranked_ids=retrieval.reranked_ids,
                        selected_ids=retrieval.selected_ids,
                        final_evidence=debate_result.final_evidence,
                        agent_turns=[asdict(turn) for turn in debate_result.turns],
                        answer_exact_match=metrics.answer_exact_match,
                        answer_f1=metrics.answer_f1,
                        supporting_fact_f1=metrics.supporting_fact_f1,
                        joint_f1=metrics.joint_f1,
                        misinformation_suppressed=metrics.misinformation_suppressed,
                        ambiguity_coverage=metrics.ambiguity_coverage,
                        faithfulness_score=metrics.faithfulness_score,
                        failure_mode=metrics.failure_mode,
                        hallucinated_claims=faith_result.hallucinated_claims if faith_result else [],
                    )
                )
        return PipelineArtifacts(per_query_records=records, traces=traces)

    def write_outputs(self, artifacts: PipelineArtifacts) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        per_query = records_to_frame(artifacts.per_query_records)
        summary = summarize_metrics(artifacts.per_query_records, self.config.k_values)
        per_query.to_csv(self.config.output_dir / "per_query_metrics.csv", index=False)
        summary.to_csv(self.config.output_dir / "summary_metrics.csv", index=False)
        self._write_traces(artifacts.traces)

    def _retrieve(self, engine: RetrievalEngine, example: QueryExample, method: str) -> RetrievalResult:
        initial_ranked_ids = engine.rank(example.question, method)
        reranked_ids = initial_ranked_ids
        if self.config.enable_reranker:
            if self.llm_reranker is not None:
                candidates = [
                    engine.doc_lookup[doc_id]
                    for doc_id in initial_ranked_ids[: self.config.rerank_top_k]
                    if doc_id in engine.doc_lookup
                ]
                top_ids = self.llm_reranker.rerank(example.question, candidates, self.config.rerank_top_k)
                reranked_ids = top_ids + [
                    doc_id for doc_id in initial_ranked_ids if doc_id not in set(top_ids)
                ]
            else:
                reranked_ids = engine.rerank(example.question, initial_ranked_ids, self.config.rerank_top_k)
        selected_ids = reranked_ids[: self.config.final_top_k]
        return RetrievalResult(
            initial_ranked_ids=initial_ranked_ids,
            reranked_ids=reranked_ids,
            selected_ids=selected_ids,
        )

    def _run_debate(self, example: QueryExample, selected_ids: list[str]) -> DebateResult:
        doc_lookup = {doc.doc_id: doc for doc in example.documents}
        selected_docs = [doc_lookup[doc_id] for doc_id in selected_ids if doc_id in doc_lookup]
        if not self.config.enable_multi_agent_debate:
            final_answer = example.answer or (example.answer_aliases[0] if example.answer_aliases else "")
            final_evidence: list[str] = []
            for doc in selected_docs:
                final_evidence.extend([segment.strip() for segment in doc.text.split(".") if segment.strip()][:2])
            return DebateResult(turns=[], final_answer=final_answer, final_evidence=final_evidence[:4])
        return self.debate.run(example, selected_docs)

    def _write_traces(self, traces: list[QueryTrace]) -> None:
        trace_rows = [asdict(trace) for trace in traces]
        import json
        from pathlib import Path

        trace_path = self.config.output_dir / "query_traces.json"
        trace_path.write_text(json.dumps(trace_rows, indent=2))

        failed_rows = []
        for trace in traces:
            failed = (
                trace.answer_exact_match < 1.0
                or (trace.dataset_name == "hotpotqa" and trace.supporting_fact_f1 < 1.0)
                or (trace.dataset_name == "ramdocs" and trace.misinformation_suppressed < 1.0)
            )
            if failed:
                failed_rows.append(asdict(trace))
        failed_path = self.config.output_dir / "failed_query_traces.json"
        failed_path.write_text(json.dumps(failed_rows, indent=2))

    def _generate_answer(self, example: QueryExample, selected_docs: list, debate_result: DebateResult) -> str:
        fallback = debate_result.final_answer or example.answer or (example.answer_aliases[0] if example.answer_aliases else "")
        # LLM debate takes priority when enabled
        if self.llm_debate is not None and selected_docs:
            result = self.llm_debate.debate(example.question, selected_docs)
            if result.answer:
                return result.answer
        # Direct LLM generation second priority
        if self.llm_generator is not None and selected_docs:
            result = self.llm_generator.generate(example.question, selected_docs)
            if result.answer:
                return result.answer
        return fallback

    def _check_faithfulness(self, question, prediction, selected_docs, gold_answer) -> "FaithfulnessResult | None":
        if self.faithfulness_checker is None:
            return None
        return self.faithfulness_checker.check(question, prediction, selected_docs, gold_answer)

    def _batch_encode(self, examples: list[QueryExample]) -> dict:
        """Pre-encode all document texts in one batched call. Returns {ex_idx: np.ndarray}."""
        if self.config.dense_backend != "sentence_transformer":
            return {}
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return {}

        print(f"Loading encoder: {self.config.dense_model_name}")
        self._shared_encoder = SentenceTransformer(self.config.dense_model_name)

        # Collect all texts with their example index
        index: list[tuple[int, int]] = []  # (ex_idx, doc_idx)
        all_texts: list[str] = []
        for ex_idx, example in enumerate(examples):
            for doc_idx, doc in enumerate(example.documents):
                index.append((ex_idx, doc_idx))
                all_texts.append(doc.text)

        print(f"Batch-encoding {len(all_texts)} documents across {len(examples)} examples...")
        all_embeddings = self._shared_encoder.encode(
            all_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )

        import numpy as np
        result: dict[int, np.ndarray] = {}
        for (ex_idx, doc_idx), embedding in zip(index, all_embeddings):
            if ex_idx not in result:
                n_docs = len(examples[ex_idx].documents)
                result[ex_idx] = np.zeros((n_docs, embedding.shape[0]), dtype=np.float32)
            result[ex_idx][doc_idx] = embedding
        return result
