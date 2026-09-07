from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    hotpotqa_path: Path
    ramdocs_path: Path
    output_dir: Path
    retrieval_methods: list[str]
    k_values: list[int]
    rerank_top_k: int
    final_top_k: int
    enable_reranker: bool
    enable_multi_agent_debate: bool
    debate_max_sentences_per_agent: int
    hybrid_alpha: float
    mmr_lambda: float
    dense_backend: str
    dense_model_name: str
    dense_lsa_components: int
    max_hotpot_samples: int | None
    max_ramdocs_queries: int | None
    reranker_backend: str = "rule_based"
    llm_reranker_model: str = "claude-haiku-4-5-20251001"
    enable_faithfulness_check: bool = False
    faithfulness_model: str = "claude-haiku-4-5-20251001"
    enable_llm_generation: bool = False
    generator_model: str = "claude-haiku-4-5-20251001"
    debate_mode: str = "rule"  # "rule" | "llm"

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        payload = json.loads(path.read_text())
        return cls(
            hotpotqa_path=Path(payload["hotpotqa_path"]),
            ramdocs_path=Path(payload["ramdocs_path"]),
            output_dir=Path(payload["output_dir"]),
            retrieval_methods=payload["retrieval_methods"],
            k_values=payload["k_values"],
            rerank_top_k=payload.get("rerank_top_k", 5),
            final_top_k=payload.get("final_top_k", 3),
            enable_reranker=payload.get("enable_reranker", True),
            enable_multi_agent_debate=payload.get("enable_multi_agent_debate", True),
            debate_max_sentences_per_agent=payload.get("debate_max_sentences_per_agent", 2),
            hybrid_alpha=payload["hybrid_alpha"],
            mmr_lambda=payload["mmr_lambda"],
            dense_backend=payload.get("dense_backend", "lsa"),
            dense_model_name=payload.get("dense_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            dense_lsa_components=payload.get("dense_lsa_components", 128),
            max_hotpot_samples=payload.get("max_hotpot_samples"),
            max_ramdocs_queries=payload.get("max_ramdocs_queries"),
            reranker_backend=payload.get("reranker_backend", "rule_based"),
            llm_reranker_model=payload.get("llm_reranker_model", "claude-haiku-4-5-20251001"),
            enable_faithfulness_check=payload.get("enable_faithfulness_check", False),
            faithfulness_model=payload.get("faithfulness_model", "claude-haiku-4-5-20251001"),
            enable_llm_generation=payload.get("enable_llm_generation", False),
            generator_model=payload.get("generator_model", "claude-haiku-4-5-20251001"),
            debate_mode=payload.get("debate_mode", "rule"),
        )
