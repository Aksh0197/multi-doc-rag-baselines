# Research Notes

Best offline baseline on the paper-grounded benchmark: `hybrid`.

## Key results

- `recall_at_3`: 1.0
- `mrr`: 0.9643
- `multi_doc_hit_rate`: 1.0
- `answer_exact_match`: 1.0
- `answer_f1`: 1.0

## Interpretation

- `hybrid` tests the main top-k context claim from your pipeline: mixing sparse and semantic relevance works better than a single retriever.
- `hybrid_mmr` adds diversity pressure at context selection time, which is useful when a question depends on multiple distinct documents.
- `bm25` stays strong on explicit phrase matching, but it is less robust on paraphrased multi-document questions.
- `tfidf` is the lightest baseline and acts as the lexical floor.

## Next implementation step for the full project

Replace the paper-grounded benchmark with real `HotpotQA` and `RAMDocs` loaders, then keep the same evaluation shape:

1. retrieval recall and MRR on HotpotQA
2. multi-document ambiguity and misinformation metrics on RAMDocs
3. reranker and debate ablations after the retriever is stable
