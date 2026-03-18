# multi-doc-rag-baselines

Paper-grounded baseline project for multi-document RAG and top-k context selection.

This repository contains an offline, reproducible mini-benchmark for the two main ideas you highlighted:

- multi-document RAG
- top-k context selection in RAG

The implementation is grounded in your local `MultiDocRAG_Pipeline.docx` plus paper-level method summaries for:

- Lewis et al. RAG
- RankRAG
- Self-RAG
- RAGChecker
- MADAM-RAG

Because the environment did not have the HotpotQA/RAMDocs datasets or local dense-retrieval models preinstalled, this project uses a paper-grounded evaluation set instead of the full coursework datasets. The code is structured so you can swap in HotpotQA/RAMDocs later.

## What is implemented

- `tfidf`: sparse lexical retrieval baseline
- `bm25`: classic BM25 retrieval baseline
- `hybrid`: score interpolation between TF-IDF and BM25
- `hybrid_mmr`: hybrid retrieval plus MMR diversification for better multi-document top-k coverage

## Metrics

- `recall_at_1`, `recall_at_3`, `recall_at_5`
- `mrr`
- `multi_doc_hit_rate`
- `answer_exact_match`
- `answer_f1`

## Run

```bash
python3 src/run_experiments.py
```

## Outputs

Results are written to:

- `results/metrics_summary.csv`
- `results/metrics_summary.json`
- `results/predictions.csv`
- `results/research_notes.md`
