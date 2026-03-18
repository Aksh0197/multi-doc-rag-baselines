# Simple English Summary

## What I did

I created a small RAG research project in:

`/Users/akshunagnihotri/Documents/Playground/rag_paper_baselines`

The goal was to study the two main ideas from your papers:

- multi-document RAG
- top-k context selection in RAG

I then built a small offline benchmark and tested a few common retrieval baselines.

## Step-by-step approach

### Step 1: Read your project document

I first read your file:

- `MultiDocRAG_Pipeline.docx`

From that file, I extracted the main pipeline ideas:

- HotpotQA for top-k retrieval evaluation
- RAMDocs for misinformation and ambiguity evaluation
- RankRAG for reranking
- MADAM-RAG for multi-document debate
- Self-RAG for evidence generation
- RAGChecker for faithfulness metrics

### Step 2: Check what is available locally

I checked whether this machine already had:

- dataset files
- PDF text extraction tools
- dense retrieval libraries
- transformer models

Result:

- your DOCX file was readable
- the PDF files were not cleanly extractable with the local tools available
- HotpotQA and RAMDocs were not locally installed
- sentence-transformer and FAISS style dense-retrieval packages were also not available locally

Because of that, I used an offline fallback approach instead of pretending I had the full datasets.

### Step 3: Build a small research benchmark

Since the full datasets were not available locally, I created a paper-grounded benchmark.

This benchmark contains:

- a small corpus of document chunks summarizing the main papers and methods
- a set of evaluation questions based on your project pipeline
- gold document labels for each question

Files used:

- `data/paper_corpus.json`
- `data/eval_questions.json`

### Step 4: Implement baseline retrievers

I implemented these baselines:

- `tfidf`
- `bm25`
- `hybrid`
- `hybrid_mmr`

What they mean:

- `tfidf`: simple keyword-weighted retrieval
- `bm25`: stronger lexical retrieval baseline
- `hybrid`: combines TF-IDF style and BM25-style scores
- `hybrid_mmr`: hybrid retrieval plus diversity, so top-k documents are less redundant

### Step 5: Add multi-document evaluation

I added metrics that check whether the system retrieves all required documents for a question, not just one document.

This is important because multi-document RAG needs:

- more than one relevant document
- good top-k selection
- less duplication inside the final context

### Step 6: Run experiments

I ran the experiment script:

- `src/run_experiments.py`

That script produced:

- `results/metrics_summary.csv`
- `results/metrics_summary.json`
- `results/predictions.csv`

## Did I train a model?

No. I did not train a new model.

Why:

- the required full datasets were not available locally
- the dense retrieval model packages were not installed
- this phase was focused on building a working baseline pipeline first

So this is an evaluation and prototyping step, not a training step.

What I actually did instead:

- built retrieval baselines
- ran them on a small benchmark
- compared their retrieval metrics

## What dataset was used?

### What the papers recommend

Your project document recommends:

- HotpotQA
- RAMDocs

### What I actually used in this run

I used a custom offline benchmark made from:

- your `MultiDocRAG_Pipeline.docx`
- paper and method summaries

This was necessary because the real datasets were not locally available in the environment.

So the current metrics are:

- useful for comparing baseline retrieval methods
- not yet the final coursework metrics on HotpotQA or RAMDocs

## Result metrics

### Best method

Best current method:

- `hybrid`

### Metrics

- `hybrid`: `recall_at_1=0.6786`, `recall_at_3=1.0000`, `recall_at_5=1.0000`, `mrr=0.9643`, `multi_doc_hit_rate=1.0000`
- `tfidf`: `recall_at_3=0.9643`, `multi_doc_hit_rate=0.9286`
- `hybrid_mmr`: `recall_at_3=0.9643`, `multi_doc_hit_rate=0.9286`
- `bm25`: `recall_at_3=0.9286`, `multi_doc_hit_rate=0.8571`

## Simple meaning of the metrics

- `recall_at_1`: did the first result already contain a gold document?
- `recall_at_3`: were the needed documents found in the top 3?
- `recall_at_5`: were the needed documents found in the top 5?
- `mrr`: how early the first correct document appears
- `multi_doc_hit_rate`: whether all required documents were captured for multi-document questions

## Main conclusion

The current offline experiments show:

- hybrid retrieval is the best starting point
- multi-document retrieval benefits from using more than one scoring method
- simple lexical baselines work, but they are weaker than the hybrid method for multi-document coverage

## What should happen next

To move this into a real RAG project:

1. download and load HotpotQA
2. download and load RAMDocs
3. replace the offline benchmark with the real datasets
4. add real dense embeddings
5. add reranking
6. add debate / multi-agent aggregation
7. measure final answer quality and faithfulness

## Important limitation

This is not yet the final full RAG system from the papers.

It is the first practical baseline stage:

- project setup
- baseline retrieval
- metric evaluation
- method comparison

That was the correct first step because we need retrieval working before adding ranking, debate, and final answer generation.
