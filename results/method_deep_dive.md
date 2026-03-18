# Method Deep Dive

## What matters most from the papers

### 1. Multi-document RAG is not just retrieval

The main lesson from the MADAM-RAG direction is that once the retriever surfaces conflicting evidence, the system needs a conflict-resolution layer instead of simple concatenation. For your project, that means:

- retrieve several candidate documents
- keep document identity separate
- let each document produce its own provisional answer or evidence
- aggregate after comparing conflicts

This is why a multi-agent or document-wise reasoning stage is useful: it prevents one noisy document from dominating a flat context window.

### 2. Top-k context is a selection problem, not only a scoring problem

RankRAG is important because it reframes ranking as part of generation rather than a separate utility model. In practice, your system should not stop after initial retrieval. A strong pipeline is:

1. retrieve `k_initial`
2. rerank with a stronger model or generator-aware scorer
3. compress to `k_final`

In small or offline settings, MMR is a good stand-in for this idea because it forces diversity across the final top-k set. That is valuable on multi-hop questions where two relevant documents may be semantically different from each other.

### 3. Self-RAG matters for evidence, not only answers

Self-RAG is useful here mainly for two behaviors:

- adaptive retrieval when the current evidence is not enough
- explicit evidence tracing and self-critique

For your implementation, the easiest win is evidence-aware outputs:

- final answer
- supporting snippets
- optional confidence or critique field

That makes debugging much easier than answer-only RAG.

### 4. RAGChecker should be used after retrieval is stable

Do not add claim-level faithfulness checks before the retriever is working. First stabilize:

- Recall@K
- MRR
- multi-document coverage

Then add:

- claim support rate
- hallucination rate
- answer faithfulness

## Recommended implementation order

### Phase 1: Retrieval foundation

- baseline 1: TF-IDF
- baseline 2: BM25
- baseline 3: hybrid sparse plus semantic retrieval
- baseline 4: hybrid plus diversified top-k selection

Goal:

- maximize recall of all gold documents, not only the first one

### Phase 2: Context reduction

Use two-level context control:

- `k_initial = 8 to 10`
- `k_final = 3 to 5`

Selection policy:

- relevance score
- diversity bonus
- optional document-type penalty for noisy sources

### Phase 3: Multi-document reasoning

Keep each selected document separate and produce:

- document summary
- local answer
- local evidence span

Then aggregate:

- majority agreement
- confidence-weighted merge
- conflict flag if answers disagree

### Phase 4: Evaluation

For HotpotQA:

- Recall@2, Recall@5, MRR
- answer F1 and EM
- supporting fact F1

For RAMDocs:

- gold precision@k
- misinformation suppression rate
- ambiguity coverage
- ranking suppression accuracy

## Current offline benchmark result

This folder contains a paper-grounded mini-benchmark because the full datasets and dense models were not locally available in the environment.

Best retriever on the mini-benchmark:

- `hybrid`

Best summary metrics:

- `recall_at_3 = 1.0000`
- `recall_at_5 = 1.0000`
- `mrr = 0.9643`
- `multi_doc_hit_rate = 1.0000`

Interpretation:

- the mixed retriever is the strongest default starting point
- BM25 alone is competitive but weaker on multi-document coverage
- adding top-k diversification is useful, but on this small benchmark it did not beat pure hybrid ranking

## What to implement next in the real project

1. Load HotpotQA distractor data and treat the 10 paragraphs as the per-query retrieval pool.
2. Load RAMDocs and keep gold, misinformation, and noise labels separate through the pipeline.
3. Replace TF-IDF semantic proxy with a real embedding model.
4. Add reranking after retrieval and before final top-k compression.
5. Add document-wise debate only after retrieval metrics are acceptable.

## Source papers

- Lewis et al., RAG: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- RankRAG: [arXiv:2407.02485](https://arxiv.org/abs/2407.02485)
- Self-RAG: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- RAGChecker: [arXiv:2408.08067](https://arxiv.org/abs/2408.08067)
- Retrieval-Augmented Generation with Conflicting Evidence / MADAM-RAG direction: [arXiv:2504.13079](https://arxiv.org/abs/2504.13079)

## Limitation

The supplied PDF files could not be cleanly text-extracted with the currently available local tools, so the implementation is grounded primarily in:

- your `MultiDocRAG_Pipeline.docx`
- runnable offline baselines
- primary paper metadata and abstracts
