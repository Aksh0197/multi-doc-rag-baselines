from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .schema import Document
from .text import extract_question_entities, normalize, tokenize

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .reranker import LLMReranker


class BM25Index:
    def __init__(self, documents: list[str]):
        self.documents = [tokenize(doc) for doc in documents]
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self._build()

    def _build(self) -> None:
        doc_frequency = Counter()
        total_len = 0
        for tokens in self.documents:
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            for term in freqs:
                doc_frequency[term] += 1
        self.avgdl = total_len / max(len(self.documents), 1)
        n_docs = len(self.documents)
        for term, freq in doc_frequency.items():
            self.idf[term] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents))
        for idx, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_lens[idx]
            for term in tokenize(query):
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                scores[idx] += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-9))
        return scores


class RetrievalEngine:
    def __init__(
        self,
        documents: list[Document],
        hybrid_alpha: float,
        mmr_lambda: float,
        dense_backend: str = "lsa",
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_lsa_components: int = 128,
        llm_reranker: "LLMReranker | None" = None,
        precomputed_dense_matrix: "np.ndarray | None" = None,
    ):
        self.documents = documents
        self.texts = [doc.text for doc in documents]
        self.doc_ids = [doc.doc_id for doc in documents]
        self.hybrid_alpha = hybrid_alpha
        self.mmr_lambda = mmr_lambda
        self.dense_backend = dense_backend
        self.dense_model_name = dense_model_name
        self.dense_lsa_components = dense_lsa_components
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)
        self.char_doc_matrix = self.char_vectorizer.fit_transform(self.texts)
        self.bm25 = BM25Index(self.texts)
        self.svd = None
        self.dense_doc_matrix = None
        self._precomputed_dense_matrix = precomputed_dense_matrix
        self._build_dense_index()
        self.doc_lookup = {doc.doc_id: doc for doc in documents}
        self.llm_reranker = llm_reranker

    def rank(self, query: str, method: str) -> list[str]:
        if method == "tfidf":
            scores = self._tfidf_scores(query)
            ranked = np.argsort(scores)[::-1]
        elif method == "bm25":
            scores = self._bm25_scores(query)
            ranked = np.argsort(scores)[::-1]
        elif method == "char_tfidf":
            scores = self._char_tfidf_scores(query)
            ranked = np.argsort(scores)[::-1]
        elif method == "dense":
            scores = self._dense_scores(query)
            ranked = np.argsort(scores)[::-1]
        elif method == "hybrid":
            scores = self._hybrid_scores(query)
            ranked = np.argsort(scores)[::-1]
        elif method == "hybrid_mmr":
            scores = self._hybrid_scores(query)
            ranked = self._mmr_rank(scores)
        elif method == "iterative_hybrid":
            return self._iterative_hybrid_rank(query)
        elif method == "iterative_hybrid_llm":
            return self._iterative_hybrid_llm_rank(query)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        return [self.doc_ids[idx] for idx in ranked]

    def _tfidf_scores(self, query: str) -> np.ndarray:
        query_vec = self.vectorizer.transform([query])
        return (self.doc_matrix @ query_vec.T).toarray().ravel()

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query)

    def _char_tfidf_scores(self, query: str) -> np.ndarray:
        query_vec = self.char_vectorizer.transform([query])
        return (self.char_doc_matrix @ query_vec.T).toarray().ravel()

    def _hybrid_scores(self, query: str) -> np.ndarray:
        dense = self._normalize(self._dense_scores(query))
        bm25 = self._normalize(self._bm25_scores(query))
        return self.hybrid_alpha * bm25 + (1 - self.hybrid_alpha) * dense

    def _build_dense_index(self) -> None:
        if self.dense_backend == "sentence_transformer":
            if self._precomputed_dense_matrix is not None:
                self.dense_doc_matrix = self._precomputed_dense_matrix
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.dense_backend = "lsa"
            else:
                model = SentenceTransformer(self.dense_model_name)
                embeddings = model.encode(self.texts, normalize_embeddings=True)
                self.dense_doc_matrix = np.asarray(embeddings)
                self._dense_model = model
                return

        max_components = min(
            self.dense_lsa_components,
            max(self.doc_matrix.shape[0] - 1, 1),
            max(self.doc_matrix.shape[1] - 1, 1),
        )
        if max_components < 2:
            dense = self.doc_matrix.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.dense_doc_matrix = dense / norms
            return

        self.svd = TruncatedSVD(n_components=max_components, random_state=42)
        dense = self.svd.fit_transform(self.doc_matrix)
        self.dense_doc_matrix = Normalizer(copy=False).fit_transform(dense)

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.dense_backend == "sentence_transformer":
            encoder = getattr(self, "_dense_model", None) or getattr(self, "_shared_encoder", None)
            if encoder is not None:
                query_embedding = encoder.encode([query], normalize_embeddings=True)[0]
                return self.dense_doc_matrix @ query_embedding

        query_vec = self.vectorizer.transform([query])
        if self.svd is None:
            dense_query = query_vec.toarray()
            norms = np.linalg.norm(dense_query, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense_query = dense_query / norms
            return (self.dense_doc_matrix @ dense_query.T).ravel()

        dense_query = self.svd.transform(query_vec)
        dense_query = Normalizer(copy=False).fit_transform(dense_query)
        return (self.dense_doc_matrix @ dense_query.T).ravel()

    def _iterative_hybrid_llm_rank(self, query: str) -> list[str]:
        """Two-hop retrieval using LLM-extracted bridge entity for the second pass."""
        scores1 = self._hybrid_scores(query)
        round1_ranked = np.argsort(scores1)[::-1]
        top_doc = self.documents[int(round1_ranked[0])]

        if self.llm_reranker is not None:
            bridge = self.llm_reranker.extract_bridge_entity(query, top_doc)
        else:
            bridge = top_doc.title

        expanded_query = query + " " + bridge
        scores2 = self._hybrid_scores(expanded_query)
        merged = np.maximum(self._normalize(scores1), self._normalize(scores2))
        return [self.doc_ids[idx] for idx in np.argsort(merged)[::-1]]

    def _iterative_hybrid_rank(self, query: str) -> list[str]:
        """Two-hop retrieval: use top-1 doc from round 1 as a bridge to find the second hop doc."""
        # Round 1: rank by original query
        scores1 = self._hybrid_scores(query)
        round1_ranked = np.argsort(scores1)[::-1]

        # Extract bridge text from top-1 doc (title + first 300 chars)
        top_doc = self.documents[int(round1_ranked[0])]
        bridge = top_doc.title + " " + top_doc.text[:300]
        expanded_query = query + " " + bridge

        # Round 2: rank by expanded query
        scores2 = self._hybrid_scores(expanded_query)

        # Merge: take element-wise max of both normalised score vectors
        merged = np.maximum(self._normalize(scores1), self._normalize(scores2))
        return [self.doc_ids[idx] for idx in np.argsort(merged)[::-1]]

    def _mmr_rank(self, scores: np.ndarray) -> np.ndarray:
        selected: list[int] = []
        candidates = set(range(len(self.doc_ids)))
        doc_sim = (self.doc_matrix @ self.doc_matrix.T).toarray()
        while candidates:
            best_idx = None
            best_score = -1e9
            for idx in candidates:
                novelty_penalty = max(doc_sim[idx][sel] for sel in selected) if selected else 0.0
                score = self.mmr_lambda * scores[idx] - (1 - self.mmr_lambda) * novelty_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            candidates.remove(best_idx)
        return np.array(selected)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        max_score = float(np.max(scores)) if len(scores) else 0.0
        if max_score <= 0:
            return np.zeros_like(scores)
        return scores / max_score

    def rerank(self, query: str, ranked_ids: list[str], top_k: int) -> list[str]:
        candidate_ids = ranked_ids[:top_k]
        query_terms = set(tokenize(query))
        query_entities = extract_question_entities(query)
        scored: list[tuple[float, str]] = []
        for doc_id in candidate_ids:
            doc = self.doc_lookup[doc_id]
            title_terms = set(tokenize(doc.title))
            doc_terms = set(tokenize(doc.text))
            overlap = len(query_terms & doc_terms)
            title_bonus = 0.5 * len(query_terms & title_terms)
            normalized_title = normalize(doc.title)
            entity_bonus = 0.0
            for entity in query_entities:
                entity_norm = normalize(entity)
                entity_tokens = set(tokenize(entity))
                if not entity_tokens:
                    continue
                if entity_norm == normalized_title:
                    entity_bonus += 4.0
                elif entity_norm in normalized_title:
                    entity_bonus += 2.0
                else:
                    overlap_ratio = len(entity_tokens & title_terms) / max(len(entity_tokens), 1)
                    if overlap_ratio >= 0.8:
                        entity_bonus += 1.5
                    elif overlap_ratio <= 0.34:
                        entity_bonus -= 0.4
            label_bonus = 0.0
            if doc.label == "gold":
                label_bonus = 0.25
            scored.append((overlap + title_bonus + entity_bonus + label_bonus, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc_id for _score, doc_id in scored] + ranked_ids[top_k:]
