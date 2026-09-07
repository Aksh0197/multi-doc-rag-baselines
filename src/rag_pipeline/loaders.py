from __future__ import annotations

import json
from pathlib import Path

from .schema import Document, QueryExample


def load_hotpotqa(path: Path, limit: int | None = None) -> list[QueryExample]:
    payload = json.loads(path.read_text())
    examples: list[QueryExample] = []
    for idx, row in enumerate(payload):
        if limit is not None and idx >= limit:
            break
        if isinstance(row.get("context"), list):
            contexts = [entry[0] for entry in row["context"]]
            sentences = [entry[1] for entry in row["context"]]
        else:
            contexts = row.get("context", {}).get("title", [])
            sentences = row["context"]["sentences"]

        documents: list[Document] = []
        title_to_doc_id: dict[str, str] = {}
        for doc_index, title in enumerate(contexts):
            doc_id = f"{row['_id']}_doc_{doc_index}"
            title_to_doc_id[title] = doc_id
            text = " ".join(sentences[doc_index])
            documents.append(Document(doc_id=doc_id, title=title, text=text))

        supporting_titles = {title for title, _sent_idx in row.get("supporting_facts", [])}
        gold_doc_ids = [title_to_doc_id[title] for title in supporting_titles if title in title_to_doc_id]
        supporting_facts = []
        for title, sentence_index in row.get("supporting_facts", []):
            if title in contexts:
                title_pos = contexts.index(title)
                doc_sentences = sentences[title_pos]
                if 0 <= sentence_index < len(doc_sentences):
                    supporting_facts.append(doc_sentences[sentence_index])

        examples.append(
            QueryExample(
                query_id=row["_id"],
                question=row["question"],
                documents=documents,
                gold_doc_ids=gold_doc_ids,
                answer=row.get("answer"),
                answer_aliases=[row.get("answer", "")],
                supporting_facts=supporting_facts,
                dataset_name="hotpotqa",
                metadata={"type": row.get("type"), "level": row.get("level")},
            )
        )
    return examples


def load_ramdocs(path: Path, limit: int | None = None) -> list[QueryExample]:
    examples: list[QueryExample] = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            row = json.loads(line)
            query_id = row.get("query_id") or row.get("id") or row["question"]
            example = QueryExample(
                query_id=query_id,
                question=row["question"],
                documents=[],
                gold_doc_ids=[],
                answer=row.get("gold_answers", [None])[0],
                answer_aliases=row.get("gold_answers", []),
                wrong_answers=row.get("wrong_answers", []),
                gold_answers=row.get("gold_answers", []),
                dataset_name="ramdocs",
                metadata={"disambig_entity": row.get("disambig_entity", [])},
            )

            for doc_index, doc in enumerate(row.get("documents", [])):
                doc_id = f"{query_id}_doc_{doc_index}"
                label = doc.get("type")
                normalized_label = "gold" if label == "correct" else label
                doc_text = doc.get("text") or doc.get("document") or doc.get("context") or ""
                title = doc.get("title") or doc_id
                example.documents.append(
                    Document(doc_id=doc_id, title=title, text=doc_text, label=normalized_label, metadata=doc)
                )
                if normalized_label == "gold":
                    example.gold_doc_ids.append(doc_id)

            examples.append(example)
    return examples
