"""Two separate retrieval contexts: reference stories and the active case's facts."""

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import TypeAdapter

from mystery_graph.config import Settings
from mystery_graph.schemas import Mystery, ShortText, StrictModel


class ReferenceStory(StrictModel):
    id: str
    title: ShortText
    theme: ShortText
    scene: ShortText
    explanation: ShortText
    mechanism: ShortText


def load_corpus(path: str = "") -> list[ReferenceStory]:
    raw = (
        Path(path).read_text(encoding="utf-8")
        if path
        else files("mystery_graph").joinpath("data/references.json").read_text(encoding="utf-8")
    )
    stories = TypeAdapter(list[ReferenceStory]).validate_json(raw)
    if not stories or len({s.id for s in stories}) != len(stories):
        raise ValueError("The reference corpus must be non-empty with unique IDs.")
    return stories


class Retrieval:
    def __init__(self, embeddings: Embeddings, stories: list[ReferenceStory], k: int = 3):
        self.embeddings = embeddings
        self.k = k
        self.examples = InMemoryVectorStore(embeddings)
        self.examples.add_documents(
            [
                Document(
                    page_content=s.model_dump_json(),
                    metadata={"reference_id": s.id, "title": s.title},
                )
                for s in stories
            ]
        )
        # Per-instance bounded cache avoids retaining all games for the lifetime of the process.
        self._case_index = lru_cache(maxsize=32)(self._build_case_index)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Retrieval":
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device, "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True},
        )
        return cls(embeddings, load_corpus(settings.corpus_path), settings.retrieval_k)

    def references(self, query: str) -> list[Document]:
        return self.examples.as_retriever(search_kwargs={"k": self.k}).invoke(query)

    def _build_case_index(self, serialized_case: str) -> InMemoryVectorStore:
        case = Mystery.model_validate_json(serialized_case)
        store = InMemoryVectorStore(self.embeddings)
        store.add_documents(
            [Document(page_content=f.text, metadata={"fact_id": f.id}) for f in case.facts]
        )
        return store

    def evidence(self, case: Mystery, query: str) -> list[Document]:
        # The full validated case is the cache key: no retrieval from another player's story.
        return (
            self._case_index(case.model_dump_json())
            .as_retriever(search_kwargs={"k": min(self.k + 2, len(case.facts))})
            .invoke(query)
        )


def serialize_documents(docs: list[Document]) -> str:
    return json.dumps(
        [{"content": d.page_content, "metadata": d.metadata} for d in docs], ensure_ascii=False
    )
