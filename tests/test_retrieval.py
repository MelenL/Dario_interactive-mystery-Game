import hashlib

from langchain_core.embeddings import Embeddings

from mystery_graph.retrieval import Retrieval, load_corpus
from mystery_graph.schemas import Mystery


class OfflineEmbeddings(Embeddings):
    """Small deterministic fixture; not a substitute for semantic model evaluation."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 64
        for token in text.lower().split():
            slot = hashlib.sha256(token.encode()).digest()[0] % len(vector)
            vector[slot] += 1
        return vector


def test_retrieval_is_scoped_to_active_case(case):
    retrieval = Retrieval(OfflineEmbeddings(), load_corpus(), k=2)
    data = case.model_dump()
    for fact in data["facts"]:
        fact["text"] = "Another case: " + fact["text"]
    other_case = Mystery.model_validate(data)
    first = retrieval.evidence(case, "cold box")
    second = retrieval.evidence(other_case, "cold box")
    assert all(not d.page_content.startswith("Another case:") for d in first)
    assert all(d.page_content.startswith("Another case:") for d in second)


def test_bundled_reference_corpus_and_retriever():
    stories = load_corpus()
    retrieval = Retrieval(OfflineEmbeddings(), stories, k=2)
    docs = retrieval.references("archive cold storage")
    assert len(docs) == 2
    assert all(d.metadata["reference_id"] in {s.id for s in stories} for d in docs)
