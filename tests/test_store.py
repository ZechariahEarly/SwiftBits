"""Tests for the vector store module."""

import pytest

from swiftbits.processor import Chunk
from swiftbits.store import VectorStore


def _make_chunks(source="test.pdf", count=3):
    """Create test chunks."""
    return [
        Chunk(
            text=f"Chunk {i} text content for {source}",
            metadata={
                "source": source,
                "page_numbers": [1],
                "chunk_index": i,
                "total_chunks": count,
                "char_count": len(f"Chunk {i} text content for {source}"),
            },
        )
        for i in range(count)
    ]


def _make_embeddings(count=3, dim=384):
    """Create test embeddings."""
    return [[float(i) / 10.0] * dim for i in range(count)]


@pytest.fixture
def store(tmp_path):
    return VectorStore(data_dir=str(tmp_path / "chroma"))


class TestAddAndQuery:
    def test_add_and_query_round_trip(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        added = store.add_document("default", chunks, embeddings, "local", 384)
        assert added == 3

        results = store.query("default", embeddings[0], n_results=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_query_result_format(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        results = store.query("default", embeddings[0], n_results=1)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result["text"], str)
        assert isinstance(result["metadata"], dict)
        assert "source" in result["metadata"]
        assert "page_numbers" in result["metadata"]

    def test_n_results_limits_output(self, store):
        chunks = _make_chunks(count=5)
        embeddings = _make_embeddings(count=5)
        store.add_document("default", chunks, embeddings, "local", 384)

        results = store.query("default", embeddings[0], n_results=2)
        assert len(results) == 2

    def test_query_empty_collection(self, store):
        # Create empty collection manually
        store._client.get_or_create_collection(
            name="empty",
            metadata={"embedding_provider": "local", "embedding_dimension": 384},
        )
        results = store.query("empty", [0.0] * 384)
        assert results == []

    def test_query_nonexistent_collection(self, store):
        with pytest.raises(ValueError, match="does not exist"):
            store.query("nonexistent", [0.0] * 384)


class TestEmptyChunksGuard:
    def test_add_empty_chunks_returns_zero(self, store):
        result = store.add_document("default", [], [], "local", 384)
        assert result == 0

    def test_add_empty_chunks_does_not_create_collection(self, store):
        store.add_document("default", [], [], "local", 384)
        assert store.get_collection_metadata("default") is None


class TestReVectorization:
    def test_no_doubling_on_revectorize(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        # Re-vectorize same source
        new_chunks = _make_chunks(count=5)
        new_embeddings = _make_embeddings(count=5)
        added = store.add_document("default", new_chunks, new_embeddings, "local", 384)
        assert added == 5

        # Should have 5, not 8
        docs = store.list_documents("default")
        assert docs[0]["chunk_count"] == 5


class TestRemove:
    def test_remove_document(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        removed = store.remove_document("default", "test.pdf")
        assert removed == 3

        docs = store.list_documents("default")
        assert len(docs) == 0

    def test_remove_document_not_found(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        with pytest.raises(ValueError, match="not found"):
            store.remove_document("default", "nonexistent.pdf")

    def test_remove_collection(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        store.remove_collection("default")
        collections = store.list_collections()
        assert not any(c["name"] == "default" for c in collections)

    def test_remove_nonexistent_collection(self, store):
        with pytest.raises(ValueError, match="does not exist"):
            store.remove_collection("nonexistent")


class TestListCollections:
    def test_list_collections(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("col1", chunks, embeddings, "local", 384)
        store.add_document("col2", _make_chunks("other.pdf"), _make_embeddings(), "local", 384)

        collections = store.list_collections()
        names = [c["name"] for c in collections]
        assert "col1" in names
        assert "col2" in names

    def test_list_collections_counts(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        collections = store.list_collections()
        col = [c for c in collections if c["name"] == "default"][0]
        assert col["document_count"] == 1
        assert col["chunk_count"] == 3
        assert col["embedding_provider"] == "local"


class TestListDocuments:
    def test_list_documents(self, store):
        store.add_document("default", _make_chunks("a.pdf"), _make_embeddings(), "local", 384)
        store.add_document("default", _make_chunks("b.pdf"), _make_embeddings(), "local", 384)

        docs = store.list_documents("default")
        sources = [d["source"] for d in docs]
        assert "a.pdf" in sources
        assert "b.pdf" in sources

    def test_list_documents_nonexistent_collection(self, store):
        with pytest.raises(ValueError, match="does not exist"):
            store.list_documents("nonexistent")


class TestEmbeddingConsistency:
    def test_provider_mismatch_raises(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        with pytest.raises(ValueError, match="Cannot add"):
            store.add_document(
                "default",
                _make_chunks("other.pdf"),
                _make_embeddings(dim=1536),
                "openai",
                1536,
            )

    def test_same_provider_works(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        # Same provider should work fine
        added = store.add_document(
            "default", _make_chunks("other.pdf"), _make_embeddings(), "local", 384
        )
        assert added == 3


class TestGetDocumentChunks:
    def test_returns_chunks_in_order(self, store):
        chunks = _make_chunks(count=5)
        embeddings = _make_embeddings(count=5)
        store.add_document("default", chunks, embeddings, "local", 384)

        result = store.get_document_chunks("default", "test.pdf")
        assert len(result) == 5
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i
            assert "text" in chunk
            assert "metadata" in chunk

    def test_returns_only_requested_source(self, store):
        store.add_document("default", _make_chunks("a.pdf"), _make_embeddings(), "local", 384)
        store.add_document("default", _make_chunks("b.pdf"), _make_embeddings(), "local", 384)

        result = store.get_document_chunks("default", "a.pdf")
        assert all(c["metadata"]["source"] == "a.pdf" for c in result)

    def test_nonexistent_source_raises(self, store):
        store.add_document("default", _make_chunks(), _make_embeddings(), "local", 384)

        with pytest.raises(ValueError, match="not found"):
            store.get_document_chunks("default", "nonexistent.pdf")

    def test_nonexistent_collection_raises(self, store):
        with pytest.raises(ValueError, match="does not exist"):
            store.get_document_chunks("nonexistent", "test.pdf")


class TestMetadata:
    def test_page_numbers_stored_as_string(self, store):
        chunks = [
            Chunk(
                text="text",
                metadata={
                    "source": "test.pdf",
                    "page_numbers": [3, 4],
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "char_count": 4,
                },
            )
        ]
        store.add_document("default", chunks, [[0.0] * 384], "local", 384)

        results = store.query("default", [0.0] * 384, n_results=1)
        assert results[0]["metadata"]["page_numbers"] == "3,4"

    def test_chunk_id_format(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        collection = store._client.get_collection("default")
        all_ids = collection.get()["ids"]
        assert "test.pdf::0" in all_ids
        assert "test.pdf::1" in all_ids
        assert "test.pdf::2" in all_ids

    def test_collection_metadata(self, store):
        chunks = _make_chunks()
        embeddings = _make_embeddings()
        store.add_document("default", chunks, embeddings, "local", 384)

        meta = store.get_collection_metadata("default")
        assert meta is not None
        assert meta["embedding_provider"] == "local"
        assert meta["embedding_dimension"] == 384

    def test_collection_metadata_nonexistent(self, store):
        assert store.get_collection_metadata("nonexistent") is None
