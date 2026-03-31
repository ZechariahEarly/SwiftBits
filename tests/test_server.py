"""Tests for the MCP server module."""

from unittest.mock import MagicMock, patch

import pytest

from swiftbits.server import _handle_get_documents, _handle_list, _handle_search, create_server


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_collection_metadata.return_value = {
        "embedding_provider": "local",
        "embedding_dimension": 384,
        "created_at": "2025-03-19T10:00:00Z",
    }
    return store


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.embed.return_value = [[0.1] * 384]
    provider.dimension = 384
    provider.name = "local (all-MiniLM-L6-v2)"
    return provider


class TestCreateServer:
    @patch("swiftbits.server.get_provider")
    @patch("swiftbits.server.VectorStore")
    def test_creates_server(self, mock_store_cls, mock_get_provider):
        mock_store_cls.return_value.get_collection_metadata.return_value = {
            "embedding_provider": "local",
            "embedding_dimension": 384,
        }
        mock_get_provider.return_value = MagicMock()

        server = create_server("default", data_dir="/tmp/test")
        assert server is not None
        assert server.name == "swiftbits"

    @patch("swiftbits.server.VectorStore")
    def test_nonexistent_collection_raises(self, mock_store_cls):
        mock_store_cls.return_value.get_collection_metadata.return_value = None

        with pytest.raises(ValueError, match="does not exist"):
            create_server("nonexistent", data_dir="/tmp/test")


class TestToolRegistration:
    @patch("swiftbits.server.get_provider")
    @patch("swiftbits.server.VectorStore")
    @pytest.mark.asyncio
    async def test_three_tools_registered(self, mock_store_cls, mock_get_provider):
        from mcp.types import ListToolsRequest

        mock_store_cls.return_value.get_collection_metadata.return_value = {
            "embedding_provider": "local",
            "embedding_dimension": 384,
        }
        mock_get_provider.return_value = MagicMock()

        server = create_server("default", data_dir="/tmp/test")

        # Access the registered handler directly
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))
        tools = result.root.tools
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert "search_documents" in names
        assert "list_indexed_documents" in names
        assert "get_documents" in names

    @patch("swiftbits.server.get_provider")
    @patch("swiftbits.server.VectorStore")
    @pytest.mark.asyncio
    async def test_tool_schemas(self, mock_store_cls, mock_get_provider):
        from mcp.types import ListToolsRequest

        mock_store_cls.return_value.get_collection_metadata.return_value = {
            "embedding_provider": "local",
            "embedding_dimension": 384,
        }
        mock_get_provider.return_value = MagicMock()

        server = create_server("default", data_dir="/tmp/test")
        handler = server.request_handlers[ListToolsRequest]
        result = await handler(ListToolsRequest(method="tools/list"))
        tools = result.root.tools

        search_tool = [t for t in tools if t.name == "search_documents"][0]
        assert "query" in search_tool.inputSchema["properties"]
        assert "query" in search_tool.inputSchema["required"]


class TestHandleSearch:
    def test_search_formats_results(self, mock_store, mock_provider):
        mock_store.query.return_value = [
            {
                "text": "The mitochondria is the powerhouse of the cell.",
                "metadata": {
                    "source": "bio.pdf",
                    "page_numbers": "3,4",
                    "chunk_index": 0,
                },
                "distance": 0.13,
            }
        ]

        result = _handle_search(
            {"query": "powerhouse"}, mock_store, mock_provider, "default"
        )
        assert len(result) == 1
        text = result[0].text
        assert "bio.pdf" in text
        assert "pages 3, 4" in text
        assert "0.87" in text  # relevance = 1 - 0.13

    def test_search_empty_results(self, mock_store, mock_provider):
        mock_store.query.return_value = []
        result = _handle_search(
            {"query": "nothing"}, mock_store, mock_provider, "default"
        )
        assert "No results found" in result[0].text

    def test_search_error_returns_text(self, mock_store, mock_provider):
        mock_provider.embed.side_effect = Exception("model error")
        result = _handle_search(
            {"query": "test"}, mock_store, mock_provider, "default"
        )
        assert "Search error" in result[0].text

    def test_relevance_score_calculation(self, mock_store, mock_provider):
        mock_store.query.return_value = [
            {
                "text": "text",
                "metadata": {"source": "a.pdf", "page_numbers": "1"},
                "distance": 0.25,
            }
        ]
        result = _handle_search(
            {"query": "test"}, mock_store, mock_provider, "default"
        )
        assert "0.75" in result[0].text  # 1 - 0.25


class TestHandleList:
    def test_list_formats_documents(self, mock_store):
        mock_store.list_documents.return_value = [
            {"source": "bio.pdf", "chunk_count": 34, "vectorized_at": "2025-03-19T10:00:00Z"},
            {"source": "notes.pdf", "chunk_count": 28, "vectorized_at": "2025-03-18T10:00:00Z"},
        ]

        result = _handle_list(mock_store, "default")
        text = result[0].text
        assert "bio.pdf" in text
        assert "34 chunks" in text
        assert "notes.pdf" in text
        assert "Total: 2 documents, 62 chunks" in text

    def test_list_empty_collection(self, mock_store):
        mock_store.list_documents.return_value = []
        result = _handle_list(mock_store, "default")
        assert "No documents indexed" in result[0].text

    def test_list_error_returns_text(self, mock_store):
        mock_store.list_documents.side_effect = ValueError("Collection does not exist")
        result = _handle_list(mock_store, "nonexistent")
        assert "Error" in result[0].text


class TestHandleGetDocuments:
    def test_single_document(self, mock_store):
        mock_store.get_document_chunks.return_value = [
            {"text": "First chunk.", "metadata": {"source": "paper.pdf", "page_numbers": "1", "chunk_index": 0}},
            {"text": "Second chunk.", "metadata": {"source": "paper.pdf", "page_numbers": "1,2", "chunk_index": 1}},
        ]

        result = _handle_get_documents({"sources": ["paper.pdf"]}, mock_store, "default")
        text = result[0].text
        assert "paper.pdf" in text
        assert "2 chunks" in text
        assert "2 pages" in text
        assert "First chunk." in text
        assert "Second chunk." in text

    def test_multiple_documents(self, mock_store):
        def side_effect(collection, source):
            if source == "a.pdf":
                return [{"text": "Content A", "metadata": {"source": "a.pdf", "page_numbers": "1", "chunk_index": 0}}]
            elif source == "b.pdf":
                return [{"text": "Content B", "metadata": {"source": "b.pdf", "page_numbers": "1", "chunk_index": 0}}]

        mock_store.get_document_chunks.side_effect = side_effect

        result = _handle_get_documents({"sources": ["a.pdf", "b.pdf"]}, mock_store, "default")
        text = result[0].text
        assert "a.pdf" in text
        assert "b.pdf" in text
        assert "Content A" in text
        assert "Content B" in text

    def test_missing_source_includes_error(self, mock_store):
        def side_effect(collection, source):
            if source == "exists.pdf":
                return [{"text": "Content", "metadata": {"source": "exists.pdf", "page_numbers": "1", "chunk_index": 0}}]
            raise ValueError(f"'{source}' not found in collection '{collection}'")

        mock_store.get_document_chunks.side_effect = side_effect

        result = _handle_get_documents({"sources": ["exists.pdf", "missing.pdf"]}, mock_store, "default")
        text = result[0].text
        assert "exists.pdf" in text
        assert "Content" in text
        assert "missing.pdf" in text
        assert "Error" in text

    def test_empty_sources(self, mock_store):
        result = _handle_get_documents({"sources": []}, mock_store, "default")
        assert "No document sources provided" in result[0].text
