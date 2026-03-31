"""Tests for the embedding engine module. All mocked — no model downloads."""

from unittest.mock import MagicMock, patch

import pytest

from swiftbits.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_provider,
)


class TestGetProvider:
    @patch("swiftbits.embeddings.LocalEmbeddingProvider.__init__", return_value=None)
    def test_factory_returns_local(self, mock_init):
        provider = get_provider("local")
        assert isinstance(provider, LocalEmbeddingProvider)

    @patch("swiftbits.embeddings.OpenAIEmbeddingProvider.__init__", return_value=None)
    def test_factory_returns_openai(self, mock_init):
        provider = get_provider("openai", api_key="sk-test")
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_factory_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_provider("unknown")

    def test_factory_openai_no_key(self):
        with pytest.raises(ValueError, match="OpenAI API key required"):
            get_provider("openai")

    def test_factory_openai_empty_key(self):
        with pytest.raises(ValueError, match="OpenAI API key required"):
            get_provider("openai", api_key="")


class TestLocalProvider:
    @patch("swiftbits.embeddings.LocalEmbeddingProvider.__init__", return_value=None)
    def test_dimension(self, mock_init):
        provider = LocalEmbeddingProvider()
        assert provider.dimension == 384

    @patch("swiftbits.embeddings.LocalEmbeddingProvider.__init__", return_value=None)
    def test_name(self, mock_init):
        provider = LocalEmbeddingProvider()
        assert provider.name == "local (all-MiniLM-L6-v2)"

    def test_embed_calls_model(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            provider = LocalEmbeddingProvider()
            result = provider.embed(["hello", "world"])

        mock_model.encode.assert_called_once_with(
            ["hello", "world"], show_progress_bar=False
        )
        assert len(result) == 2
        assert len(result[0]) == 384


class TestOpenAIProvider:
    def test_dimension(self):
        with patch("openai.OpenAI"):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            assert provider.dimension == 1536

    def test_name(self):
        with patch("openai.OpenAI"):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            assert provider.name == "openai (text-embedding-3-small)"

    def test_embed_calls_api(self):
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_item]
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            result = provider.embed(["hello"])

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["hello"]
        )
        assert len(result) == 1
        assert len(result[0]) == 1536

    def test_embed_batches_over_100(self):
        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * 1536

        response_100 = MagicMock()
        response_100.data = [mock_item] * 100
        response_50 = MagicMock()
        response_50.data = [mock_item] * 50
        mock_client.embeddings.create.side_effect = [response_100, response_50]

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            texts = [f"text {i}" for i in range(150)]
            result = provider.embed(texts)

        assert mock_client.embeddings.create.call_count == 2
        assert len(result) == 150

    def test_auth_error(self):
        from openai import AuthenticationError

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_client.embeddings.create.side_effect = AuthenticationError(
            message="bad key", response=mock_response, body=None
        )

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="bad-key")
            with pytest.raises(ValueError, match="Invalid OpenAI API key"):
                provider.embed(["test"])

    def test_rate_limit_error(self):
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.embeddings.create.side_effect = RateLimitError(
            message="rate limited", response=mock_response, body=None
        )

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            with pytest.raises(ValueError, match="rate limit"):
                provider.embed(["test"])

    def test_connection_error(self):
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test")
            with pytest.raises(ValueError, match="Could not connect"):
                provider.embed(["test"])
