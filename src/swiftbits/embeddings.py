"""SwiftBits embedding engine — provider abstraction for text embeddings."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for display."""
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses sentence-transformers/all-MiniLM-L6-v2 for local embeddings."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return 384

    @property
    def name(self) -> str:
        return "local (all-MiniLM-L6-v2)"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses OpenAI's text-embedding-3-small for API-based embeddings."""

    def __init__(self, api_key: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import APIConnectionError, AuthenticationError, RateLimitError

        all_embeddings = []
        batch_size = 100

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                all_embeddings.extend(item.embedding for item in response.data)
        except AuthenticationError:
            raise ValueError("Invalid OpenAI API key")
        except RateLimitError:
            raise ValueError("OpenAI rate limit hit. Try again shortly.")
        except APIConnectionError:
            raise ValueError("Could not connect to OpenAI API")

        return all_embeddings

    @property
    def dimension(self) -> int:
        return 1536

    @property
    def name(self) -> str:
        return "openai (text-embedding-3-small)"


def get_provider(
    provider: str = "local",
    api_key: str | None = None,
) -> EmbeddingProvider:
    """
    Factory function to get the appropriate embedding provider.

    Raises:
        ValueError: If provider is unknown or openai without api_key.
    """
    if provider == "local":
        return LocalEmbeddingProvider()
    elif provider == "openai":
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set SWIFTBITS_OPENAI_KEY or use --api-key"
            )
        return OpenAIEmbeddingProvider(api_key)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
