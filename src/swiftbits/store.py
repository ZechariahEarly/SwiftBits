"""SwiftBits vector store — ChromaDB wrapper for document vector storage."""

from datetime import datetime, timezone

import chromadb

from swiftbits.processor import Chunk


class VectorStore:
    """Wrapper around ChromaDB for document vector storage."""

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            from swiftbits.config import get_chroma_dir

            data_dir = str(get_chroma_dir())

        self._client = chromadb.PersistentClient(path=str(data_dir))

    def add_document(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedding_provider: str,
        embedding_dimension: int,
    ) -> int:
        """Add a document's chunks and embeddings to a collection."""
        if not chunks:
            return 0

        now = datetime.now(timezone.utc).isoformat()

        # Check for existing collection and enforce embedding consistency
        existing = self.get_collection_metadata(collection_name)
        if existing is not None:
            if existing["embedding_provider"] != embedding_provider:
                raise ValueError(
                    f"Collection '{collection_name}' uses {existing['embedding_provider']} "
                    f"embeddings (dim={existing['embedding_dimension']}). "
                    f"Cannot add {embedding_provider} embeddings (dim={embedding_dimension})."
                )
            if existing["embedding_dimension"] != embedding_dimension:
                raise ValueError(
                    f"Collection '{collection_name}' uses dimension {existing['embedding_dimension']}. "
                    f"Cannot add embeddings with dimension {embedding_dimension}."
                )

        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_provider": embedding_provider,
                "embedding_dimension": embedding_dimension,
                "created_at": now,
            },
        )

        # Handle re-vectorization: delete existing chunks for this source
        source = chunks[0].metadata["source"]
        existing_ids = collection.get(where={"source": source})["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        # Build IDs and metadata for ChromaDB
        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            chunk_id = f"{chunk.metadata['source']}::{chunk.metadata['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "source": chunk.metadata["source"],
                "page_numbers": ",".join(str(p) for p in chunk.metadata["page_numbers"]),
                "chunk_index": chunk.metadata["chunk_index"],
                "total_chunks": chunk.metadata["total_chunks"],
                "char_count": chunk.metadata["char_count"],
                "vectorized_at": now,
            })

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Query a collection with an embedding vector."""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        count = collection.count()
        if count == 0:
            return []

        # Don't request more results than available
        actual_n = min(n_results, count)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return formatted

    def list_collections(self) -> list[dict]:
        """List all collections with summary info."""
        collections = self._client.list_collections()
        result = []
        for col in collections:
            collection = self._client.get_collection(name=col.name)
            meta = collection.metadata or {}
            # Count unique sources
            all_meta = collection.get(include=["metadatas"])
            sources = set()
            for m in all_meta["metadatas"]:
                if m and "source" in m:
                    sources.add(m["source"])

            result.append({
                "name": col.name,
                "document_count": len(sources),
                "chunk_count": collection.count(),
                "embedding_provider": meta.get("embedding_provider", "unknown"),
                "created_at": meta.get("created_at", "unknown"),
            })

        return result

    def list_documents(self, collection_name: str) -> list[dict]:
        """List all documents in a collection."""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        all_data = collection.get(include=["metadatas"])
        docs = {}
        for meta in all_data["metadatas"]:
            if not meta:
                continue
            source = meta["source"]
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "chunk_count": 0,
                    "vectorized_at": meta.get("vectorized_at", "unknown"),
                }
            docs[source]["chunk_count"] += 1

        return list(docs.values())

    def remove_document(self, collection_name: str, source: str) -> int:
        """Remove all chunks belonging to a document from a collection."""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        existing_ids = collection.get(where={"source": source})["ids"]
        if not existing_ids:
            raise ValueError(
                f"'{source}' not found in collection '{collection_name}'"
            )

        collection.delete(ids=existing_ids)
        return len(existing_ids)

    def remove_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""
        try:
            self._client.delete_collection(name=collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist")

    def get_document_chunks(self, collection_name: str, source: str) -> list[dict]:
        """Retrieve all chunks for a document, ordered by chunk_index."""
        try:
            collection = self._client.get_collection(name=collection_name)
        except Exception:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        results = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            raise ValueError(
                f"'{source}' not found in collection '{collection_name}'"
            )

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        chunks.sort(key=lambda c: c["metadata"]["chunk_index"])
        return chunks

    def get_collection_metadata(self, collection_name: str) -> dict | None:
        """Get metadata for a collection."""
        try:
            collection = self._client.get_collection(name=collection_name)
            return collection.metadata
        except Exception:
            return None
