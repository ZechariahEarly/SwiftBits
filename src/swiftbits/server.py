"""SwiftBits MCP server — exposes document search over Model Context Protocol."""

import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from swiftbits.embeddings import get_provider
from swiftbits.store import VectorStore


def _log(msg: str) -> None:
    """Log to stderr (stdout is reserved for MCP protocol)."""
    print(msg, file=sys.stderr)


def create_server(
    collection_name: str = "default",
    data_dir: str | None = None,
) -> Server:
    """Create and configure the MCP server for a given collection."""
    store = VectorStore(data_dir=data_dir)

    # Validate collection exists
    meta = store.get_collection_metadata(collection_name)
    if meta is None:
        raise ValueError(f"Collection '{collection_name}' does not exist")

    # Set up embedding provider for query-time embedding
    provider_name = meta.get("embedding_provider", "local")
    api_key = os.environ.get("SWIFTBITS_OPENAI_KEY")

    provider = get_provider(provider_name, api_key)

    server = Server("swiftbits")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_documents",
                description=(
                    "Search through the user's local document collection using semantic similarity. "
                    "Returns the most relevant text passages from documents the user has previously indexed. "
                    "Use this when the user asks questions that might be answered by their personal documents."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific and descriptive for best results.",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5, max: 20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_indexed_documents",
                description=(
                    "List all documents that have been indexed and are available for search. "
                    "Use this to tell the user what documents are in their collection."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            Tool(
                name="get_documents",
                description=(
                    "Retrieve the full text of one or more indexed documents by name. "
                    "Use list_indexed_documents first to see available document names. "
                    "Useful when the user wants you to read, analyze, or compare their documents."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document filenames to retrieve (e.g. [\"paper1.pdf\", \"paper2.pdf\"])",
                        },
                    },
                    "required": ["sources"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "search_documents":
            return _handle_search(arguments, store, provider, collection_name)
        elif name == "list_indexed_documents":
            return _handle_list(store, collection_name)
        elif name == "get_documents":
            return _handle_get_documents(arguments, store, collection_name)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def _handle_search(arguments, store, provider, collection_name):
    """Handle the search_documents tool call."""
    query = arguments.get("query", "")
    n_results = min(arguments.get("n_results", 5), 20)

    try:
        query_embedding = provider.embed([query])[0]
        results = store.query(collection_name, query_embedding, n_results=n_results)
    except Exception as e:
        return [TextContent(type="text", text=f"Search error: {e}")]

    if not results:
        return [TextContent(type="text", text="No results found.")]

    # Count unique sources
    sources = set(r["metadata"].get("source", "") for r in results)
    lines = [f"Found {len(results)} results across {len(sources)} documents:\n"]

    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        source = meta.get("source", "unknown")
        page_numbers = meta.get("page_numbers", "")
        relevance = round(1 - result["distance"], 2)

        # Format page numbers
        if page_numbers:
            pages = page_numbers.split(",")
            if len(pages) == 1:
                page_str = f"page {pages[0]}"
            else:
                page_str = f"pages {', '.join(pages)}"
        else:
            page_str = ""

        header = f"[Result {i}] {source}"
        if page_str:
            header += f" ({page_str})"
        header += f" | Relevance: {relevance}"

        lines.append(f"---\n{header}\n{result['text']}")

    lines.append("---")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_list(store, collection_name):
    """Handle the list_indexed_documents tool call."""
    try:
        docs = store.list_documents(collection_name)
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

    if not docs:
        return [TextContent(
            type="text",
            text=f"No documents indexed in collection '{collection_name}'.",
        )]

    lines = [f"Indexed documents in collection '{collection_name}':\n"]
    total_chunks = 0
    for i, doc in enumerate(docs, 1):
        vectorized = doc.get("vectorized_at", "unknown")
        if "T" in vectorized:
            vectorized = vectorized.split("T")[0]
        lines.append(
            f"{i}. {doc['source']} — {doc['chunk_count']} chunks (indexed {vectorized})"
        )
        total_chunks += doc["chunk_count"]

    lines.append(f"\nTotal: {len(docs)} documents, {total_chunks} chunks")
    return [TextContent(type="text", text="\n".join(lines))]


def _handle_get_documents(arguments, store, collection_name):
    """Handle the get_documents tool call."""
    sources = arguments.get("sources", [])
    if not sources:
        return [TextContent(type="text", text="No document sources provided.")]

    sections = []
    for source in sources:
        try:
            chunks = store.get_document_chunks(collection_name, source)
        except ValueError as e:
            sections.append(f"=== {source} ===\nError: {e}")
            continue

        total_chunks = len(chunks)
        # Collect all unique page numbers across chunks
        all_pages = set()
        for chunk in chunks:
            page_str = chunk["metadata"].get("page_numbers", "")
            if page_str:
                for p in page_str.split(","):
                    all_pages.add(int(p))

        header = f"=== {source} ({total_chunks} chunks, {len(all_pages)} pages) ==="
        text = "\n".join(chunk["text"] for chunk in chunks)
        sections.append(f"{header}\n{text}")

    return [TextContent(type="text", text="\n\n".join(sections))]


async def run_stdio(server: Server) -> None:
    """Run the MCP server over stdio (for Claude Desktop)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
