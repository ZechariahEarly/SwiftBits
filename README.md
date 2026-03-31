# SwiftBits

Vectorize local documents and serve them as a local MCP server for Claude Desktop. Reduce "local docs → AI-accessible knowledge" to two commands.

## Installation

From PyPI:

```bash
pip install swiftbits
```

From source (for development):

```bash
git clone https://github.com/ZechariahEarly/SwiftBits.git
cd swiftbits
pip install -e ".[dev]"
```

## Quick Start

```bash
# Vectorize a document (uses local embeddings by default — no API key needed)
swiftbits vector document.pdf
swiftbits vector notes.txt
swiftbits vector readme.md

# Start the MCP server (stdio transport for Claude Desktop)
swiftbits start

# See what's been indexed
swiftbits list
swiftbits list --collection default

# Remove a document or collection
swiftbits remove document.pdf
swiftbits remove myc --all
```

### Configuration

Set a default provider and API key so you don't need flags every time:

```bash
# Set a default embedding provider
swiftbits config set default_provider openai

# Store an API key (saved in ~/.swiftbits/config.json)
swiftbits config set api_keys.openai sk-...

# Now just run — no flags needed
swiftbits vector document.pdf

# View current config
swiftbits config show

# CLI flags always override config
swiftbits vector document.pdf --provider local
```

Note: API keys are stored in plaintext in `~/.swiftbits/config.json`.

### Using OpenAI Embeddings

```bash
# Via config (recommended)
swiftbits config set default_provider openai
swiftbits config set api_keys.openai sk-...
swiftbits vector document.pdf

# Via flag
swiftbits vector document.pdf --provider openai --api-key sk-...

# Via environment variable
export SWIFTBITS_OPENAI_KEY=sk-...
swiftbits vector document.pdf --provider openai
```

### Using Voyage AI Embeddings

```bash
# Via config (recommended)
swiftbits config set default_provider voyage
swiftbits config set api_keys.voyage pa-...
swiftbits vector document.pdf

# Via flag
swiftbits vector document.pdf --provider voyage --api-key pa-...

# Via environment variable
export SWIFTBITS_VOYAGE_KEY=pa-...
swiftbits vector document.pdf --provider voyage
```

### Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "swiftbits": {
      "command": "swiftbits",
      "args": ["start"]
    }
  }
}
```

## Architecture

```
CLI (Click) → Document Processor → Embedding Engine → Vector Store (ChromaDB)
                                                    ↗
MCP Server (stdio) ──── query ────────────────────→
```

| Module | File | Description |
|--------|------|-------------|
| Config | `src/swiftbits/config.py` | Path constants, helpers, and persistent config (`~/.swiftbits/config.json`) |
| Processor | `src/swiftbits/processor.py` | Document parsing (PDF, TXT, MD) + recursive text chunking |
| Embeddings | `src/swiftbits/embeddings.py` | Provider abstraction (local, OpenAI, Voyage AI) |
| Store | `src/swiftbits/store.py` | ChromaDB wrapper for vector storage |
| Server | `src/swiftbits/server.py` | MCP server with `search_documents`, `list_indexed_documents`, and `get_documents` tools |
| CLI | `src/swiftbits/cli.py` | Click CLI orchestrating all modules |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single module's tests
pytest tests/test_processor.py -v

# Run a specific test
pytest tests/test_processor.py::TestChunkText::test_overlap_present -v
```

## License

MIT
