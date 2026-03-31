"""SwiftBits CLI — vectorize local docs and serve them as an MCP server."""

import asyncio
import os
import sys

import click

from swiftbits import __version__
from swiftbits.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION,
    VALID_CONFIG_KEYS,
    VALID_PROVIDERS,
    ensure_data_dirs,
    get_config_value,
    load_config,
    set_config_value,
)
from swiftbits.embeddings import get_provider
from swiftbits.processor import PARSERS, process_document
from swiftbits.server import create_server, run_stdio
from swiftbits.store import VectorStore


@click.group()
@click.version_option(version=__version__, prog_name="swiftbits")
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging")
@click.pass_context
def cli(ctx, verbose):
    """SwiftBits — vectorize local docs and serve them as an MCP server."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("file_path", type=click.Path())
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection name")
@click.option("--provider", default=None, type=click.Choice(["local", "openai", "voyage"]), help="Embedding provider (default: from config or 'local')")
@click.option("--api-key", default=None, help="API key for remote provider")
@click.option("--chunk-size", default=DEFAULT_CHUNK_SIZE, type=int, help="Target chunk size")
@click.option("--chunk-overlap", default=DEFAULT_CHUNK_OVERLAP, type=int, help="Overlap between chunks")
@click.pass_context
def vector(ctx, file_path, collection, provider, api_key, chunk_size, chunk_overlap):
    """Vectorize a document and store it in the collection."""
    verbose = ctx.obj.get("verbose", False)

    # Resolve provider from config if not explicitly passed
    cfg = load_config()
    if provider is None:
        provider = cfg.get("default_provider", "local")
        if provider not in VALID_PROVIDERS:
            click.secho(f"Error: Invalid default provider in config: {provider}", fg="red", err=True)
            raise SystemExit(1)

    # Validate file exists
    if not os.path.exists(file_path):
        click.secho(f"Error: File not found: {file_path}", fg="red", err=True)
        raise SystemExit(1)

    # Validate file type
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in PARSERS:
        supported = ", ".join(sorted(PARSERS.keys()))
        click.secho(f"Error: Unsupported file type: {ext} (supported: {supported})", fg="red", err=True)
        raise SystemExit(1)

    # Resolve API key for OpenAI
    if provider == "openai":
        if not api_key:
            api_key = cfg.get("api_keys", {}).get("openai")
        if not api_key:
            api_key = os.environ.get("SWIFTBITS_OPENAI_KEY")
        if not api_key:
            click.secho(
                "Error: OpenAI API key required. Use --api-key, swiftbits config set api_keys.openai, or set SWIFTBITS_OPENAI_KEY",
                fg="red",
                err=True,
            )
            raise SystemExit(1)

    # Resolve API key for Voyage
    if provider == "voyage":
        if not api_key:
            api_key = cfg.get("api_keys", {}).get("voyage")
        if not api_key:
            api_key = os.environ.get("SWIFTBITS_VOYAGE_KEY")
        if not api_key:
            click.secho(
                "Error: Voyage AI API key required. Use --api-key, swiftbits config set api_keys.voyage, or set SWIFTBITS_VOYAGE_KEY",
                fg="red",
                err=True,
            )
            raise SystemExit(1)

    # Process document
    try:
        if verbose:
            click.echo(f"Processing {file_path}...")
        chunks = process_document(file_path, chunk_size, chunk_overlap)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    # Embed chunks
    try:
        embed_provider = get_provider(provider, api_key)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    if verbose:
        click.echo(f"Embedding {len(chunks)} chunks with {embed_provider.name}...")

    texts = [c.text for c in chunks]
    with click.progressbar(length=1, label="Embedding") as bar:
        embeddings = embed_provider.embed(texts)
        bar.update(1)

    # Store in vector DB
    ensure_data_dirs()
    store = VectorStore()

    try:
        added = store.add_document(
            collection, chunks, embeddings, provider, embed_provider.dimension
        )
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    filename = os.path.basename(file_path)
    click.secho("✓ ", fg="green", nl=False)
    click.echo(f"Vectorized {filename}")
    click.echo(f"  Chunks:     {added}")
    click.echo(f"  Collection: {collection}")
    click.echo(f"  Provider:   {embed_provider.name}")


@cli.command()
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection to serve")
@click.option("--port", default=None, type=int, help="Run as HTTP/SSE server on this port")
@click.pass_context
def start(ctx, collection, port):
    """Start the MCP server."""
    try:
        server = create_server(collection)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    click.echo(f"SwiftBits MCP server running (stdio)", err=True)
    click.echo(f"Collection: {collection}", err=True)

    asyncio.run(run_stdio(server))


@cli.command("list")
@click.option("--collection", default=None, help="List documents in this collection")
@click.pass_context
def list_cmd(ctx, collection):
    """List collections or documents."""
    store = VectorStore()

    if collection:
        try:
            docs = store.list_documents(collection)
        except ValueError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            raise SystemExit(1)

        click.echo(f"Collection: {collection}")
        if not docs:
            click.echo("  (empty)")
        else:
            for doc in docs:
                vectorized = doc.get("vectorized_at", "unknown")
                if "T" in vectorized:
                    vectorized = vectorized.split("T")[0]
                click.echo(
                    f"  {doc['source']:<20} — {doc['chunk_count']} chunks (vectorized {vectorized})"
                )
    else:
        collections = store.list_collections()
        if not collections:
            click.echo("No collections found.")
        else:
            click.echo("Collections:")
            for col in collections:
                click.echo(
                    f"  {col['name']:<15} — {col['document_count']} documents, {col['chunk_count']} chunks"
                )


@cli.command()
@click.argument("identifier")
@click.option("--collection", default=DEFAULT_COLLECTION, help="Collection to remove from")
@click.option("--all", "remove_all", is_flag=True, help="Remove entire collection")
@click.pass_context
def remove(ctx, identifier, collection, remove_all):
    """Remove a document or collection."""
    store = VectorStore()

    if remove_all:
        if not click.confirm(f"Delete entire collection '{identifier}'?"):
            click.echo("Cancelled.")
            return
        try:
            store.remove_collection(identifier)
        except ValueError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            raise SystemExit(1)
        click.secho("✓ ", fg="green", nl=False)
        click.echo(f"Removed collection '{identifier}'")
    else:
        try:
            removed = store.remove_document(collection, identifier)
        except ValueError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            raise SystemExit(1)
        click.secho("✓ ", fg="green", nl=False)
        click.echo(f"Removed {identifier} from collection '{collection}'")
        click.echo(f"  Removed {removed} chunks")


def _mask_key(value: str) -> str:
    """Mask an API key for display, showing first 4 and last 4 characters."""
    if len(value) <= 12:
        return value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
    return value[:4] + "..." + value[-4:]


@cli.group()
def config():
    """Manage SwiftBits configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value (e.g. default_provider, api_keys.openai)."""
    try:
        set_config_value(key, value)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)

    display = _mask_key(value) if key.startswith("api_keys.") else value
    click.secho("✓ ", fg="green", nl=False)
    click.echo(f"Set {key} = {display}")


@config.command("get")
@click.argument("key")
def config_get(key):
    """Get a configuration value."""
    if key not in VALID_CONFIG_KEYS:
        valid = ", ".join(sorted(VALID_CONFIG_KEYS))
        click.secho(f"Error: Unknown config key: {key} (valid keys: {valid})", fg="red", err=True)
        raise SystemExit(1)

    value = get_config_value(key)
    if value is None:
        click.echo(f"{key}: (not set)")
    else:
        display = _mask_key(value) if key.startswith("api_keys.") else value
        click.echo(f"{key}: {display}")


@config.command("show")
def config_show():
    """Show all configuration settings."""
    cfg = load_config()
    if not cfg:
        click.echo("No configuration set. Use 'swiftbits config set <key> <value>' to configure.")
        return

    if "default_provider" in cfg:
        click.echo(f"  default_provider: {cfg['default_provider']}")

    api_keys = cfg.get("api_keys", {})
    for provider_name, key_value in api_keys.items():
        click.echo(f"  api_keys.{provider_name}: {_mask_key(key_value)}")
