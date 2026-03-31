"""Tests for the CLI module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from swiftbits.cli import cli


@patch("swiftbits.cli.VectorStore")
@patch("swiftbits.cli.get_provider")
@patch("swiftbits.cli.process_document")
def _run_vector_success(mock_proc, mock_get_prov, mock_store_cls, tmp_path):
    """Helper: set up mocks for a successful vector command."""
    from swiftbits.processor import Chunk

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    mock_proc.return_value = [
        Chunk(text="chunk text", metadata={
            "source": "test.pdf", "page_numbers": [1],
            "chunk_index": 0, "total_chunks": 1, "char_count": 10,
        })
    ]
    mock_provider = MagicMock()
    mock_provider.embed.return_value = [[0.1] * 384]
    mock_provider.dimension = 384
    mock_provider.name = "local (all-MiniLM-L6-v2)"
    mock_get_prov.return_value = mock_provider
    mock_store_cls.return_value.add_document.return_value = 1

    return pdf


class TestVersion:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestVector:
    def test_file_not_found(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["vector", "/nonexistent/file.pdf"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_unsupported_type(self, tmp_path):
        f = tmp_path / "test.docx"
        f.write_text("fake")
        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(f)])
        assert result.exit_code != 0
        assert "Unsupported file type" in result.output

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_vector_success(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_proc.return_value = [
            Chunk(text="chunk text", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 10,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384]
        mock_provider.dimension = 384
        mock_provider.name = "local (all-MiniLM-L6-v2)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf)])
        assert result.exit_code == 0
        assert "Vectorized test.pdf" in result.output
        assert "Chunks:     1" in result.output

    def test_openai_no_key(self, tmp_path, monkeypatch):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.delenv("SWIFTBITS_OPENAI_KEY", raising=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf), "--provider", "openai"])
        assert result.exit_code != 0
        assert "OpenAI API key required" in result.output

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_openai_env_key(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path, monkeypatch):
        from swiftbits.processor import Chunk

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setenv("SWIFTBITS_OPENAI_KEY", "sk-test123")

        mock_proc.return_value = [
            Chunk(text="chunk", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 5,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 1536]
        mock_provider.dimension = 1536
        mock_provider.name = "openai (text-embedding-3-small)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf), "--provider", "openai"])
        assert result.exit_code == 0
        mock_get_prov.assert_called_once_with("openai", "sk-test123")

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_vector_txt_success(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        txt = tmp_path / "notes.txt"
        txt.write_text("some text content")

        mock_proc.return_value = [
            Chunk(text="some text content", metadata={
                "source": "notes.txt", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 17,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384]
        mock_provider.dimension = 384
        mock_provider.name = "local (all-MiniLM-L6-v2)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(txt)])
        assert result.exit_code == 0
        assert "Vectorized notes.txt" in result.output

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_vector_md_success(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        md = tmp_path / "readme.md"
        md.write_text("# Hello\n\nWorld")

        mock_proc.return_value = [
            Chunk(text="# Hello\n\nWorld", metadata={
                "source": "readme.md", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 14,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384]
        mock_provider.dimension = 384
        mock_provider.name = "local (all-MiniLM-L6-v2)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(md)])
        assert result.exit_code == 0
        assert "Vectorized readme.md" in result.output

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_verbose_flag(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_proc.return_value = [
            Chunk(text="chunk", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 5,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384]
        mock_provider.dimension = 384
        mock_provider.name = "local (all-MiniLM-L6-v2)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "vector", str(pdf)])
        assert result.exit_code == 0
        assert "Processing" in result.output


class TestStart:
    @patch("swiftbits.cli.run_stdio")
    @patch("swiftbits.cli.create_server")
    def test_start_collection_not_found(self, mock_create, mock_run):
        mock_create.side_effect = ValueError("Collection 'default' does not exist")
        runner = CliRunner()
        result = runner.invoke(cli, ["start"])
        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestList:
    @patch("swiftbits.cli.VectorStore")
    def test_list_collections(self, mock_store_cls):
        mock_store_cls.return_value.list_collections.return_value = [
            {"name": "default", "document_count": 2, "chunk_count": 50,
             "embedding_provider": "local", "created_at": "2025-03-19"},
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "default" in result.output
        assert "2 documents" in result.output

    @patch("swiftbits.cli.VectorStore")
    def test_list_documents_in_collection(self, mock_store_cls):
        mock_store_cls.return_value.list_documents.return_value = [
            {"source": "test.pdf", "chunk_count": 10, "vectorized_at": "2025-03-19T10:00:00Z"},
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--collection", "default"])
        assert result.exit_code == 0
        assert "test.pdf" in result.output
        assert "10 chunks" in result.output


class TestRemove:
    @patch("swiftbits.cli.VectorStore")
    def test_remove_document(self, mock_store_cls):
        mock_store_cls.return_value.remove_document.return_value = 5
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "test.pdf"])
        assert result.exit_code == 0
        assert "Removed test.pdf" in result.output
        assert "5 chunks" in result.output

    @patch("swiftbits.cli.VectorStore")
    def test_remove_collection(self, mock_store_cls):
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "myc", "--all"], input="y\n")
        assert result.exit_code == 0
        assert "Removed collection" in result.output

    @patch("swiftbits.cli.VectorStore")
    def test_remove_not_found(self, mock_store_cls):
        mock_store_cls.return_value.remove_document.side_effect = ValueError(
            "'missing.pdf' not found in collection 'default'"
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["remove", "missing.pdf"])
        assert result.exit_code != 0
        assert "not found" in result.output


@pytest.fixture()
def _tmp_config(tmp_path, monkeypatch):
    """Redirect config to a temp directory for CLI tests."""
    monkeypatch.setattr("swiftbits.config.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("swiftbits.config.get_config_path", lambda: tmp_path / "config.json")
    # Also patch the cli module's imported references
    monkeypatch.setattr("swiftbits.cli.load_config", lambda: _load_json(tmp_path / "config.json"))


def _load_json(path):
    """Helper to load JSON from a path, returning {} on missing/bad file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


@pytest.mark.usefixtures("_tmp_config")
class TestConfig:
    def test_config_set_provider(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "default_provider", "openai"])
        assert result.exit_code == 0
        assert "Set default_provider = openai" in result.output

    def test_config_set_api_key_masked(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "api_keys.openai", "sk-test1234567890"])
        assert result.exit_code == 0
        assert "sk-t" in result.output
        assert "7890" in result.output
        assert "sk-test1234567890" not in result.output

    def test_config_get(self):
        runner = CliRunner()
        runner.invoke(cli, ["config", "set", "default_provider", "voyage"])
        result = runner.invoke(cli, ["config", "get", "default_provider"])
        assert result.exit_code == 0
        assert "voyage" in result.output

    def test_config_get_not_set(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "get", "default_provider"])
        assert result.exit_code == 0
        assert "not set" in result.output

    def test_config_show(self):
        runner = CliRunner()
        runner.invoke(cli, ["config", "set", "default_provider", "openai"])
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "default_provider: openai" in result.output

    def test_config_show_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "No configuration set" in result.output

    def test_config_set_invalid_key(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "bad_key", "value"])
        assert result.exit_code != 0
        assert "Unknown config key" in result.output

    def test_config_set_invalid_provider(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "set", "default_provider", "badname"])
        assert result.exit_code != 0
        assert "Invalid provider" in result.output

    def test_config_get_invalid_key(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "get", "bad_key"])
        assert result.exit_code != 0
        assert "Unknown config key" in result.output


@pytest.mark.usefixtures("_tmp_config")
class TestVectorWithConfig:
    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_uses_config_provider(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        # Set config default to openai with API key
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "default_provider": "openai",
            "api_keys": {"openai": "sk-fromconfig"},
        }))

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_proc.return_value = [
            Chunk(text="chunk", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 5,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 1536]
        mock_provider.dimension = 1536
        mock_provider.name = "openai (text-embedding-3-small)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf)])
        assert result.exit_code == 0
        mock_get_prov.assert_called_once_with("openai", "sk-fromconfig")

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_flag_overrides_config(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path):
        from swiftbits.processor import Chunk

        # Config says openai, but we pass --provider local
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"default_provider": "openai"}))

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_proc.return_value = [
            Chunk(text="chunk", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 5,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384]
        mock_provider.dimension = 384
        mock_provider.name = "local (all-MiniLM-L6-v2)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf), "--provider", "local"])
        assert result.exit_code == 0
        mock_get_prov.assert_called_once_with("local", None)

    @patch("swiftbits.cli.ensure_data_dirs")
    @patch("swiftbits.cli.VectorStore")
    @patch("swiftbits.cli.get_provider")
    @patch("swiftbits.cli.process_document")
    def test_config_api_key_used(self, mock_proc, mock_get_prov, mock_store_cls, mock_dirs, tmp_path, monkeypatch):
        from swiftbits.processor import Chunk

        monkeypatch.delenv("SWIFTBITS_VOYAGE_KEY", raising=False)

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "default_provider": "voyage",
            "api_keys": {"voyage": "pa-fromconfig"},
        }))

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_proc.return_value = [
            Chunk(text="chunk", metadata={
                "source": "test.pdf", "page_numbers": [1],
                "chunk_index": 0, "total_chunks": 1, "char_count": 5,
            })
        ]
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 512]
        mock_provider.dimension = 512
        mock_provider.name = "voyage (voyage-3-lite)"
        mock_get_prov.return_value = mock_provider
        mock_store_cls.return_value.add_document.return_value = 1

        runner = CliRunner()
        result = runner.invoke(cli, ["vector", str(pdf)])
        assert result.exit_code == 0
        mock_get_prov.assert_called_once_with("voyage", "pa-fromconfig")
