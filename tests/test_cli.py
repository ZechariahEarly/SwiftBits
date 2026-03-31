"""Tests for the CLI module."""

from unittest.mock import MagicMock, patch

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
