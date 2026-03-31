from pathlib import Path

from swiftbits.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DATA_DIR,
    ensure_data_dirs,
    get_chroma_dir,
    get_data_dir,
)


def test_get_data_dir_is_expanded():
    result = get_data_dir()
    assert "~" not in str(result)
    assert result.is_absolute()


def test_chroma_dir_is_subdirectory_of_data_dir():
    data = get_data_dir()
    chroma = get_chroma_dir()
    assert str(chroma).startswith(str(data))


def test_ensure_data_dirs_creates_directories(tmp_path, monkeypatch):
    fake_data = tmp_path / ".swiftbits"
    fake_chroma = fake_data / "chroma"
    monkeypatch.setattr("swiftbits.config.DEFAULT_DATA_DIR", str(fake_data))
    monkeypatch.setattr("swiftbits.config.DEFAULT_CHROMA_DIR", str(fake_chroma))
    # Re-import won't help since functions read module-level vars at call time,
    # but our functions use Path(DEFAULT_*).expanduser() — monkeypatch the functions
    monkeypatch.setattr("swiftbits.config.get_data_dir", lambda: fake_data)
    monkeypatch.setattr("swiftbits.config.get_chroma_dir", lambda: fake_chroma)

    assert not fake_data.exists()
    ensure_data_dirs()
    assert fake_data.is_dir()
    assert fake_chroma.is_dir()


def test_ensure_data_dirs_idempotent(tmp_path, monkeypatch):
    fake_data = tmp_path / ".swiftbits"
    fake_chroma = fake_data / "chroma"
    monkeypatch.setattr("swiftbits.config.get_data_dir", lambda: fake_data)
    monkeypatch.setattr("swiftbits.config.get_chroma_dir", lambda: fake_chroma)

    ensure_data_dirs()
    ensure_data_dirs()  # should not raise
    assert fake_data.is_dir()
    assert fake_chroma.is_dir()


def test_constants():
    assert DEFAULT_DATA_DIR == "~/.swiftbits"
    assert DEFAULT_CHROMA_DIR == "~/.swiftbits/chroma"
    assert DEFAULT_COLLECTION == "default"
    assert DEFAULT_CHUNK_SIZE == 512
    assert DEFAULT_CHUNK_OVERLAP == 50
