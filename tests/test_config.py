import json
from pathlib import Path

import pytest

from swiftbits.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_DATA_DIR,
    ensure_data_dirs,
    get_chroma_dir,
    get_config_value,
    get_data_dir,
    load_config,
    save_config,
    set_config_value,
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


# --- Persistent config tests ---

@pytest.fixture()
def _tmp_config(tmp_path, monkeypatch):
    """Redirect config path to a temp directory."""
    monkeypatch.setattr("swiftbits.config.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("swiftbits.config.get_config_path", lambda: tmp_path / "config.json")


@pytest.mark.usefixtures("_tmp_config")
class TestLoadConfig:
    def test_missing_file(self):
        assert load_config() == {}

    def test_malformed_json(self, tmp_path):
        (tmp_path / "config.json").write_text("not json{{{")
        assert load_config() == {}

    def test_valid_config(self, tmp_path):
        data = {"default_provider": "openai"}
        (tmp_path / "config.json").write_text(json.dumps(data))
        assert load_config() == data


@pytest.mark.usefixtures("_tmp_config")
class TestSaveConfig:
    def test_roundtrip(self):
        data = {"default_provider": "voyage", "api_keys": {"voyage": "pa-test"}}
        save_config(data)
        assert load_config() == data

    def test_creates_file(self, tmp_path):
        save_config({"default_provider": "local"})
        assert (tmp_path / "config.json").exists()


@pytest.mark.usefixtures("_tmp_config")
class TestGetConfigValue:
    def test_top_level_key(self):
        save_config({"default_provider": "openai"})
        assert get_config_value("default_provider") == "openai"

    def test_nested_key(self):
        save_config({"api_keys": {"openai": "sk-abc"}})
        assert get_config_value("api_keys.openai") == "sk-abc"

    def test_missing_key(self):
        assert get_config_value("default_provider") is None

    def test_missing_nested_key(self):
        save_config({"api_keys": {}})
        assert get_config_value("api_keys.openai") is None


@pytest.mark.usefixtures("_tmp_config")
class TestSetConfigValue:
    def test_set_provider(self):
        set_config_value("default_provider", "openai")
        assert get_config_value("default_provider") == "openai"

    def test_set_api_key(self):
        set_config_value("api_keys.openai", "sk-test123")
        assert get_config_value("api_keys.openai") == "sk-test123"

    def test_invalid_key(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            set_config_value("invalid_key", "value")

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Invalid provider"):
            set_config_value("default_provider", "badname")

    def test_overwrites_existing(self):
        set_config_value("default_provider", "openai")
        set_config_value("default_provider", "voyage")
        assert get_config_value("default_provider") == "voyage"

    def test_preserves_other_keys(self):
        set_config_value("default_provider", "openai")
        set_config_value("api_keys.openai", "sk-test")
        assert get_config_value("default_provider") == "openai"
        assert get_config_value("api_keys.openai") == "sk-test"
