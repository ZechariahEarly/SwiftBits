"""SwiftBits configuration — constants, path helpers, and persistent config."""

import json
from pathlib import Path

DEFAULT_DATA_DIR = "~/.swiftbits"
DEFAULT_CHROMA_DIR = "~/.swiftbits/chroma"
DEFAULT_COLLECTION = "default"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

VALID_PROVIDERS = {"local", "openai", "voyage"}
VALID_CONFIG_KEYS = {"default_provider", "api_keys.openai", "api_keys.voyage"}


def get_data_dir() -> Path:
    """Return the expanded path to the SwiftBits data directory."""
    return Path(DEFAULT_DATA_DIR).expanduser()


def get_chroma_dir() -> Path:
    """Return the expanded path to the ChromaDB storage directory."""
    return Path(DEFAULT_CHROMA_DIR).expanduser()


def get_config_path() -> Path:
    """Return the expanded path to the config file."""
    return get_data_dir() / "config.json"


def ensure_data_dirs() -> None:
    """Create the data and chroma directories if they don't exist."""
    get_data_dir().mkdir(parents=True, exist_ok=True)
    get_chroma_dir().mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Read and return the config JSON. Returns {} if missing or malformed."""
    path = get_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict) -> None:
    """Write the config dict to the config file."""
    ensure_data_dirs()
    get_config_path().write_text(json.dumps(config, indent=2) + "\n")


def get_config_value(key: str) -> str | None:
    """Get a config value by dot-notation key (e.g. 'api_keys.openai')."""
    config = load_config()
    parts = key.split(".")
    obj = config
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj if isinstance(obj, str) else None


def set_config_value(key: str, value: str) -> None:
    """Set a config value by dot-notation key. Validates key and value."""
    if key not in VALID_CONFIG_KEYS:
        valid = ", ".join(sorted(VALID_CONFIG_KEYS))
        raise ValueError(f"Unknown config key: {key} (valid keys: {valid})")

    if key == "default_provider" and value not in VALID_PROVIDERS:
        valid = ", ".join(sorted(VALID_PROVIDERS))
        raise ValueError(f"Invalid provider: {value} (valid: {valid})")

    config = load_config()
    parts = key.split(".")
    obj = config
    for part in parts[:-1]:
        if part not in obj or not isinstance(obj[part], dict):
            obj[part] = {}
        obj = obj[part]
    obj[parts[-1]] = value
    save_config(config)
