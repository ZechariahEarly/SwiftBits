"""SwiftBits configuration — constants and path helpers."""

from pathlib import Path

DEFAULT_DATA_DIR = "~/.swiftbits"
DEFAULT_CHROMA_DIR = "~/.swiftbits/chroma"
DEFAULT_COLLECTION = "default"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


def get_data_dir() -> Path:
    """Return the expanded path to the SwiftBits data directory."""
    return Path(DEFAULT_DATA_DIR).expanduser()


def get_chroma_dir() -> Path:
    """Return the expanded path to the ChromaDB storage directory."""
    return Path(DEFAULT_CHROMA_DIR).expanduser()


def ensure_data_dirs() -> None:
    """Create the data and chroma directories if they don't exist."""
    get_data_dir().mkdir(parents=True, exist_ok=True)
    get_chroma_dir().mkdir(parents=True, exist_ok=True)
