# kurrent/config.py

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


KURRENT_DB_FILENAME = "kurrent.db"
KURRENT_CHROMA_DIRNAME = "chroma"
KURRENT_PDFS_DIRNAME = "pdfs"
CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class KurrentStatePaths:
    """Filesystem paths for one kurrent state directory."""

    state_dir: Path
    sqlite_path: Path
    chroma_path: Path
    pdfs_path: Path


def get_crossref_mailto() -> str | None:
    """Return the configured Crossref mailto address, if any."""

    load_dotenv()
    return os.environ.get("KURRENT_CROSSREF_MAILTO")


def get_default_kurrent_state_dir() -> Path:
    """Return the default kurrent state directory from .env.

    Expected .env entry:

        KURRENT_STATE_DIR=/path/to/kurrent/state
    """

    load_dotenv()
    raw_path = os.environ.get("KURRENT_STATE_DIR")

    if raw_path is None or not raw_path.strip():
        raise RuntimeError(
            "No default kurrent state directory configured. Set "
            "KURRENT_STATE_DIR in your .env file, or pass --state-dir."
        )

    return Path(raw_path).expanduser().resolve()


def get_kurrent_state_paths(
    state_dir: str | Path | None = None,
) -> KurrentStatePaths:
    """Return SQLite, Chroma, and managed-PDF paths for a state directory."""

    if state_dir is None:
        resolved_state_dir = get_default_kurrent_state_dir()
    else:
        resolved_state_dir = Path(state_dir).expanduser().resolve()

    return KurrentStatePaths(
        state_dir=resolved_state_dir,
        sqlite_path=resolved_state_dir / KURRENT_DB_FILENAME,
        chroma_path=resolved_state_dir / KURRENT_CHROMA_DIRNAME,
        pdfs_path=resolved_state_dir / KURRENT_PDFS_DIRNAME,
    )
