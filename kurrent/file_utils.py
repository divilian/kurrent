# Generic kurrent-agnostic functions for file control.

import hashlib
from pathlib import Path

def is_pdf(path: str | Path) -> bool:
    path = normalize_path(path)
    if not path.is_file():
        return False
    with path.open("rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def sha256_file(path: str | Path) -> str:
    path = normalize_path(path)

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
