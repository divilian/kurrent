"""Shared helpers for kurrent playground scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shutil

from tqdm import tqdm

from kurrent.file_utils import is_pdf
from kurrent.terminal import QUIT_COMMANDS


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
PLAYGROUND_BASE_DIR = Path("/tmp/kurrent-playgrounds")


class TqdmProgress:
    """Tiny wrapper for tqdm progress callbacks used by playgrounds."""

    def __init__(
        self,
        desc: str,
        unit: str,
    ) -> None:
        self.desc = desc
        self.unit = unit
        self.progress_bar = None

    def start(self, total: int) -> None:
        """Start a new progress bar with the given total."""

        self.close()

        if total <= 0:
            print("No heading candidates will be sent to Ollama.")
            return

        self.progress_bar = tqdm(
            total=total,
            desc=self.desc,
            unit=self.unit,
        )

    def update(self, completed: int) -> None:
        """Advance the progress bar."""

        if self.progress_bar is not None:
            self.progress_bar.update(completed)

    def close(self) -> None:
        """Close the progress bar if it is active."""

        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


def playground_dir(name: str) -> Path:
    """Return the temporary state directory for one playground."""

    return PLAYGROUND_BASE_DIR / name


def discover_pdfs(path: str | Path) -> list[Path]:
    """Return one PDF path or all PDFs recursively under a directory."""

    path = Path(path).expanduser().resolve()

    if path.is_file():
        if not is_pdf(path):
            raise ValueError(f"Not a PDF file: {path}")

        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )


def print_pdf_list(pdf_paths: Sequence[Path]) -> None:
    """Print a numbered list of PDF basenames."""

    if not pdf_paths:
        print("No PDFs found.")
        return

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"{i}. {pdf_path.name}")


def existing_playground_paths(
    db_path: Path,
    chroma_path: Path | None = None,
) -> list[Path]:
    """Return existing SQLite sidecars and optional Chroma path."""

    candidates = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ]

    if chroma_path is not None:
        candidates.append(chroma_path)

    return [path for path in candidates if path.exists()]


def delete_playground_paths(paths: Sequence[Path]) -> None:
    """Delete playground files/directories."""

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def prepare_fresh_playground_state(
    db_path: Path,
    chroma_path: Path | None = None,
    label: str = "playground state",
) -> None:
    """Delete existing playground state after confirmation."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    if not existing_paths:
        return

    print()
    print(f"Existing {label} found.")
    print(f"This playground is intended to start with fresh {label} each run.")
    print()

    if chroma_path is None:
        print("Files to delete:")
    else:
        print("Files/directories to delete:")

    for path in existing_paths:
        print(f"  {path}")

    print()

    try:
        response = input(f"Delete existing {label}? [Y/n] ")
    except EOFError:
        raise SystemExit(
            f"Existing {label} was not deleted; aborting."
        )

    response = response.strip().lower()

    if response not in {"", "y", "yes"}:
        raise SystemExit(f"Cancelled; existing {label} left in place.")

    delete_playground_paths(existing_paths)

    print(f"Deleted existing {label}.")


def cleanup_playground_state(
    db_path: Path,
    chroma_path: Path | None = None,
    label: str = "playground state",
) -> None:
    """Delete playground state on normal program exit."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    delete_playground_paths(existing_paths)

    if existing_paths:
        print()
        print(f"Deleted {label}.")
