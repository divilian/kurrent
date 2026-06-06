"""Best-effort helpers for opening PDFs in a system viewer.

These helpers keep OS-specific viewer-launching code out of CLI workflows.
They intentionally do not raise for normal user-environment failures such as a
missing viewer, a headless shell, or a missing file. Callers can inspect the
returned OpenPdfResult and decide what to print.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


@dataclass(frozen=True, slots=True)
class OpenPdfResult:
    """Result of a best-effort PDF open request."""

    path: Path
    success: bool
    page: int | None = None
    page_supported: bool = False
    command: tuple[str, ...] | None = None
    message: str | None = None


def _normalized_page(page: int | None) -> int | None:
    """Return a usable one-based PDF page number, if supplied."""

    if page is None:
        return None

    try:
        page = int(page)
    except (TypeError, ValueError):
        return None

    if page < 1:
        return None

    return page


def _open_command_for_platform(
    path: Path,
    page: int | None,
) -> tuple[tuple[str, ...] | None, bool]:
    """Return the command and whether it should honor page navigation."""

    if sys.platform == "win32":
        return None, False

    if sys.platform == "darwin":
        return ("open", str(path)), False

    if page is not None and shutil.which("okular") is not None:
        return ("okular", "--page", str(page), str(path)), True

    return ("xdg-open", str(path)), False


def open_pdf(path: str | Path, page: int | None = None) -> OpenPdfResult:
    """Best-effort open of a PDF in the user's system viewer.

    On Linux, Okular is preferred when a page number is provided because it
    supports ``--page``. Otherwise this falls back to the platform default
    opener. Failures are reported in the returned result instead of being
    raised.
    """

    pdf_path = Path(path)
    page = _normalized_page(page)

    if not pdf_path.exists():
        return OpenPdfResult(
            path=pdf_path,
            success=False,
            page=page,
            message=f"PDF path does not exist: {pdf_path}",
        )

    command, page_supported = _open_command_for_platform(pdf_path, page)

    try:
        if sys.platform == "win32":
            os.startfile(str(pdf_path))  # type: ignore[attr-defined]
            return OpenPdfResult(
                path=pdf_path,
                success=True,
                page=page,
                page_supported=False,
                command=None,
            )

        if command is None:
            return OpenPdfResult(
                path=pdf_path,
                success=False,
                page=page,
                message="No PDF opener command is available.",
            )

        subprocess.Popen(
            list(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return OpenPdfResult(
            path=pdf_path,
            success=True,
            page=page,
            page_supported=page_supported,
            command=command,
        )
    except OSError as exc:
        return OpenPdfResult(
            path=pdf_path,
            success=False,
            page=page,
            page_supported=page_supported,
            command=command,
            message=f"Could not open PDF automatically: {type(exc).__name__}: {exc}",
        )
