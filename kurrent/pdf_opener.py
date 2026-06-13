"""Best-effort helpers for opening PDFs in a system viewer.

These helpers keep OS-specific viewer-launching code out of CLI workflows.
They intentionally do not raise for normal user-environment failures such as a
missing viewer, a headless shell, or a missing file. Callers can inspect the
returned OpenPdfResult and decide what to print.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
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
    process: Any | None = None


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
    prefer_managed_process: bool = False,
) -> tuple[tuple[str, ...] | None, bool]:
    """Return the command and whether it should honor page navigation."""

    if sys.platform == "win32":
        return None, False

    if sys.platform == "darwin":
        return ("open", str(path)), False

    if shutil.which("okular") is not None:
        if page is not None:
            return ("okular", "--page", str(page), str(path)), True

        if prefer_managed_process:
            return ("okular", str(path)), False

    return ("xdg-open", str(path)), False


def open_pdf(
    path: str | Path,
    page: int | None = None,
    *,
    prefer_managed_process: bool = False,
) -> OpenPdfResult:
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

    command, page_supported = _open_command_for_platform(
        pdf_path,
        page,
        prefer_managed_process=prefer_managed_process,
    )

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

        process = subprocess.Popen(
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
            process=process,
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


def close_open_pdf(result: OpenPdfResult | None) -> bool:
    """Best-effort close for a PDF viewer process Kurrent launched directly.

    This can only close viewers when the platform opener leaves Kurrent with a
    live child process. Desktop helpers such as xdg-open may hand off the PDF to
    another application and exit immediately; in those cases there is nothing
    reliable for Kurrent to close.
    """

    if result is None:
        return False

    process = getattr(result, "process", None)

    if process is None:
        return False

    try:
        if process.poll() is not None:
            return False

        process.terminate()
        return True
    except OSError:
        return False
