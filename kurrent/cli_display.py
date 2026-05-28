"""Terminal display helpers for kurrent CLI commands."""

from __future__ import annotations

import os
import re
import shutil
import sys

ANSI_BOLD = "\033[1m"
ANSI_BOLD_YELLOW = "\033[1;33m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def ansi_enabled() -> bool:
    """Return whether ANSI formatting should be used for terminal output."""

    if sys.stdout is None:
        return False

    if not sys.stdout.isatty():
        return False

    if "NO_COLOR" in os.environ:
        return False

    if os.environ.get("TERM") == "dumb":
        return False

    return True


def terminal_width() -> int:
    """Return the current terminal width, with a conservative fallback."""

    return shutil.get_terminal_size(fallback=(79, 20)).columns


def visible_len(text: str) -> int:
    """Return display width after ignoring ANSI color codes."""

    return len(ANSI_ESCAPE_RE.sub("", text))


def wrapped_lines(
    text: str,
    indent: str = "",
    subsequent_indent: str | None = None,
    width: int | None = None,
) -> list[str]:
    """Return terminal-width-wrapped lines, counting ANSI escapes as zero."""

    if width is None:
        width = terminal_width()

    if subsequent_indent is None:
        subsequent_indent = indent

    output: list[str] = []

    for raw_line in str(text).splitlines() or [""]:
        words = raw_line.split()

        if not words:
            output.append(indent.rstrip())
            continue

        prefix = indent
        available = max(1, width - visible_len(prefix))
        current = ""
        current_len = 0

        for word in words:
            word_len = visible_len(word)

            if not current:
                current = word
                current_len = word_len
                continue

            if current_len + 1 + word_len <= available:
                current += " " + word
                current_len += 1 + word_len
                continue

            output.append(prefix + current)
            prefix = subsequent_indent
            available = max(1, width - visible_len(prefix))
            current = word
            current_len = word_len

        if current:
            output.append(prefix + current)

    return output


def print_wrapped(
    text: str,
    indent: str = "",
    subsequent_indent: str | None = None,
    width: int | None = None,
    file=None,
) -> None:
    """Print user-facing CLI prose wrapped to the terminal width."""

    for line in wrapped_lines(
        text,
        indent=indent,
        subsequent_indent=subsequent_indent,
        width=width,
    ):
        print(line, file=file)


def separator_line() -> str:
    """Return a separator line that fits the current terminal width."""

    return "-" * min(terminal_width(), 79)


def collapse_whitespace(text: str) -> str:
    """Normalize text to a single display-friendly line."""

    return " ".join(text.split())


def bold_matches(text: str, search_text: str | None) -> str:
    """Return text with literal case-insensitive matches bolded."""

    if not ansi_enabled():
        return text

    if search_text is None:
        return text

    search_text = search_text.strip()

    if not search_text:
        return text

    pattern = re.compile(re.escape(search_text), flags=re.IGNORECASE)

    return pattern.sub(
        lambda match: f"{ANSI_BOLD}{match.group(0)}{ANSI_RESET}",
        text,
    )


def context_window(
    text: str,
    search_text: str | None,
    width: int = 240,
) -> str:
    """Return a display window centered around the first literal match."""

    text = collapse_whitespace(text)

    if len(text) <= width:
        return text

    if search_text is None:
        return text[:width].rstrip() + " [...]"

    search_text = search_text.strip()

    if not search_text:
        return text[:width].rstrip() + " [...]"

    match = re.search(re.escape(search_text), text, flags=re.IGNORECASE)

    if match is None:
        return text[:width].rstrip() + " [...]"

    match_center = (match.start() + match.end()) // 2
    start = max(0, match_center - width // 2)
    end = min(len(text), start + width)
    start = max(0, end - width)

    window = text[start:end].strip()

    if start > 0:
        window = "[...] " + window

    if end < len(text):
        window = window + " [...]"

    return window


def print_field(label: str, value: object | None) -> None:
    """Print a wrapped label/value line for search results."""

    if value is None:
        return

    label_text = f"{label}:"
    indent = "  "
    subsequent = " " * (len(indent) + len(label_text) + 1)
    print_wrapped(
        f"{label_text} {value}",
        indent=indent,
        subsequent_indent=subsequent,
    )


def print_body(text: str, search_text: str | None = None) -> None:
    """Print wrapped body text for a result preview or detail view."""

    if search_text is not None:
        text = bold_matches(text, search_text)

    print_wrapped(text, indent="  ", subsequent_indent="  ")


def section_label(hit) -> str | None:
    """Return a compact section label for a chunk hit, if available."""

    pieces = []

    if hit.section_number is not None:
        pieces.append(str(hit.section_number))

    if hit.section_title is not None:
        pieces.append(hit.section_title)

    if not pieces:
        return None

    return " ".join(pieces)


def reference_marker(hit) -> str:
    """Return a visible marker for reference-section hits."""

    from kurrent.sectioner import is_reference_section_chunk

    if is_reference_section_chunk(hit):
        return " [REFERENCE SECTION]"

    return ""


def source_name_for_hit(hit) -> str | None:
    """Return a display-friendly source filename without exposing full paths."""

    if hit.path is None:
        return None

    return hit.path.name


def pages_label(hit) -> str | None:
    """Return a compact page-range label, if page data is available."""

    if hit.page_start is None and hit.page_end is None:
        return None

    if hit.page_start == hit.page_end:
        return f"p. {hit.page_start}"

    return f"pp. {hit.page_start}–{hit.page_end}"


def distance_label(hit) -> str | None:
    """Return a formatted semantic distance, if present."""

    if hit.distance is None:
        return None

    return f"{hit.distance:.4f}"


def highlighted_metadata_value(
    value: object | None,
    search_text: str | None,
) -> object | None:
    """Return a metadata value with exact query matches bolded for display."""

    if value is None:
        return None

    return bold_matches(str(value), search_text)
