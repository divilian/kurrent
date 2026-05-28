"""Shared terminal interaction constants and helpers."""

QUIT_COMMANDS = {"q", "quit", "done", "exit"}


def is_quit_command(text: str) -> bool:
    """Return whether text is one of kurrent's standard quit commands."""

    return text.strip().lower() in QUIT_COMMANDS
