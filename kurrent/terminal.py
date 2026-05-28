# kurrent/terminal.py

QUIT_COMMANDS = {"q", "quit", "done", "exit"}


def is_quit_command(text: str) -> bool:
    """Return whether user input is a standard kurrent quit command."""

    return text.strip().lower() in QUIT_COMMANDS
