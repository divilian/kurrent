# kurrent/config.py

from __future__ import annotations

import os

from dotenv import load_dotenv


def get_crossref_mailto() -> str | None:
    """Return the configured Crossref mailto address, if any."""

    load_dotenv()
    return os.environ.get("KURRENT_CROSSREF_MAILTO")
