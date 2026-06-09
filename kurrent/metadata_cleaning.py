"""Small metadata-cleaning helpers shared across ingestion paths."""

from __future__ import annotations

import re

__all__ = [
    "clean_author_metadata_text",
    "clean_metadata_text",
    "clean_title_metadata_text",
    "title_case_if_all_caps",
]

SMALL_TITLE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "into",
    "nor",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "with",
    "without",
    "via",
    "vs",
    "v",
}

AUTHOR_CONNECTOR_WORDS = {
    "and",
    "de",
    "del",
    "der",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
}

TITLE_ACRONYMS = {
    "ABM",
    "ACM",
    "AI",
    "AIDS",
    "API",
    "CPU",
    "DNA",
    "DOI",
    "GPU",
    "HIV",
    "HTML",
    "HTTP",
    "IEEE",
    "LLM",
    "LLMS",
    "ML",
    "NLP",
    "PDF",
    "PNAS",
    "RNA",
    "SARS",
    "URL",
    "XML",
}

ROMAN_NUMERAL_RE = re.compile(r"[IVXLCDM]+")
WORD_RE = re.compile(r"([A-Za-z]+(?:'[A-Za-z]+)?)")


def clean_metadata_text(value: object) -> str | None:
    """Normalize a metadata value and turn empty values into None."""

    if value is None:
        return None

    value = " ".join(str(value).split()).strip()

    if not value:
        return None

    return value


def _looks_like_all_caps_text(
    value: str,
    min_letters: int = 4,
    uppercase_ratio: float = 1.0,
) -> bool:
    letters = [char for char in value if char.isalpha()]

    if len(letters) < min_letters:
        return False

    uppercase_count = sum(1 for char in letters if char.isupper())

    if uppercase_count == 0:
        return False

    return uppercase_count / len(letters) >= uppercase_ratio


def _looks_like_all_caps_title(value: str) -> bool:
    return _looks_like_all_caps_text(value)


def _looks_like_all_caps_author(value: str) -> bool:
    # Author strings sometimes include lower-case connectors, for example
    # "WILLIAM W. COHEN and YORAM SINGER". Treat these as shouty metadata
    # when the overwhelming majority of letters are uppercase.
    return _looks_like_all_caps_text(
        value,
        min_letters=2,
        uppercase_ratio=0.75,
    )


def _title_case_word(word: str, is_first: bool, is_last: bool) -> str:
    upper_word = word.upper()
    lower_word = word.lower()

    if upper_word in TITLE_ACRONYMS:
        return upper_word

    if ROMAN_NUMERAL_RE.fullmatch(upper_word):
        return upper_word

    if not is_first and not is_last and lower_word in SMALL_TITLE_WORDS:
        return lower_word

    # str.capitalize() gives predictable behavior for ALL-CAPS input without
    # disturbing punctuation outside the matched word.
    return lower_word.capitalize()


def _title_case_piece(piece: str, is_first: bool, is_last: bool) -> str:
    if "-" not in piece:
        return _title_case_word(piece, is_first=is_first, is_last=is_last)

    subpieces = piece.split("-")
    converted: list[str] = []

    for index, subpiece in enumerate(subpieces):
        if not subpiece:
            converted.append(subpiece)
            continue

        converted.append(
            _title_case_word(
                subpiece,
                is_first=is_first and index == 0,
                is_last=is_last and index == len(subpieces) - 1,
            )
        )

    return "-".join(converted)


def title_case_if_all_caps(value: object) -> str | None:
    """Convert obvious ALL-CAPS titles to readable title case.

    Mixed-case titles are returned unchanged. This is intentionally a metadata
    cleanup rule for titles only; it should not be used for author names.
    """

    value = clean_metadata_text(value)

    if value is None:
        return None

    if not _looks_like_all_caps_title(value):
        return value

    matches = list(WORD_RE.finditer(value))
    total = len(matches)

    if total == 0:
        return value

    pieces: list[str] = []
    cursor = 0

    for index, match in enumerate(matches):
        pieces.append(value[cursor:match.start()])
        pieces.append(
            _title_case_piece(
                match.group(0),
                is_first=index == 0,
                is_last=index == total - 1,
            )
        )
        cursor = match.end()

    pieces.append(value[cursor:])

    return "".join(pieces)


def clean_title_metadata_text(value: object) -> str | None:
    """Normalize title text, including ALL-CAPS title cleanup."""

    return title_case_if_all_caps(value)


def _name_case_word(word: str) -> str:
    upper_word = word.upper()
    lower_word = word.lower()

    if lower_word in AUTHOR_CONNECTOR_WORDS:
        return lower_word

    if upper_word in TITLE_ACRONYMS:
        return upper_word

    if ROMAN_NUMERAL_RE.fullmatch(upper_word):
        return upper_word

    # Handle common Irish-style apostrophe names reasonably for obvious
    # ALL-CAPS metadata: O'NEIL -> O'Neil, D'ANGELO -> D'Angelo.
    if "'" in lower_word:
        pieces = lower_word.split("'")
        return "'".join(
            piece[:1].upper() + piece[1:]
            if piece else piece
            for piece in pieces
        )

    # This intentionally does not try to infer Mc/Mac casing. It is a safe
    # cleanup for shouty all-caps names, not a full personal-name parser.
    return lower_word[:1].upper() + lower_word[1:]


def _name_case_piece(piece: str) -> str:
    if "-" not in piece:
        return _name_case_word(piece)

    return "-".join(
        _name_case_word(subpiece) if subpiece else subpiece
        for subpiece in piece.split("-")
    )


def name_case_if_all_caps(value: object) -> str | None:
    """Convert obvious ALL-CAPS author metadata to readable name case.

    Mixed-case author strings are returned unchanged. This handles the common
    PDF/Crossref/LLM cleanup case of names such as "WILLIAM W. COHEN and
    YORAM SINGER" without trying to be a perfect personal-name formatter.
    """

    value = clean_metadata_text(value)

    if value is None:
        return None

    if not _looks_like_all_caps_author(value):
        return value

    matches = list(WORD_RE.finditer(value))

    if not matches:
        return value

    pieces: list[str] = []
    cursor = 0

    for match in matches:
        pieces.append(value[cursor:match.start()])
        pieces.append(_name_case_piece(match.group(0)))
        cursor = match.end()

    pieces.append(value[cursor:])

    return "".join(pieces)


def clean_author_metadata_text(value: object) -> str | None:
    """Normalize author text, including obvious ALL-CAPS cleanup."""

    return name_case_if_all_caps(value)
