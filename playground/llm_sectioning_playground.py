"""Manual playground for LLM-assisted section recognition.

Run from the project root with:

    python playground/llm_sectioning_playground.py /path/to/pdf/or/root

Or specify an Ollama model:

    python playground/llm_sectioning_playground.py /path/to/pdfs \
        --model llama3.1:8b

The playground does not write to kurrent state. It extracts heading
candidates, asks Ollama to choose the real section headings, then shows the
resulting section spans and in-memory chunks.

For diagnostics, the "Heading candidates" section prints the exact JSON
candidate payload sent to Ollama.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import textwrap

from kurrent.chunker import make_section_aware_fixed_size_chunks
from kurrent.file_utils import is_pdf
from kurrent.llm_sectioner import (
    candidate_to_prompt_dict,
    filtered_candidates,
    select_section_headings_with_ollama,
)
from kurrent.schema import SectionSpan
from kurrent.sectioner import (
    HeadingCandidate,
    detect_heading_candidates_with_context,
    is_reference_section_chunk,
    make_section_spans_from_llm_decisions,
)


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
QUIT_COMMANDS = {":q", ":quit", "done", "quit", "exit"}


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


def candidate_payloads(
    candidates: Sequence[HeadingCandidate],
) -> list[dict]:
    """Return the exact candidate payload objects sent to Ollama."""

    return [
        candidate_to_prompt_dict(candidate)
        for candidate in filtered_candidates(list(candidates))
    ]


def print_candidates(candidates: Sequence[HeadingCandidate]) -> None:
    """Print the exact JSON candidate payload sent to Ollama."""

    print()
    print("Heading candidates sent to Ollama")
    print("---------------------------------")

    if not candidates:
        print("No candidates.")
        return

    print(
        json.dumps(
            candidate_payloads(candidates),
            ensure_ascii=False,
            indent=2,
        )
    )


def print_decisions(decisions) -> None:
    """Print LLM-selected headings."""

    print()
    print("LLM-selected section headings")
    print("-----------------------------")

    if not decisions:
        print("No headings selected.")
        return

    for decision in decisions:
        number = (
            f"{decision.section_number} "
            if decision.section_number is not None
            else ""
        )
        confidence = (
            f" [{decision.confidence}]"
            if decision.confidence is not None
            else ""
        )
        print(
            f"candidate {decision.candidate_id}: "
            f"{number}{decision.section_title}{confidence}"
        )


def section_label(section: SectionSpan) -> str:
    """Return a readable label for a section span."""

    pieces = []

    if section.section_number is not None:
        pieces.append(str(section.section_number))

    if section.section_title is not None:
        pieces.append(section.section_title)

    if pieces:
        return " ".join(pieces)

    if section.section_index is not None:
        return f"section index {section.section_index}"

    return "front matter / unsectioned"


def print_sections(sections: Sequence[SectionSpan]) -> None:
    """Print generated section spans."""

    print()
    print("Generated section spans")
    print("-----------------------")

    if not sections:
        print("No sections.")
        return

    for section in sections:
        text = " ".join(section.text.split())
        preview = text[:240] + (" [...]" if len(text) > 240 else "")

        print()
        print(
            f"Section {section.section_index}: {section_label(section)} "
            f"(pp. {section.page_start}–{section.page_end})"
        )
        print(
            textwrap.fill(
                preview,
                width=79,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )


def chunk_label(chunk) -> str:
    """Return a readable label for a chunk."""

    pieces = []

    if chunk.section_number is not None:
        pieces.append(str(chunk.section_number))

    if chunk.section_title is not None:
        pieces.append(chunk.section_title)

    if not pieces:
        pieces.append("front matter / unsectioned")

    if is_reference_section_chunk(chunk):
        pieces.append("[REFERENCE SECTION]")

    return " ".join(pieces)


def print_chunks(sections: Sequence[SectionSpan]) -> None:
    """Print in-memory chunks grouped by LLM section decisions."""

    chunks = make_section_aware_fixed_size_chunks(
        sections=sections,
        doc_id="llm-sectioning-playground",
    )

    print()
    print("Generated chunks")
    print("----------------")

    if not chunks:
        print("No chunks.")
        return

    last_key = object()

    for chunk in chunks:
        key = (
            chunk.section_index,
            chunk.section_number,
            chunk.section_title,
        )

        if key != last_key:
            print()
            print(f"Section: {chunk_label(chunk)}")
            last_key = key

        text = " ".join(chunk.text.split())
        preview = text[:240] + (" [...]" if len(text) > 240 else "")

        print(
            f"  chunk {chunk.chunk_index} "
            f"(pp. {chunk.page_start}–{chunk.page_end})"
        )
        print(
            textwrap.fill(
                preview,
                width=79,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )


def inspect_pdf(
    pdf_path: Path,
    model: str | None,
    ollama_url: str | None,
    max_pages: int,
) -> None:
    """Run the LLM section-recognition experiment for one PDF."""

    print()
    print(f"PDF: {pdf_path}")

    candidates = detect_heading_candidates_with_context(
        pdf_path,
        max_pages=max_pages,
    )
    print_candidates(candidates)

    if not candidates:
        return

    print()
    print("Asking Ollama to select real section headings...")

    decisions = select_section_headings_with_ollama(
        candidates=list(candidates),
        model=model,
        ollama_url=ollama_url,
        temperature=0.0,
    )

    print_decisions(decisions)
    input("\nPress Enter to see sections. ")

    sections = make_section_spans_from_llm_decisions(
        pdf_path=pdf_path,
        doc_id="llm-sectioning-playground",
        candidates=candidates,
        decisions=decisions,
    )

    print_sections(sections)
    input("\nPress Enter to see chunks. ")
    print_chunks(sections)


def llm_sectioning_loop(
    pdf_paths: Sequence[Path],
    model: str | None,
    ollama_url: str | None,
    max_pages: int,
) -> None:
    """Prompt for PDFs and run LLM-assisted section recognition."""

    print()
    print("LLM sectioning playground")
    print("Choose a PDF number to inspect.")
    print("Type list, ls, or pdfs to redisplay the numbered PDF list.")
    print("Type :q, :quit, done, quit, or exit to leave.")
    print()
    print_pdf_list(pdf_paths)

    while True:
        print()

        try:
            user_input = input("kurrent> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if user_input.lower() in {"list", "ls", "pdfs"}:
            print()
            print_pdf_list(pdf_paths)
            continue

        if not user_input.isdigit():
            print("Please enter a PDF number, list, ls, pdfs, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(pdf_paths):
            print(f"Please enter a number from 1 to {len(pdf_paths)}.")
            continue

        inspect_pdf(
            pdf_path=pdf_paths[index - 1],
            model=model,
            ollama_url=ollama_url,
            max_pages=max_pages,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the playground argument parser."""

    parser = argparse.ArgumentParser(
        description="Experiment with LLM-assisted PDF section recognition.",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT_DIR,
        help="PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ollama model name. Defaults to KURRENT_OLLAMA_MODEL or "
            "llama3.1:8b."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help=(
            "Ollama base URL. Defaults to KURRENT_OLLAMA_URL or "
            "http://127.0.0.1:11434."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Maximum number of early pages to scan for heading candidates.",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    pdf_paths = discover_pdfs(args.path)

    print(f"PDF source: {args.path}")
    print(f"PDFs found: {len(pdf_paths)}")

    llm_sectioning_loop(
        pdf_paths=pdf_paths,
        model=args.model,
        ollama_url=args.ollama_url,
        max_pages=args.max_pages,
    )
