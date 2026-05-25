"""Manual playground for generating and browsing proximity alerts.

Run from the project root with:

    python playground/proximity_alert_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/proximity_alert_playground.py /path/to/pdf/root
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shutil
import sys
import textwrap

from kurrent.chunker import chunker_version
from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdfs_recursively
from kurrent.proximity_alerter import ProximityAlerter
from kurrent.schema import Chunk, Document, ProximityAlert
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
PLAYGROUND_DIR = Path("/tmp/kurrent-proximity-alert-playground")
QUIT_COMMANDS = {"q", "done", "quit", "exit"}


def existing_playground_paths(db_path: Path, chroma_path: Path) -> list[Path]:
    """Return existing playground database and Chroma paths."""

    candidates = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
        chroma_path,
    ]

    return [path for path in candidates if path.exists()]


def prepare_fresh_playground_state(db_path: Path, chroma_path: Path) -> None:
    """Delete existing playground state after confirmation."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    if not existing_paths:
        return

    print()
    print("Existing playground state found.")
    print("This playground is intended to start with fresh state each run.")
    print()
    print("Files/directories to delete:")

    for path in existing_paths:
        print(f"  {path}")

    print()

    try:
        response = input("Delete existing playground state? [Y/n] ")
    except EOFError:
        raise SystemExit(
            "Existing playground state was not deleted; aborting."
        )

    response = response.strip().lower()

    if response not in {"", "y", "yes"}:
        raise SystemExit("Cancelled; existing playground state left in place.")

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    print("Deleted existing playground state.")


def cleanup_playground_state(db_path: Path, chroma_path: Path) -> None:
    """Delete playground state on normal program exit."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if existing_paths:
        print()
        print("Deleted playground state.")


def boxed(text: str) -> str:
    """Return text inside a compact one-line box."""

    inner = f"* {text} *"
    border = "*" * len(inner)
    return f"{border}\n{inner}\n{border}"


def head_tail_wrap(
    text: str,
    width: int = 79,
    head_chars: int = 400,
    tail_chars: int = 400,
    sep: str = " [...] ",
    indent: str = "",
) -> str:
    """Wrap text, preserving the head and tail when text is long."""

    if len(text) <= head_chars + tail_chars:
        return textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
        )

    head = textwrap.wrap(
        text[:head_chars],
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    tail = textwrap.wrap(
        text[-tail_chars:],
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    head_text = " ".join(head)
    tail_text = " ".join(tail)

    return textwrap.fill(
        f"{head_text}{sep}{tail_text}",
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def page_range(
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Format a page range compactly."""

    if page_start is None and page_end is None:
        return "pp.?"

    if page_start == page_end:
        return f"p.{page_start}"

    return f"pp.{page_start}-{page_end}"


def format_chunk_section(chunk: Chunk | None) -> str | None:
    """Return a compact section label for a chunk, if available."""

    if chunk is None:
        return None

    pieces = []

    if chunk.section_number is not None:
        pieces.append(str(chunk.section_number))

    if chunk.section_title is not None:
        pieces.append(chunk.section_title)

    if not pieces:
        return None

    return " ".join(pieces)


def document_name(document: Document) -> str:
    """Return the display name for a document."""

    return document.pdf_path.name


def alert_source_label(
    alert: ProximityAlert,
    source_chunk: Chunk | None = None,
) -> str:
    """Return a compact source label for an alert."""

    source_name = (
        alert.source_path.name
        if alert.source_path is not None
        else "(unknown PDF)"
    )
    pages = page_range(alert.source_page_start, alert.source_page_end)
    section = format_chunk_section(source_chunk)

    if section is None:
        return f"{source_name} ({pages})"

    return f"{source_name}, section {section} ({pages})"


def alert_target_label(
    alert: ProximityAlert,
    target_chunk: Chunk | None = None,
) -> str:
    """Return a compact target label for an alert."""

    target_name = (
        alert.target_path.name
        if alert.target_path is not None
        else "(unknown PDF)"
    )
    pages = page_range(alert.target_page_start, alert.target_page_end)
    section = format_chunk_section(target_chunk)

    if section is None:
        return f"{target_name} ({pages})"

    return f"{target_name}, section {section} ({pages})"


def print_document_list(documents: Sequence[Document]) -> None:
    """Print a numbered list of documents."""

    if not documents:
        print("No documents.")
        return

    for i, document in enumerate(documents, start=1):
        print(f"{i}. {document_name(document)}")


def print_chunk_list(
    chunks: Sequence[Chunk],
    alerts_by_source_chunk_id: dict[str, list[ProximityAlert]],
) -> None:
    """Print chunks from the source document, marking those with alerts."""

    if not chunks:
        print("No chunks.")
        return

    for i, chunk in enumerate(chunks, start=1):
        alerts = alerts_by_source_chunk_id.get(chunk.chunk_id, [])
        marker = "PA" if alerts else "--"
        pages = page_range(chunk.page_start, chunk.page_end)

        section = format_chunk_section(chunk)

        if section is None:
            print(
                f"{i}. [{marker}] chunk {chunk.chunk_index} "
                f"({pages})  alerts={len(alerts)}"
            )
        else:
            print(
                f"{i}. [{marker}] chunk {chunk.chunk_index} "
                f"section={section} ({pages})  alerts={len(alerts)}"
            )


def print_alert_list(
    alerts: Sequence[ProximityAlert],
    store: StateStore,
) -> None:
    """Print a numbered list of PAs triggered by one source chunk."""

    if not alerts:
        print("No proximity alerts for this chunk.")
        return

    for i, alert in enumerate(alerts, start=1):
        target_chunk = store.get_chunk(alert.target_chunk_id)

        print(f"{i}. distance={alert.distance:.4f}")
        print(f"   target: {alert_target_label(alert, target_chunk)}")


def print_alert_detail(
    alert: ProximityAlert,
    index: int,
    store: StateStore,
) -> None:
    """Print source and target chunk details for a selected PA."""

    source_chunk = store.get_chunk(alert.source_chunk_id)
    target_chunk = store.get_chunk(alert.target_chunk_id)

    print()
    print(f"Proximity alert {index}")
    print(f"distance: {alert.distance:.4f}")
    print()

    print(boxed(f"source: {alert_source_label(alert, source_chunk)}"))
    print(
        head_tail_wrap(
            " ".join(alert.source_text.split()),
            sep=" [...] ",
        )
    )

    print()
    print(boxed(f"target: {alert_target_label(alert, target_chunk)}"))
    print(
        head_tail_wrap(
            " ".join(alert.target_text.split()),
            sep=" [...] ",
        )
    )
    print()


def group_alerts_by_source_chunk(
    alerts: Sequence[ProximityAlert],
) -> dict[str, list[ProximityAlert]]:
    """Group generated alerts by source chunk id."""

    grouped: dict[str, list[ProximityAlert]] = {}

    for alert in alerts:
        grouped.setdefault(alert.source_chunk_id, []).append(alert)

    return grouped


def get_documents_from_ingest_result(
    store: StateStore,
    doc_ids_by_path: dict[Path, str],
) -> list[Document]:
    """Retrieve ingested documents in path-sorted order."""

    documents: list[Document] = []

    for path in sorted(doc_ids_by_path):
        doc_id = doc_ids_by_path[path]
        document = store.get_document(doc_id)

        if document is None:
            raise ValueError(f"Document not found in kurrent state: {doc_id}")

        documents.append(document)

    return documents


def choose_document(documents: Sequence[Document]) -> Document | None:
    """Let the user choose a source document from a numbered list."""

    while True:
        print()
        print("Choose a source document.")
        print_document_list(documents)
        print()

        try:
            user_input = input("PA (choose doc)> ").strip()
        except EOFError:
            print()
            return None

        if user_input in QUIT_COMMANDS:
            return None

        if not user_input:
            continue

        if not user_input.isdigit():
            print("Please enter a document number, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(documents):
            print(f"Please enter a number from 1 to {len(documents)}.")
            continue

        return documents[index - 1]


def browse_alerts_for_chunk(
    chunk: Chunk,
    alerts: Sequence[ProximityAlert],
    store: StateStore,
) -> None:
    """Let the user inspect PAs triggered by one source chunk."""

    while True:
        print()
        section = format_chunk_section(chunk)

        if section is None:
            print(f"Chunk {chunk.chunk_index} triggered {len(alerts)} PA(s).")
        else:
            print(
                f"Chunk {chunk.chunk_index} "
                f"(section {section}) triggered {len(alerts)} PA(s)."
            )

        print_alert_list(alerts, store)
        print()
        print("Enter an alert number for details, or done.")

        try:
            user_input = input("PA> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if not user_input.isdigit():
            print("Please enter an alert number, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(alerts):
            print(f"Please enter a number from 1 to {len(alerts)}.")
            continue

        print_alert_detail(alerts[index - 1], index, store)


def browse_chunks_for_document(
    store: StateStore,
    document: Document,
    alerts: Sequence[ProximityAlert],
) -> None:
    """Let the user choose source chunks and inspect triggered PAs."""

    chunks = store.get_chunks_for_document(
        doc_id=document.doc_id,
        chunker_version=chunker_version(),
    )
    alerts_by_source_chunk_id = group_alerts_by_source_chunk(alerts)

    while True:
        print()
        print(f"Source document: {document_name(document)}")
        print("Chunks marked [PA] triggered one or more proximity alerts.")
        print()
        print_chunk_list(chunks, alerts_by_source_chunk_id)
        print()
        print("Enter a chunk number to browse its PAs, or done.")

        try:
            user_input = input("PA (choose chunk)> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if not user_input.isdigit():
            print("Please enter a chunk number, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(chunks):
            print(f"Please enter a number from 1 to {len(chunks)}.")
            continue

        chunk = chunks[index - 1]
        chunk_alerts = alerts_by_source_chunk_id.get(chunk.chunk_id, [])

        if not chunk_alerts:
            print("That chunk did not trigger any proximity alerts.")
            continue

        browse_alerts_for_chunk(chunk, chunk_alerts, store)


def proximity_alert_loop(
    store: StateStore,
    alerter: ProximityAlerter,
    documents: Sequence[Document],
    n_results_per_chunk: int = 5,
    max_distance: float | None = None,
) -> None:
    """Prompt for source documents and browse generated PAs."""

    print()
    print("Proximity alert playground")
    print("Choose a document to generate and browse proximity alerts.")
    print(f"Type {', '.join(QUIT_COMMANDS)} to leave.")

    while True:
        document = choose_document(documents)

        if document is None:
            break

        print()
        print(f"Generating PAs for {document_name(document)} ...")

        alerts = alerter.find_alerts_for_document(
            document.doc_id,
            n_results_per_chunk=n_results_per_chunk,
            max_distance=max_distance,
        )

        print(f"Alerts generated: {len(alerts)}")
        browse_chunks_for_document(store, document, alerts)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = DEFAULT_ROOT_DIR

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PLAYGROUND_DIR / "kurrent.db"
    chroma_path = PLAYGROUND_DIR / "chroma"
    prepare_fresh_playground_state(db_path, chroma_path)

    store = StateStore(db_path)
    embedder = Embedder(chroma_path=chroma_path)

    try:
        print(f"Ingesting PDFs under: {root_dir}")
        print(f"Database path:        {db_path}")
        print(f"Chroma path:          {chroma_path}")
        print()

        doc_ids_by_path = ingest_pdfs_recursively(
            root_dir=root_dir,
            store=store,
            embedder=embedder,
        )

        documents = get_documents_from_ingest_result(
            store=store,
            doc_ids_by_path=doc_ids_by_path,
        )

        alerter = ProximityAlerter(
            state_store=store,
            embedder=embedder,
        )

        print()
        print(f"Documents ingested/indexed: {len(documents)}")

        proximity_alert_loop(
            store=store,
            alerter=alerter,
            documents=documents,
            n_results_per_chunk=5,
            max_distance=None,
        )
    finally:
        cleanup_playground_state(db_path, chroma_path)
