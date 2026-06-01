"""Find candidate proximity alerts between semantically similar chunks.

A proximity alert (PA) is a candidate relationship discovered automatically by
comparing chunk embeddings. At this stage, proximity alerts are returned
in-memory only; they are not yet persisted to SQLite.

The basic workflow is:

    document id
    -> get this document's chunks
    -> for each chunk, query nearby chunks in Chroma
    -> exclude same-document matches
    -> return candidate ProximityAlert objects
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Sequence
from uuid import uuid4

from kurrent.chunker import chunker_version
from kurrent.embedder import Embedder
from kurrent.schema import (
    Chunk,
    ProximityAlert,
    ProximityAlertRecord,
    parse_chunk_id,
)
from kurrent.sectioner import is_reference_section_chunk
from kurrent.state_store import StateStore

__all__ = [
    "make_proximity_alert_record",
    "ProximityAlerter",
]

def make_proximity_alert_record(
    alert: ProximityAlert,
    explanation: str = "",
) -> ProximityAlertRecord:
    """Convert a generated proximity alert into a persisted PA record."""

    chunk_a_id, chunk_b_id = sorted([
        alert.source_chunk_id,
        alert.target_chunk_id,
    ])

    doc_a_id, chunker_a_version, chunk_a_index = parse_chunk_id(chunk_a_id)
    doc_b_id, chunker_b_version, chunk_b_index = parse_chunk_id(chunk_b_id)

    return ProximityAlertRecord(
        pa_id=str(uuid4()),
        doc_a_id=doc_a_id,
        chunker_a_version=chunker_a_version,
        chunk_a_index=chunk_a_index,
        doc_b_id=doc_b_id,
        chunker_b_version=chunker_b_version,
        chunk_b_index=chunk_b_index,
        score=alert.distance,
        status="pending",
        explanation=explanation,
        created_at=datetime.now(timezone.utc),
        decided_at=None,
    )

class ProximityAlerter:
    """Find candidate proximity alerts using the vector index."""

    def __init__(
        self,
        state_store: StateStore,
        embedder: Embedder,
    ) -> None:
        self.state_store = state_store
        self.embedder = embedder

    def find_alerts_for_document(
        self,
        doc_id: str,
        n_results_per_chunk: int = 10,
        max_distance: float | None = None,
        include_reference_sections: bool = False,
    ) -> list[ProximityAlert]:
        """Find candidate proximity alerts for one document.

        For each chunk in the source document, query the vector index for
        semantically similar chunks, excluding chunks from the same document.

        Results are returned in vector-distance order within each source chunk,
        and source chunks are processed in chunk_index order.

        By default, chunks from reference/bibliography sections are excluded
        as both source and target chunks because they often create noisy false
        positives.

        This method intentionally does not persist alerts yet.
        """
        source_chunks = self.state_store.get_chunks_for_document(
            doc_id=doc_id,
            chunker_version=chunker_version(),
        )

        if not source_chunks:
            raise ValueError(f"No chunks found for document: {doc_id}")

        source_document = self.state_store.get_document(doc_id)

        if source_document is None:
            raise ValueError(f"Document not found in kurrent state: {doc_id}")

        alerts: list[ProximityAlert] = []

        for source_chunk in source_chunks:
            if (
                not include_reference_sections
                and is_reference_section_chunk(source_chunk)
            ):
                continue

            alerts.extend(
                self.find_alerts_for_chunk(
                    source_chunk=source_chunk,
                    source_path=source_document.pdf_path,
                    n_results=n_results_per_chunk,
                    max_distance=max_distance,
                    exclude_doc_ids=[doc_id],
                    include_reference_sections=include_reference_sections,
                )
            )

        return alerts

    def find_alerts_for_chunk(
        self,
        source_chunk: Chunk,
        source_path: Path | None = None,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: Sequence[str] | None = None,
        include_reference_sections: bool = False,
    ) -> list[ProximityAlert]:
        """Find candidate proximity alerts for one source chunk.

        This uses the source chunk's existing indexed embedding as the vector
        query, rather than re-embedding the chunk text.
        """

        if (
            not include_reference_sections
            and is_reference_section_chunk(source_chunk)
        ):
            return []

        vector_matches = self.embedder.query_similar_chunks_by_chunk_id(
            source_chunk.chunk_id,
            n_results=n_results,
            max_distance=max_distance,
            exclude_doc_ids=exclude_doc_ids,
        )

        alerts: list[ProximityAlert] = []

        for match in vector_matches:
            if match.chunk_id == source_chunk.chunk_id:
                continue

            target_chunk = self.state_store.get_chunk(match.chunk_id)

            if target_chunk is None:
                raise ValueError(
                    "Vector index returned a chunk_id not found in "
                    f"kurrent state: {match.chunk_id!r}"
                )

            if (
                not include_reference_sections
                and is_reference_section_chunk(target_chunk)
            ):
                continue

            target_document = self.state_store.get_document(
                target_chunk.doc_id,
            )

            if target_document is None:
                raise ValueError(
                    "Target chunk exists in kurrent state, but parent "
                    f"document is missing: {target_chunk.doc_id!r}"
                )

            alerts.append(
                ProximityAlert(
                    source_chunk_id=source_chunk.chunk_id,
                    target_chunk_id=target_chunk.chunk_id,
                    distance=match.distance,
                    source_doc_id=source_chunk.doc_id,
                    target_doc_id=target_chunk.doc_id,
                    source_text=source_chunk.text,
                    target_text=target_chunk.text,
                    source_path=source_path,
                    target_path=target_document.pdf_path,
                    source_page_start=source_chunk.page_start,
                    source_page_end=source_chunk.page_end,
                    target_page_start=target_chunk.page_start,
                    target_page_end=target_chunk.page_end,
                )
            )

        return alerts


def _head_tail_wrap(
    text,
    head_chars=400,
    tail_chars=400,
    sep=" ... ",
    indent="",
    **kwargs,
):
    import textwrap

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

def _boxed(text):
    inner = f"* {text} *"
    border = "*" * len(inner)
    return f"{border}\n{inner}\n{border}"

if __name__ == "__main__":

    # Smoke test / IPython playground.
    #
    # Run from IPython with:
    #
    #     run -m kurrent.proximity_alerter
    #
    # Then inspect:
    #
    #     store
    #     embedder
    #     alerter
    #     doc_ids
    #     alerts

    import sys
    import random

    from kurrent.ingester import ingest_pdfs_recursively

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path("/home/stephen/teaching/350")

    tmpdir = Path("/tmp/kurrent-proximity-alerter")
    tmpdir.mkdir(parents=True, exist_ok=True)

    db_path = tmpdir / "kurrent.db"
    chroma_path = tmpdir / "chroma"

    store = StateStore(db_path)
    embedder = Embedder(chroma_path=chroma_path)

    print(f"Ingesting PDFs under: {root_dir}")
    print(f"Database path:        {db_path}")
    print(f"Chroma path:          {chroma_path}")
    print()

    doc_ids = ingest_pdfs_recursively(
        root_dir=root_dir,
        store=store,
        embedder=embedder,
        #no_more_than=6,
    )

    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    if not doc_ids:
        raise ValueError(f"No PDFs found under: {root_dir}")

    source_doc_id = random.choice(list(doc_ids.values()))
    source_doc = store.get_document(source_doc_id)
    print("*********************************")
    print(f"***Chose {source_doc.pdf_path.name} as smoke test source doc.***")

    alerts = alerter.find_alerts_for_document(
        source_doc_id,
        n_results_per_chunk=5,
        max_distance=None,
    )

    print()
    print(f"Documents ingested/indexed: {len(doc_ids)}")
    print(f"Alerts found:               {len(alerts)}")
    print()

    for i, alert in enumerate(alerts, start=1):
        print(f"{i}. distance={alert.distance:.4f}")

        print()
        if (
            alert.source_page_start is not None
            or alert.source_page_end is not None
        ):
            print(
                _boxed(
                    f"source: {alert.source_path.name} ("
                    f"pp.{alert.source_page_start}-{alert.source_page_end})"
                )
            )
        else:
            print(_boxed(f"   source: {alert.source_path.name}"))

        print(
            _head_tail_wrap(
                " ".join(alert.source_text.split()),
                indent="",
                sep=" [...] ",
            )
        )

        print()
        if (
            alert.target_page_start is not None
            or alert.target_page_end is not None
        ):
            print(
                _boxed(
                    f"target: {alert.target_path.name} ("
                    f"pp.{alert.target_page_start}-{alert.target_page_end})"
                )
            )
        else:
            print(_boxed(f"   target: {alert.target_path.name}"))

        print(
            _head_tail_wrap(
                " ".join(alert.target_text.split()),
                indent="",
                sep=" [...] ",
            )
        )
        print()
        if i < len(alerts):
            user_input = input(
                f"Press Enter for alert {i + 1} of "
                f"{len(alerts)} (or done): "
            )
            if 'done' in user_input.lower():
                sys.exit(0)
