from pathlib import Path
from datetime import datetime, timezone

import sqlite3

from kurrent.file_utils import normalize_path
from kurrent.schema import (
    Document,
    Chunk,
    ChunkHit,
    make_chunk_id,
    parse_chunk_id,
    ProximityAlertRecord,
    ConfirmedLink,
    PAStatus,
    ExtractedMetadata,
)

__all__ = [
    "TEXT_PIPELINE_STATUS_OK",
    "TEXT_PIPELINE_STATUS_NO_EXTRACTABLE_TEXT",
    "canonical_chunk_pair",
    "StateStore",
]

TEXT_PIPELINE_STATUS_OK = "ok"
TEXT_PIPELINE_STATUS_NO_EXTRACTABLE_TEXT = "no_extractable_text"
VALID_TEXT_PIPELINE_STATUSES = {
    TEXT_PIPELINE_STATUS_OK,
    TEXT_PIPELINE_STATUS_NO_EXTRACTABLE_TEXT,
}

def canonical_chunk_pair(
    chunk_a_id: str,
    chunk_b_id: str,
) -> tuple[str, str]:
    """Return chunk ids in canonical order for undirected PA lookup."""

    if chunk_a_id == chunk_b_id:
        raise ValueError(
            f"Proximity alert cannot link a chunk to itself: {chunk_a_id!r}"
        )

    return tuple(sorted([chunk_a_id, chunk_b_id]))

class StateStore:
    """
    Maintains all basic information for kurrent in an SQLite database (missing
    only the vector embeddings, which are in Chroma; and the source PDF files,
    which are in the kurrent_files directory).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self.conn:
            self.conn.executescript(schema_sql)
            self._migrate_document_pipeline_state()

    def _migrate_document_pipeline_state(self) -> None:
        """Add pipeline-state columns needed by newer kurrent versions."""

        rows = self.conn.execute(
            "PRAGMA table_info(document_pipeline_state)"
        ).fetchall()
        column_names = {row["name"] for row in rows}

        if "status" not in column_names:
            self.conn.execute(
                """
                ALTER TABLE document_pipeline_state
                ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'
                """
            )

        if "message" not in column_names:
            self.conn.execute(
                """
                ALTER TABLE document_pipeline_state
                ADD COLUMN message TEXT
                """
            )

    def insert_document(self, document: Document) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT INTO documents
                (
                    doc_id,
                    pdf_sha256,
                    storage_mode,
                    pdf_path,
                    ingested_at,
                    title,
                    authors,
                    year,
                    doi
                )
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,    # note to SD: already str, not UUID
                    document.pdf_sha256,
                    document.storage_mode,
                    str(document.pdf_path),   # note to SD: convert Path to str
                    document.ingested_at.isoformat(),
                    document.title,
                    document.authors,
                    document.year,
                    document.doi,
                )
            )

    def get_document_by_sha256(self, pdf_sha256: str) -> Document | None:
        """
        Look up a document by its contents, essentially. Do we already have
        this exact PDF file stored in kurrent, no matter what path that might
        have been at? If so, return that document. Otherwise, None.
        """
        row = self.conn.execute(
            """
            SELECT doc_id, pdf_path, pdf_sha256
            FROM documents
            WHERE pdf_sha256 = ?
            """,
            (pdf_sha256,),
        ).fetchone()

        if row is None:
            return None

        doc_id = row["doc_id"]
        return self.get_document(doc_id)

    def get_or_create_document(
        self,
        pdf_path: str | Path,
        pdf_sha256: str,
        metadata: ExtractedMetadata | None = None,
    ) -> Document:
        """
        Attempt to create a document with the given contents. Oh, but if those
        contents already exist (no matter at what location) by all means return
        that existing one instead.
        """
        existing = self.get_document_by_sha256(pdf_sha256)

        if existing is not None:
            return existing

        doc = Document.for_pdf(
            pdf_path=Path(pdf_path),
            pdf_sha256=pdf_sha256,
            metadata=metadata,
        )

        self.insert_document(doc)
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_document(row)

    def list_documents(self) -> list[Document]:
        """Return all documents known to kurrent, ordered for display."""

        rows = self.conn.execute(
            """
            SELECT *
            FROM documents
            ORDER BY
                title IS NULL,
                title COLLATE NOCASE,
                pdf_path COLLATE NOCASE
            """
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def update_document_metadata(
        self,
        doc_id: str,
        title: str | None = None,
        authors: str | None = None,
        year: int | None = None,
        doi: str | None = None,
    ) -> Document:
        """Update selected metadata fields for an existing document.

        Fields passed as None are unchanged.
        """

        if self.get_document(doc_id) is None:
            raise ValueError(f"Document not found: {doc_id}")

        updates = {}

        if title is not None:
            updates["title"] = title

        if authors is not None:
            updates["authors"] = authors

        if year is not None:
            updates["year"] = year

        if doi is not None:
            updates["doi"] = doi

        if not updates:
            raise ValueError("No metadata fields were provided for update.")

        set_clause = ", ".join(f"{field} = ?" for field in updates)
        values = list(updates.values()) + [doc_id]

        with self.conn:
            self.conn.execute(
                f"""
                UPDATE documents
                SET {set_clause}
                WHERE doc_id = ?
                """,
                values,
            )

        document = self.get_document(doc_id)

        if document is None:
            raise ValueError(f"Document disappeared after update: {doc_id}")

        return document

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            pdf_sha256=row["pdf_sha256"],
            storage_mode=row["storage_mode"],
            # note to SD: convert str to fully normalized Path
            pdf_path=normalize_path(row["pdf_path"]),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            doi=row["doi"],
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        return Chunk(
            doc_id=row["doc_id"],
            chunker_version=row["chunker_version"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            text_sha256=row["text_sha256"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_index=row["section_index"],
            section_number=row["section_number"],
            section_title=row["section_title"],
        )

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        with self.conn:
            self.conn.executemany("""
                INSERT INTO chunks
                (
                    doc_id,
                    chunker_version,
                    chunk_index,
                    text,
                    text_sha256,
                    page_start,
                    page_end,
                    section_index,
                    section_number,
                    section_title
                )
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.doc_id,
                        chunk.chunker_version,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.text_sha256,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_index,
                        chunk.section_number,
                        chunk.section_title,
                    )
                    for chunk in chunks
                ],
            )

    def get_chunk_by_parts(
        self,
        doc_id: str,
        chunker_version: str,
        chunk_index: int,
    ) -> Chunk | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE doc_id = ? AND chunker_version = ? AND chunk_index = ?
            """,
            (doc_id, chunker_version, chunk_index),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_chunk(row)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        doc_id, chunker_version, chunk_index = parse_chunk_id(chunk_id)

        return self.get_chunk_by_parts(
            doc_id=doc_id,
            chunker_version=chunker_version,
            chunk_index=chunk_index,
        )

    def get_chunks_for_document(
        self,
        doc_id: str,
        chunker_version: str,
    ) -> Chunk | None:
        rows = self.conn.execute(
            """
            SELECT *
            FROM chunks
            WHERE doc_id = ?
              AND chunker_version = ?
            ORDER BY chunk_index
            """,
            (doc_id, chunker_version),
        ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    def get_document_pipeline_state(self, doc_id: str) -> sqlite3.Row | None:
        """Return the stored derived-text pipeline-state row, if present."""

        return self.conn.execute(
            """
            SELECT pipeline_fingerprint, status, message, updated_at
            FROM document_pipeline_state
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

    def get_document_pipeline_fingerprint(self, doc_id: str) -> str | None:
        """Return the stored derived-text pipeline fingerprint, if present."""

        row = self.get_document_pipeline_state(doc_id)

        if row is None:
            return None

        return row["pipeline_fingerprint"]

    def get_document_pipeline_status(self, doc_id: str) -> str | None:
        """Return the stored derived-text pipeline status, if present."""

        row = self.get_document_pipeline_state(doc_id)

        if row is None:
            return None

        return row["status"]

    def set_document_pipeline_state(
        self,
        doc_id: str,
        pipeline_fingerprint: str,
        status: str = TEXT_PIPELINE_STATUS_OK,
        message: str | None = None,
    ) -> None:
        """Record the pipeline fingerprint and text-artifact status."""

        if self.get_document(doc_id) is None:
            raise ValueError(f"Document not found: {doc_id}")

        if status not in VALID_TEXT_PIPELINE_STATUSES:
            raise ValueError(f"Invalid text pipeline status: {status!r}")

        updated_at = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO document_pipeline_state
                (doc_id, pipeline_fingerprint, status, message, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    pipeline_fingerprint = excluded.pipeline_fingerprint,
                    status = excluded.status,
                    message = excluded.message,
                    updated_at = excluded.updated_at
                """,
                (doc_id, pipeline_fingerprint, status, message, updated_at),
            )

    def set_document_pipeline_fingerprint(
        self,
        doc_id: str,
        pipeline_fingerprint: str,
    ) -> None:
        """Record that this document has current extractable text chunks."""

        self.set_document_pipeline_state(
            doc_id=doc_id,
            pipeline_fingerprint=pipeline_fingerprint,
            status=TEXT_PIPELINE_STATUS_OK,
            message=None,
        )

    def mark_document_no_extractable_text(
        self,
        doc_id: str,
        pipeline_fingerprint: str,
        message: str | None = None,
    ) -> None:
        """Record that this pipeline found no text to chunk for a document."""

        self.set_document_pipeline_state(
            doc_id=doc_id,
            pipeline_fingerprint=pipeline_fingerprint,
            status=TEXT_PIPELINE_STATUS_NO_EXTRACTABLE_TEXT,
            message=message,
        )

    def document_has_no_extractable_text(
        self,
        doc_id: str,
        pipeline_fingerprint: str | None = None,
    ) -> bool:
        """Return whether the document is known to have no extractable text.

        The optional pipeline_fingerprint argument is accepted for compatibility
        with older callers, but no-text markings intentionally survive ordinary
        text-pipeline changes. A future OCR-capable workflow should clear or
        override this status explicitly when it wants to retry these PDFs.
        """

        row = self.get_document_pipeline_state(doc_id)

        if row is None:
            return False

        return row["status"] == TEXT_PIPELINE_STATUS_NO_EXTRACTABLE_TEXT

    def document_has_current_pipeline(
        self,
        doc_id: str,
        pipeline_fingerprint: str,
    ) -> bool:
        """Return whether stored derived-artifact state matches the pipeline."""

        return self.get_document_pipeline_fingerprint(doc_id) == pipeline_fingerprint

    def delete_derived_artifacts_for_document(self, doc_id: str) -> None:
        """Delete disposable derived artifacts for one document.

        This preserves the document record and its metadata, but removes stored
        chunks and the pipeline-state row so the PDF can be extracted, sectioned,
        and chunked again. Current PA/CL rows are structurally chunk-based, so
        rows referencing this document's chunks must also be removed before the
        chunks can be deleted.
        """

        with self.conn:
            self.conn.execute(
                """
                DELETE FROM confirmed_links
                WHERE pa_id IN (
                    SELECT pa_id
                    FROM proximity_alerts
                    WHERE doc_a_id = ? OR doc_b_id = ?
                )
                """,
                (doc_id, doc_id),
            )
            self.conn.execute(
                """
                DELETE FROM proximity_alerts
                WHERE doc_a_id = ? OR doc_b_id = ?
                """,
                (doc_id, doc_id),
            )
            self.conn.execute(
                """
                DELETE FROM chunks
                WHERE doc_id = ?
                """,
                (doc_id,),
            )
            self.conn.execute(
                """
                DELETE FROM document_pipeline_state
                WHERE doc_id = ?
                """,
                (doc_id,),
            )


    @staticmethod
    def _like_search_pattern(search_text: str) -> str:
        """Return a literal SQLite LIKE pattern for substring search.

        %, _, and backslash are escaped so user search text is treated as
        ordinary text rather than as LIKE wildcard syntax.
        """

        escaped = (
            search_text
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        return f"%{escaped}%"

    def search_documents_by_metadata(
        self,
        search_text: str,
        limit: int = 20,
    ) -> list[Document]:
        """Return documents whose metadata contains the search text.

        This is a simple SQLite LIKE search across title, authors, year, DOI,
        and PDF path. It intentionally does not use embeddings or FTS5.
        """

        search_text = search_text.strip()

        if not search_text:
            return []

        pattern = self._like_search_pattern(search_text)

        rows = self.conn.execute(
            """
            SELECT *
            FROM documents
            WHERE title LIKE ? ESCAPE '\\'
               OR authors LIKE ? ESCAPE '\\'
               OR CAST(year AS TEXT) LIKE ? ESCAPE '\\'
               OR doi LIKE ? ESCAPE '\\'
               OR pdf_path LIKE ? ESCAPE '\\'
            ORDER BY
                year IS NULL,
                year,
                title IS NULL,
                title COLLATE NOCASE,
                pdf_path COLLATE NOCASE
            LIMIT ?
            """,
            (
                pattern,
                pattern,
                pattern,
                pattern,
                pattern,
                limit,
            ),
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def search_chunks_by_fulltext(
        self,
        search_text: str,
        limit: int = 20,
    ) -> list[ChunkHit]:
        """Return chunk hits whose text contains the search text.

        This is a simple SQLite LIKE search over stored chunk text. Results
        are chunk-level because the match occurs inside a specific chunk, but
        each result is enriched with document metadata for display.
        """

        search_text = search_text.strip()

        if not search_text:
            return []

        pattern = self._like_search_pattern(search_text)

        rows = self.conn.execute(
            """
            SELECT
                chunks.doc_id,
                chunks.chunker_version,
                chunks.chunk_index,
                chunks.text,
                chunks.page_start,
                chunks.page_end,
                chunks.section_index,
                chunks.section_number,
                chunks.section_title,
                documents.pdf_path,
                documents.title
            FROM chunks
            JOIN documents
              ON chunks.doc_id = documents.doc_id
            WHERE chunks.text LIKE ? ESCAPE '\\'
            ORDER BY
                documents.title IS NULL,
                documents.title COLLATE NOCASE,
                documents.pdf_path COLLATE NOCASE,
                chunks.chunker_version,
                chunks.chunk_index
            LIMIT ?
            """,
            (
                pattern,
                limit,
            ),
        ).fetchall()

        return [
            ChunkHit(
                chunk_id=make_chunk_id(
                    row["doc_id"],
                    row["chunker_version"],
                    row["chunk_index"],
                ),
                distance=None,
                text=row["text"],
                path=normalize_path(row["pdf_path"]),
                title=row["title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section_index=row["section_index"],
                section_number=row["section_number"],
                section_title=row["section_title"],
            )
            for row in rows
        ]

    def insert_proximity_alert(self, pa: ProximityAlertRecord) -> None:
        """
        Idempotent. If this PA has already been recorded in the DB, don't add
        another.
        """
        existing = self.get_proximity_alert_by_chunk_ids(
            pa.chunk_a_id,
            pa.chunk_b_id,
        )

        if existing is not None:
            return

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO proximity_alerts
                (
                    pa_id,
                    doc_a_id,
                    chunker_a_version,
                    chunk_a_index,
                    doc_b_id,
                    chunker_b_version,
                    chunk_b_index,
                    score,
                    status,
                    explanation,
                    created_at,
                    decided_at
                )
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pa.pa_id,
                    pa.doc_a_id,
                    pa.chunker_a_version,
                    pa.chunk_a_index,
                    pa.doc_b_id,
                    pa.chunker_b_version,
                    pa.chunk_b_index,
                    pa.score,
                    pa.status,
                    pa.explanation,
                    pa.created_at.isoformat(),
                    (
                        pa.decided_at.isoformat()
                        if pa.decided_at is not None
                        else None
                    ),
                ),
            )

    def get_proximity_alert(self, pa_id: str) -> ProximityAlertRecord | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM proximity_alerts
            WHERE pa_id = ?
            """,
            (pa_id,),
        ).fetchone()

        if row is None:
            return None

        return ProximityAlertRecord(
            pa_id=row["pa_id"],
            doc_a_id=row["doc_a_id"],
            chunker_a_version=row["chunker_a_version"],
            chunk_a_index=row["chunk_a_index"],
            doc_b_id=row["doc_b_id"],
            chunker_b_version=row["chunker_b_version"],
            chunk_b_index=row["chunk_b_index"],
            score=row["score"],
            status=row["status"],
            explanation=row["explanation"],
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=(
                datetime.fromisoformat(row["decided_at"])
                if row["decided_at"] is not None
                else None
            ),
        )

    def get_proximity_alert_by_chunk_ids(
        self,
        chunk_a_id: str,
        chunk_b_id: str,
    ) -> ProximityAlertRecord | None:
        """Retrieve a PA record by its two chunk ids.

        PA chunk pairs are treated as undirected, so the lookup canonicalizes
        the pair before querying. This means A/B and B/A find the same PA
        record.
        """
        chunk_a_id, chunk_b_id = canonical_chunk_pair(
            chunk_a_id,
            chunk_b_id,
        )

        (
            doc_a_id,
            chunker_a_version,
            chunk_a_index,
        ) = parse_chunk_id(chunk_a_id)

        (
            doc_b_id,
            chunker_b_version,
            chunk_b_index,
        ) = parse_chunk_id(chunk_b_id)

        row = self.conn.execute(
            """
            SELECT *
            FROM proximity_alerts
            WHERE doc_a_id = ?
              AND chunker_a_version = ?
              AND chunk_a_index = ?
              AND doc_b_id = ?
              AND chunker_b_version = ?
              AND chunk_b_index = ?
            """,
            (
                doc_a_id,
                chunker_a_version,
                chunk_a_index,
                doc_b_id,
                chunker_b_version,
                chunk_b_index,
            ),
        ).fetchone()

        if row is None:
            return None

        return ProximityAlertRecord(
            pa_id=row["pa_id"],
            doc_a_id=row["doc_a_id"],
            chunker_a_version=row["chunker_a_version"],
            chunk_a_index=row["chunk_a_index"],
            doc_b_id=row["doc_b_id"],
            chunker_b_version=row["chunker_b_version"],
            chunk_b_index=row["chunk_b_index"],
            score=row["score"],
            status=row["status"],
            explanation=row["explanation"],
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=(
                datetime.fromisoformat(row["decided_at"])
                if row["decided_at"] is not None
                else None
            ),
        )

    def update_proximity_alert_status(
        self,
        pa_id: str,
        status: PAStatus,
        decided_at: datetime | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute("""
                UPDATE proximity_alerts
                SET status=?, decided_at=?
                WHERE pa_id=?
                """,
                (
                    status,
                    (
                        decided_at.isoformat()
                        if decided_at is not None
                        else None
                    ),
                    pa_id,
                )
            )

    def insert_confirmed_link(self, cl: ConfirmedLink) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT INTO confirmed_links
                (
                    cl_id,
                    pa_id,
                    created_at,
                    relationship_type
                )
                VALUES
                (?, ?, ?, ?)
                """,
                (
                    cl.cl_id,
                    cl.pa_id,
                    cl.created_at.isoformat(),
                    cl.relationship_type,
                )
            )

    def get_confirmed_link(self, cl_id: str) -> ConfirmedLink | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM confirmed_links
            WHERE cl_id = ?
            """,
            (cl_id,),
        ).fetchone()

        if row is None:
            return None

        return ConfirmedLink(
            cl_id=row["cl_id"],
            pa_id=row["pa_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            relationship_type=row["relationship_type"],
        )

    def close(self) -> None:
        self.conn.close()

    # (note to SD: __enter__ and __exit__ supports automatic closing in "with" statements)
    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
