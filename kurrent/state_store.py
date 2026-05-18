from pathlib import Path
from datetime import datetime

import sqlite3

from kurrent.schema import (
    Document,
    Chunk,
    ProximityAlert,
    ConfirmedLink,
    PAStatus,
)


class StateStore:
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

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            doc_id=row["doc_id"],
            pdf_sha256=row["pdf_sha256"],
            storage_mode=row["storage_mode"],
            # note to SD: convert str to fully normalized Path
            pdf_path=Path(row["pdf_path"]).expanduser().resolve(),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            doi=row["doi"],
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
                    page_end
                )
                VALUES
                (?, ?, ?, ?, ?, ?, ?)
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
                    )
                    for chunk in chunks
                ],
            )

    def get_chunk(
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

        return Chunk(
            doc_id=row["doc_id"],
            chunker_version=row["chunker_version"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            text_sha256=row["text_sha256"],
            page_start=row["page_start"],
            page_end=row["page_end"],
        )

    def insert_proximity_alert(self, pa: ProximityAlert) -> None:
        with self.conn:
            self.conn.execute("""
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
                )
            )

    def get_proximity_alert(self, pa_id: str) -> ProximityAlert | None:
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

        return ProximityAlert(
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
