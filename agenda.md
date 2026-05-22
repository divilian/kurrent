# kurrent Agenda

## MVP priorities

- [x] PDF ingestion
- [x] text extraction
- [x] chunking
- [x] embedding into Chroma
- [ ] smarter chunking (section-aware)
- [x] ingest and store metadata (authors, year, title)
- [ ] document summaries via Ollama
- [x] proximity alert detection
- [ ] user confirmation/rejection of alerts
- [ ] persistent confirmed links
- [x] basic search - metadata
- [x] basic search - full text (simple LIKE-based)
- [ ] basic RAG chat over a selected corpus

## Leftover stuff

- [x] Write `corpus_store.py` functions
- [x] Write `ingester.py` functions
- [ ] Filter out unnecessary code from `llm_backend.py` and focus it
- [x] Migrate from pypdf to PyMuPdf/fitz
- [ ] Be sure to enable SQLite foreign keys from Python

## Later

- [ ] PDF annotation/highlighting
- [ ] automatic source polling
- [ ] Zotero write-back/import
- [ ] integrations beyond Zotero
- [ ] advanced section-aware chunking
- [ ] advanced relationship taxonomies
- [ ] metadata quality check, and auto-Crossref if it looks sus


# Design decisions

## Storage architecture

- Use Chroma for vector search, embeddings, and retrieval-facing chunk
  metadata.
- Use SQLite as the canonical application database.
    - Note: We do _not_ use Chroma as the only database. Reason: Chroma is
      used primarily for semantic similarity search; SQLite stores durable
      relational application state.
- Chroma should be rebuildable without losing kurrent's durable state.

## Stable IDs

### kurrent-owned identity

- kurrent assigns its own stable `doc_id` and `chunk_id` values.
- These are **kurrent IDs**: identifiers owned by kurrent, not borrowed from
  Chroma, Zotero, the filesystem, PDF metadata, or DOI metadata.
- SQLite stores durable kurrent state.
- ChromaDB stores embeddings, retrieval metadata, and supports semantic
  retrieval. (Note: it does _not_ store the chunk text itself. Instead, it
  returns chunk IDs from similarity searches, and the chunk text is retrieved
  from SQLite.)

### Document and Chunk ID scheme

- On ingestion, each document receives a stable `kurrent` ID (which is a UUID)
  stored as `doc_id`.
- Each chunk receives a zero-based `chunk_index` within its document.
- Each chunk also has a globally unique `chunk_id`, derived from
  `doc_id` and `chunk_index`.
- In Python, `chunk_id` may be a `@property`; in SQLite and ChromaDB, it may be
  materialized for convenience.

## SQLite tables

Use SQLite tables for at least:

- `documents`
- `chunks`
- `proximity_candidates`
- `confirmed_links`
- `external_refs`

Later likely additions:

- `annotations`
- `source_discoveries`
- `review_queues`

Also:

- Do not include free-text note fields in the first schema pass.
- If needed later, add fields such as `confirmed_links.user_note` or
  `proximity_alerts.decision_note` through a migration.

### Mapping to Python dataclasses

Use dataclasses for in-memory domain objects and a small repository layer for
translating between those dataclasses and SQLite rows. The dataclasses will not
contain database persistence methods. In other words, prefer this style:

```
@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    ...

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"

def insert_chunk(conn: sqlite3.Connection, chunk: Chunk) -> None:
    conn.execute(
        """
        INSERT INTO chunks (chunk_id, doc_id, chunk_index, text)
        VALUES (?, ?, ?, ?)
        """,
        (chunk.chunk_id, chunk.doc_id, chunk.chunk_index, chunk.text),
    )


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        doc_id=row["doc_id"],
        chunk_index=row["chunk_index"],
        text=row["text"],
    )
```

## Terms for "search and query" type operations

In general, **search** is a user-facing operation, while **query** is a
low-level adapter.

Types of search:

- **semantic search**: fuzzy, embedding-based search on a free text expression.
  (brass tacks: query Chroma)
- **metadata search**: exact/structured search over metadata fields. (brass
  tacks: query SQLite)
- **full-text search**: lexical/exact-ish text search. (brass tacks: FTS on
  SQLite)
  
- **retrieval**: the lower-level act of fetching ranked hits/chunks/docs

## Proximity alerts (PAs) and confirmed links (CLs)

- A proximity alert (PA) is a machine-generated candidate relationship between
  chunks. It is permanent, though it may be marked "confirmed" or "rejected"
  by the user instead of "pending".
- A confirmed link (CL) is a user-approved semantic relationship between
  chunks. A CL always refers to a PA. (If we ever allow users to manually
  create a CL, kurrent must first create a manual-origin PA row, and then
  create the CL pointing to that row.)
- CLs are first-class records, not metadata fields on individual chunks.
- Store confirmed links in SQLite because they involve multiple chunks and
  durable user decisions.
- Store candidate proximity alerts separately from confirmed links.

## PDF storage

- Do not decide behavior by sniffing whether a path is inside Zotero's storage
  directory. Use an explicit storage abstraction instead.
- Use storage-mechanism-agnostic values:
  - `storage_mode = managed` (kurrent owns/copies the PDF)
  - `storage_mode = library` (an external tool, like Zotero, owns the PDF)
  - `storage_mode = external` (kurrent indexes a user-specified file in place)
- MVP may support only `managed` and `library`; postpone `external` unless
  needed.

### Zotero and other library managers

- Do not bake Zotero into the heart of the data model. Aim for library-manager
  agnosticism from the beginning. Simply implement Zotero integration first
  because Stephen uses Zotero.
- Represent Zotero and other systems through a general `external_references`
  table, with fields:
  - `external_refs.system_name` (e.g., "`zotero`")
  - `external_refs.doc_id`
  - `external_refs.external_item_id`
  - `external_refs.external_attachment_id`
  - `external_refs.external_uri`
  - `external_refs.metadata_json`

### Zotero-managed PDFs

- If a PDF is ingested through Zotero, Zotero retains custody of the PDF.
- kurrent indexes Zotero-managed PDFs but does not copy or mutate them by
  default.
- Store the Zotero item/attachment identity in `external_refs`.
- Store a path/hash reference sufficient for rechecking or reingesting.

### Non-Zotero PDFs

- If the user ingests a raw PDF path, default to copying it into
  kurrent-managed storage.
- Such documents use `storage_mode = managed`.
- Later, optionally support in-place indexing with `storage_mode = external`.

## RAG corpus

- Do not assume the RAG corpus is always the entire Chroma database.
- Treat the RAG corpus as an explicit document set.
- Support modes such as:
  - all documents
  - specified documents
  - documents matching tags/metadata/search results
  - documents related by confirmed links
  - automatically inferred temporary corpus
- Always make the active RAG corpus visible to the user.

## CLI shape

- Use one command-line executable, kurrent, with subcommands.
- Do not split the project into separate top-level executables.
- Likely commands:
  - `kurrent ingest ...`
  - `kurrent chat ...`
  - `kurrent search ...`
  - `kurrent links ...`
  - `kurrent sources ...`
- Prefer source-specific ingestion routes, for example:
  - `kurrent ingest file paper.pdf`
  - `kurrent ingest zotero JSFWJ7G6`

