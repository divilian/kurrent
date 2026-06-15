# kurrent Agenda

## MVP priorities

- [x] PDF ingestion
- [x] text extraction
- [x] chunking
- [x] embedding into Chroma
- [x] smarter chunking (section-aware)
- [x] ingest and store metadata (authors, year, title)
- [ ] document summaries via Ollama
- [x] proximity alert detection
- [ ] user confirmation/rejection of alerts
- [x] basic search - metadata (wire in to cli.py)
- [x] basic search - full text (simple LIKE-based) (wire in to cli.py)
- [x] semantic search (wire in to cli.py)
- [x] basic RAG chat over a selected corpus
- [x] boldface the parts of the chunks that semantically match
- [x] manual ingest mode
- [x] two-column parsing
- [x] pop up PDFs in user PDF reader
- [x] pop up PDFs to correct page in user PDF reader
- [x] pop up PDFs highlighting key text in user PDF reader
- [ ] summary verb
- [ ] list verb

## Bugs

## Later

- [ ] metadata quality check, and auto-Crossref if it looks sus (my prev question: "Why can't an LLM check metadata on ingestion like it does sections?")
- [x] PDF annotation/highlighting
- [ ] automatic source polling
- [ ] Zotero write-back/import
- [ ] integrations beyond Zotero
- [x] advanced section-aware chunking
- [ ] advanced relationship taxonomies
- [ ] "-f" option to ingest.py, to overwrite previous data explicitly (like for
  refreshing data for a new chunker/sectioner pipeline)
- [ ] Use citation count (or citation count divided by years since published)
  as a proxy for turning up in searches
- [ ] Alternatives to local Ollama for the corpus-grounded assessment (an API
  key, etc)
- [ ] OCR (see "Handling OCR Documents in Kurrent" chat)
- [ ] cascading chunking preferences: paragraph, sentence, word (chat called
  this a "modest boundary-aware chunker")
- [ ] zotero ingest mode
- [ ] external search (Be able to explicitly ask kurrent: "this chunk...what
  else is out there (not currently in kurrent db) that resembles it?"


# Design decisions

## User interface concepts

- User-facing focus should be on documents, not chunks. (They'll remember "thus
  and such was explored by Smith 2009," not "paragraph 6 of p. 3 of Smith
  2009.")
- For PAs/CLs, also surface it primarily as docs, secondarily as chunks.

## Storage architecture

- Use Chroma for vector search, embeddings, and retrieval-facing chunk
  metadata.
- Use SQLite as the canonical application database.
    - Note: We do _not_ use Chroma as the only database. Reason: Chroma is
      used primarily for semantic similarity search; SQLite stores durable
      relational application state.
- Chunk text is stored in both SQLite and Chroma.
    - SQLite remains the canonical home for clean chunk text, source browsing,
    highlighting, provenance, and durable application state.
    - Chroma stores a duplicate copy of the clean chunk text as a retrieval
    convenience, so vector hits can return usable passage text directly.
    - This duplication is acceptable for now: for a personal PDF corpus, the
    extra text storage is small enough compared with the convenience win.
    - Metadata-enriched embedding input is not stored as canonical chunk text.
    It is constructed temporarily for embedding and discarded.
- Chroma should be rebuildable without losing kurrent's durable state.
- We like SQLite even though DuckDB exists, because SQLite is good at OLTP as
  opposed to OLAP. (kinda ret-con reasoning)

## Stable IDs

### kurrent-owned identity

- kurrent assigns its own stable `doc_id` and `chunk_id` values.
- These are **kurrent IDs**: identifiers owned by kurrent, not borrowed from
  Chroma, Zotero, the filesystem, PDF metadata, or DOI metadata.
- SQLite stores durable kurrent state.
- ChromaDB stores embeddings, retrieval metadata, and supports semantic
  retrieval. (It also stores a duplicate copy of the clean chunk text for
  retrieval convenience. SQLite remains the canonical source of chunk text and
  durable kurrent state, however.)

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
  - `kurrent ask ...`
  - `kurrent search ...`
  - `kurrent links ...`
  - `kurrent sources ...`
- Prefer source-specific ingestion routes, for example:
  - `kurrent ingest file paper.pdf`
  - `kurrent ingest zotero JSFWJ7G6`




