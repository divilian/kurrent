# kurrent

A command-line tool to help keep your academic **k**nowledge c**urrent**.
Processes text files and PDFs, gives summaries, allows the user to identify and
preserve relationships between them, and provides an interactive RAG chat
interface to probe their contents.

## Design Overview

kurrent is a local-first, command-line tool for helping a researcher manage
and explore an academic literature corpus. Its central purpose is not merely
RAG chat, but the discovery, curation, and reuse of meaningful relationships
among papers.

The core workflow is:

```text
PDF → chunks → embeddings → proximity alerts → user confirmation → persistent semantic links
```

kurrent ingests academic PDFs, chunks their text, embeds those chunks into
Chroma, summarizes documents with a local Ollama model, and detects
semantically similar passages across papers. These machine-suggested
similarities are called **proximity alerts** (PAs). When the user confirms one,
it becomes a durable semantic **confirmed link** (CL) between chunks from two
documents.

### Storage Architecture

kurrent uses multiple storage layers, each with a distinct responsibility:

- SQLite: canonical application database for durable kurrent state
- Chroma: vector index for chunk embeddings and similarity search
- PDFs: original source files, managed according to storage mode
- Ollama: local LLM backend for summaries, PA explanations, and RAG answers


SQLite stores:

```text
document identities and PDF locations
chunks
proximity_alerts
confirmed_links
```

Later additions will include:

```text
annotations
source_discoveries
review_queues
```

Note: the reason for SQLite is not that Chroma cannot hold metadata. The reason
is that kurrent has durable relational state: documents have chunks, chunks
link to other chunks, confirmed links connect multiple records, and the vector
index may need to be rebuilt or replaced.

### Documents and Chunks

kurrent assigns its own stable UUIDs for:

- documents (`doc_id`)
- chunks (identified by combination of `doc_id` and chunk index number)

These IDs are independent of Chroma internal IDs, Zotero item keys, file paths,
or PDF filenames. Chroma records can include these IDs as metadata, but
kurrent's canonical identities live in SQLite.

### Proximity Alerts and Confirmed Links

kurrent assigns its own stable UUIDs for:

- proximity alerts (`pa_id`)
- confirmed links (`cl_id`)

Note that neither a PA nor a CL "belongs" to one chunk. Each is a separate
relational object involving at least two chunks.

### PDF Storage Model

kurrent does not decide behavior by sniffing whether a path happens to be
inside a Zotero (or other) storage directory. File handling is explicit and
semantic.

Each PDF has its own `storage_mode` recorded. These choices are available:

```text
managed  = kurrent owns/copies the PDF into its own storage
library  = an external reference manager owns the PDF
external = kurrent indexes a user-specified file in place
```

(Currently only `managed` is supported. `library` will be implemented for
Zotero very soon. `external` will be implemented only if in-place indexing
proves useful.)

### External Library Integrations

Zotero, and other tools, are integration adapters, not the conceptual center of
kurrent.

Instead of baking Zotero-specific fields directly into `documents`, a
separate `external_references` table contains tool-specific identifying info:

- `system_name`
- `external_item_id` - tool-specific item identifier
- `external_attachment_id` = tool-specific attachment identifier
- `external_uri` = tool-specific URI (optional)
- `metadata_json`: arbitrary tool-specific meta info

For example, for Zotero, we envision:

- `system_name` = `zotero`
- `external_item_id` = Zotero item key
- `external_attachment_id` = Zotero attachment key
- `external_uri` = `zotero://select/...`

This leaves room for future integrations with Mendeley, EndNote, OneNote,
filesystem folders, or other systems.



### Use Cases

The major use cases are:

- UC 1: interactive ingestion
- UC 2: RAG querying / console chat
- UC 3: document and chunk search
- UC 4: ongoing source discovery and review

(UC 4 is in the long-term design but will not be implemented immediately.)

### CLI Design

kurrent uses a single command-line tool with subcommands, rather than separate
executables.

Examples:

```bash
kurrent ingest file paper.pdf
kurrent ingest zotero JSFWJ7G6
kurrent chat --all
kurrent chat --docs smith2020 jones2023
kurrent search "bounded confidence"
kurrent links smith2020
kurrent sources check
kurrent sources review
```

This keeps kurrent conceptually unified while allowing distinct workflows.

### Summary Principle

```text
kurrent owns the research graph.
Chroma owns vector retrieval.
SQLite owns durable application state.
Zotero or another library manager may own some PDFs.
Ollama supplies local language-model intelligence.
```
