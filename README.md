# kurrent

A command-line tool to help keep your academic **k**nowledge c**urrent**.
Processes text files and PDFs, gives summaries, allows the user to identify and
preserve relationships between them, and provides an interactive RAG chat
interface to probe their contents.

## Installation

Clone the repo, create and activate a Python environment, and install Kurrent in
editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Create your local environment file from the visible template:

```bash
cp dot_env.example .env
```

The defaults in `dot_env.example` are local Ollama defaults. Copying the file
is enough if the listed Ollama models are installed locally. Otherwise, edit
`.env` to use the Ollama models or aliases available on your machine. Also edit
`.env` when you want a different state directory or OpenAI-backed LLM roles.

Run the test suite with:

```bash
python -m pytest
```

For a faster local development pass, run:

```bash
python -m pytest -m "not slow"
```

## LLM configuration

Kurrent reads configuration from `.env` using the variables documented in
`dot_env.example`. Each LLM-backed role has its own model setting so different
jobs can use different models.

For a local Ollama-only setup, use settings like these:

```bash
KURRENT_RAG_LLM=ollama:qwen3:kurrent-rag-8b
KURRENT_RELEVANCE_LLM=ollama:llama3.1:kurrent-relevance-8b-instruct-q4_K_M
KURRENT_METADATA_LLM=ollama:llama3.1:kurrent-metadata-8b-instruct-q4_K_M
KURRENT_PDF_EXCERPT_LLM=ollama:llama3.1:kurrent-pdf-excerpt-8b-instruct-q4_K_M
KURRENT_SECTION_RECOGNITION_LLM=ollama:llama3.2:kurrent-sectioning-3b
KURRENT_SUMMARY_LLM=ollama:qwen2.5:kurrent-summarization-7b
KURRENT_SUMMARIZE_LLM=ollama:qwen2.5:kurrent-summarization-7b
```

LLM model strings must include an explicit provider prefix:

```text
ollama:<model-name>
openai:<model-name>
```

For example, `ollama:qwen3:kurrent-rag-8b` uses a local Ollama model, while
`openai:gpt-4.1-mini` uses OpenAI. To use OpenAI for any role, provide an API
key and set that role to an OpenAI model spec:

```bash
OPENAI_KEY=sk-...

KURRENT_RAG_LLM=openai:gpt-4.1-mini
KURRENT_RELEVANCE_LLM=openai:gpt-4.1-mini
KURRENT_SUMMARY_LLM=openai:gpt-4.1-mini
KURRENT_SUMMARIZE_LLM=openai:gpt-4.1-mini
```

You can also mix providers. For example, this configuration uses OpenAI for
interactive RAG, relevance judging, and deliberate summaries, while keeping
metadata extraction, PDF excerpt selection, section recognition, and screening
summaries local:

```bash
OPENAI_KEY=sk-...

KURRENT_RAG_LLM=openai:gpt-4.1-mini
KURRENT_RELEVANCE_LLM=openai:gpt-4.1-mini
KURRENT_METADATA_LLM=ollama:llama3.1:kurrent-metadata-8b-instruct-q4_K_M
KURRENT_PDF_EXCERPT_LLM=ollama:llama3.1:kurrent-pdf-excerpt-8b-instruct-q4_K_M
KURRENT_SECTION_RECOGNITION_LLM=ollama:llama3.2:kurrent-sectioning-3b
KURRENT_SUMMARY_LLM=ollama:qwen2.5:kurrent-summarization-7b
KURRENT_SUMMARIZE_LLM=openai:gpt-4.1-mini
```

Command-line switches override `.env` values and use the same explicit model-spec
format:

```bash
kurrent ask --rag-llm openai:gpt-4.1-mini "What does my corpus say about reciprocity?"
kurrent search --relevance-llm openai:gpt-4.1-mini "bounded confidence"
kurrent ingest --summary-llm openai:gpt-4.1-mini paper.pdf
kurrent summarize --summary-llm openai:gpt-4.1-mini
```

`KURRENT_SUMMARY_LLM` controls the fast screening summary shown during ingest.
`KURRENT_SUMMARIZE_LLM` controls the more complete summary produced by
`kurrent summarize`, so the two summary workflows may use different providers
and models.

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
Chroma, summarizes documents with configured LLM providers, and detects
semantically similar passages across papers. These machine-suggested
similarities are called **proximity alerts** (PAs). When the user confirms one,
it becomes a durable semantic **confirmed link** (CL) between chunks from two
documents.

### Storage Architecture

kurrent uses multiple storage layers, each with a distinct responsibility:

- SQLite: canonical application database for durable kurrent state
- Chroma: vector index for chunk embeddings and similarity search
- PDFs: original source files, managed according to storage mode
- LLM provider: explicit local Ollama or remote OpenAI model specs

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
kurrent ingest SmithWesson1948.pdf
kurrent ingest -y directory-with-pdfs
kurrent list
kurrent stats
kurrent search "bounded confidence"
kurrent ask "How is homophily defined?"
```

This keeps kurrent conceptually unified while allowing distinct workflows.

### Summary Principle

```text
kurrent owns the research graph.
Chroma owns vector retrieval.
SQLite owns durable application state.
Zotero or another library manager may own some PDFs.
LLM providers supply language-model intelligence through explicit model specs such as ollama:<model> or openai:<model>.
```

## Running tests

Run the fast test suite with:

```bash
python -m pytest -m "not slow"
```

Run only slow/integration-style tests with:

```bash
python -m pytest -m slow
```

Run all tests with:

```bash
python -m pytest
```

Slow tests include tests that load real embedding models, use heavier local resources, or otherwise take noticeably longer than ordinary unit tests.
