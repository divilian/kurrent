# Refactoring of Bouchard tutorial "`rag.py`" to more encapsulated structure.

## `chunk.py`

* `chunk_text()`
* `load_txt_chunks()`
* `load_pdf_chunks()`

## `cli.py`

* `print_wrapped()`
* `parse_args()`
* main code

## `corpus_store.py`

* `get_document()`
* `list_documents()`
* `upsert_document()`
* `upsert_chunks()`
* `search()`

## `embed.py`

* `get_cache_filepath()`
* `get_embed_model()`
* `generate_embeddings()`

## `ingest.py`

* `ingest_path()`
* `discover_documents()`
* `ingest_file()`
* `compute_document_id()`
* `extract_text()`
* `embed_chunks()`

## `llm_backend.py`

* `LLMBackend` (abstract)
* `OpenAIBackend`
* `LocalLlamaBackend`
* `LocalHFBackend`

## `rag.py`

* `build_rag_prompts()`
* `perform_rag_query()`

## `schema.py`

* `Chunk`

## Removed altogether

* `load_csv_chunks()`
