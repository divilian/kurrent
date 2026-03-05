# Main orchestrator that ingests files into the kurrent corpus:
# - discover files
# - compute document_id (content hash)
# - extract text (PDF/TXT)
# - chunk the text
# - compute embeddings
# - upsert into the corpus store

def ingest_path(path: Path, store: CorpusStore) -> list[str]:
    """
    Purpose:

    accept either
        a single file
        a directory

    discover ingestable files
    call ingest_file() for each

    Example usage from CLI:

    doc_ids = ingest_path(Path(args.path), store)

    Return value: list of document IDs ingested.
    """
    pass

def discover_documents(path: Path) -> list[Path]:
    """
    Handle directory traversal.

    Rules:

    if path is file → [path]

    if directory → recursively collect:

    *.pdf
    *.txt

    Later you might add:

    *.md
    *.html

    This keeps file discovery separate from ingestion.
    """
    pass


def ingest_file(path: Path, store: CorpusStore) -> str:
    """
    The core ingestion code.

    Pipeline inside this function:

    read bytes
    compute document hash
    extract text
    chunk text
    embed chunks
    store chunks

    Return value:

    document_id

    Example pipeline:

    doc_id = compute_document_id(path)

    text = extract_text(path)

    chunks = chunk_text(text, doc_id, source_path=path)

    embed_chunks(chunks)

    store.upsert_chunks(doc_id, chunks)
    """
    pass

def compute_document_id(path: Path) -> str:
    """
    Implementation:

    sha256(file_bytes)

    This gives you:

    deduplication

    stable identity

    independence from filenames
    """
    pass

def extract_text(path: Path) -> str:
    """
    if .pdf → extract_pdf_text()
    if .txt → read_text()

    Actual PDF logic belongs in loader.py or pdf_loader.py, but you can keep it simple initially.
    """
    pass


def embed_chunks(chunks: list[Chunk]) -> None:
    """
    Applies embedding model to all chunks.

    This mutates the chunks:

    chunk.embedding = vector

    The embedding model itself lives in embedder.py.
    """
    pass

def extract_txt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)

    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)

def extract_text(path: str) -> str:
    if path.lower().endswith(".txt"):
        return extract_txt_text(path)

    if path.lower().endswith(".pdf"):
        return extract_pdf_text(path)

    raise ValueError(f"Unsupported file type: {path}")

def load_txt_chunks(
    txt_path: str,
    chunk_size:int =1024,
) -> List[str]:
    """
    Split a plain text file into chunks.
    Returns: 1 (one document), and the chunks.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text, chunk_size)
    return chunks

def load_pdf_chunks(
    path: str,
    chunk_size: int=1024,
) -> str:
    """
    Extract whatever's readable from a PDF file, and split it into chunks.
    Returns: 1 (one document), and the chunks.
    """
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return chunk_text("\n".join(pages), chunk_size)

def wrap_chunks(
    chunk_texts: list[str],
    document_id: str,
    source_path: str,
    zotero_item_key: str | None = None,
    start_chunk_id: int = 0,     # If called more than once for the same doc
) -> list[Chunk]:
    """
    Given some chunks of raw text, return a list of Chunk objects corresponding
    to each one.
    """
    chunks = []

    for i, text in enumerate(chunk_texts):
        chunks.append(
            Chunk(
                text=text,
                document_id=document_id,
                chunk_id=i,
                source_path=source_path,
                zotero_item_key=zotero_item_key,
            )
        )

    return chunks
