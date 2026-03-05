# Data type definitons.

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np


@dataclass(slots=True)
class Chunk:
    """
    A single retrievable unit of text.

    This is the canonical representation used downstream for embedding,
    indexing, and retrieval.
    """

    # Core identity
    chunk_id: int     # unique id within a document (0-based indexing)
    doc_id: str       # a hash of the document

    # Provenance
    source_path: str  # e.g., /home/stephen/research/WSC15/data/papers/328.pdf

    # Content
    text: str

    # Structural metadata (optional for plain text)

    # The raw, document-facing label (often messy).
    section_title: Optional[str] = None

    # Best-effort bucketing of this section into standard categories like
    #   "abstract", "methods", "conclusion", etc.
    section_type: Optional[str] = None  # e.g., "abstract", "methods", etc.

    starting_page_num: Optional[int] = None
    zotero_item_key: Optional[str] = None

    # The computed embedding for the entire text of the chunk.
    embedding: Optional[np.ndarray] = None

    # Arbitrary extensibility
    extra_metadata: Dict[str, str] = field(default_factory=dict)

    def text_for_embedding(self) -> str:
        """
        Return the text that should be embedded.
        We optionally prefix section information.
        """
        if self.section_title:
            return f"[{self.section_title}]\n{self.text}"
        return self.text


@dataclass
class Document(slots=True):
    """
    A document that has been ingested.
    """
    doc_id: str         # a hash of the document
    filename: str       # human-friendly name for display (at ingest time)
    source_path: str    # loc where doc was originally read
    source_type: str    # "filesys" or "zotero"
    file_size: int      # size of document in bytes
    ingested_at: float  # timestamp (time.time()) of ingestion


@dataclass
class SearchHit(slots=True):
    """
    A retrieved chunk that matched a query embedding.
    """
    doc_id: str     # the ID of the chunk's parent document
    chunk_id: int   # the ID of the chunk within that document
    score: float    # similarity between query and chunk

    section_title: Optional[str]

    # Best-effort "starting page number" for the chunk.
    starting_page_num: Optional[int]
