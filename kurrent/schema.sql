CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    pdf_sha256 TEXT NOT NULL UNIQUE,
    storage_mode TEXT NOT NULL,
    pdf_path TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    year INTEGER,
    doi TEXT,

    CHECK (storage_mode IN ('managed', 'library', 'external'))
);


CREATE TABLE IF NOT EXISTS duplicate_decisions (
    doc_id_a TEXT NOT NULL,
    doc_id_b TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    decided_at TEXT NOT NULL,

    PRIMARY KEY (doc_id_a, doc_id_b),
    FOREIGN KEY (doc_id_a) REFERENCES documents(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (doc_id_b) REFERENCES documents(doc_id) ON DELETE CASCADE,

    CHECK (doc_id_a < doc_id_b),
    CHECK (decision IN ('not_duplicate'))
);

CREATE TABLE IF NOT EXISTS document_pipeline_state (
    doc_id TEXT PRIMARY KEY,
    pipeline_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    message TEXT,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,

    CHECK (status IN ('ok', 'no_extractable_text'))
);

CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    section_index INTEGER,
    section_number TEXT,
    section_title TEXT,
    page_start INTEGER,
    page_end INTEGER,

    PRIMARY KEY (doc_id, chunker_version, chunk_index),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),

    CHECK (chunk_index >= 0),
    CHECK (page_start IS NULL OR page_start >= 1),
    CHECK (page_end IS NULL OR page_end >= 1),
    CHECK (
        page_start IS NULL
        OR page_end IS NULL
        OR page_end >= page_start
    )
);

-- For future Zotero etc integration:
--
-- CREATE TABLE external_references (
--     ref_id TEXT PRIMARY KEY,
--
--     doc_id TEXT NOT NULL,
--
--     system_name TEXT NOT NULL,
--     external_item_id TEXT,
--     external_attachment_id TEXT,
--     external_uri TEXT,
--     metadata_json TEXT,
--
--     FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
--
--     UNIQUE (system_name, external_item_id, external_attachment_id)
-- );

CREATE TABLE IF NOT EXISTS proximity_alerts (
    pa_id TEXT PRIMARY KEY,
    doc_a_id TEXT NOT NULL,
    chunker_a_version TEXT NOT NULL,
    chunk_a_index INTEGER NOT NULL,
    doc_b_id TEXT NOT NULL,
    chunker_b_version TEXT NOT NULL,
    chunk_b_index INTEGER NOT NULL,
    score REAL,
    explanation TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,

    FOREIGN KEY (doc_a_id, chunker_a_version, chunk_a_index)
        REFERENCES chunks(doc_id, chunker_version, chunk_index),

    FOREIGN KEY (doc_b_id, chunker_b_version, chunk_b_index)
        REFERENCES chunks(doc_id, chunker_version, chunk_index),

    CHECK (status IN ('pending', 'confirmed', 'rejected')),

    UNIQUE (
        doc_a_id,
        chunker_a_version,
        chunk_a_index,
        doc_b_id,
        chunker_b_version,
        chunk_b_index
    )
);

CREATE TABLE IF NOT EXISTS confirmed_links (
    cl_id TEXT PRIMARY KEY,
    pa_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    relationship_type TEXT,

    FOREIGN KEY (pa_id) REFERENCES proximity_alerts(pa_id)
);

CREATE TRIGGER IF NOT EXISTS confirmed_links_pa_must_be_confirmed
BEFORE INSERT ON confirmed_links
FOR EACH ROW
BEGIN
    SELECT
        CASE
            WHEN (
                SELECT status
                FROM proximity_alerts
                WHERE pa_id = NEW.pa_id
            ) <> 'confirmed'
            THEN RAISE(ABORT, 'confirmed_links.pa_id must reference a confirmed PA')
        END;
END;
