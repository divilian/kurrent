
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,

    title TEXT,
    authors TEXT,
    year INTEGER,
    doi TEXT,

    pdf_sha256 TEXT NOT NULL,

    storage_mode TEXT NOT NULL,

    pdf_path TEXT NOT NULL,

    ingested_at TEXT NOT NULL,

    CHECK (storage_mode IN ('managed', 'library', 'external'))
);

CREATE TABLE chunks (
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,

    text TEXT NOT NULL,
    text_sha256 TEXT,

    page_start INTEGER,
    page_end INTEGER,

    PRIMARY KEY (doc_id, chunk_index),

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

CREATE TABLE proximity_alerts (
    pa_id TEXT PRIMARY KEY,

    doc_a_id TEXT NOT NULL,
    chunk_a_index INTEGER NOT NULL,

    doc_b_id TEXT NOT NULL,
    chunk_b_index INTEGER NOT NULL,

    score REAL,
    explanation TEXT,

    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decided_at TEXT,

    FOREIGN KEY (doc_a_id, chunk_a_index)
        REFERENCES chunks(doc_id, chunk_index),

    FOREIGN KEY (doc_b_id, chunk_b_index)
        REFERENCES chunks(doc_id, chunk_index),

    CHECK (
        doc_a_id < doc_b_id
        OR (
            doc_a_id = doc_b_id
            AND chunk_a_index < chunk_b_index
        )
    )

    CHECK (status IN ('pending', 'confirmed', 'rejected')),

    UNIQUE (
        doc_a_id,
        chunk_a_index,
        doc_b_id,
        chunk_b_index
    )
);

CREATE TABLE confirmed_links (
    cl_id TEXT PRIMARY KEY,

    pa_id TEXT NOT NULL UNIQUE,

    relationship_type TEXT,

    created_at TEXT NOT NULL,

    FOREIGN KEY (pa_id) REFERENCES proximity_alerts(pa_id)
);

CREATE TRIGGER confirmed_links_pa_must_be_confirmed
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
