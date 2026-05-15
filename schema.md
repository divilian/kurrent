# kurrent "schema decisions" (both Chroma and SQLite)

## SQLite

### Documents

We don't store the actual content here; that's external, on disk (either in
Zotero storage, kurrent-managed PDF storage, or something else.) By contrast,
`documents` is about identity, location, bibliographic metadata, and ingestion
state.

```
documents(
    doc_id uuid,
    title text,
    authors text,
    year integer,
    doi text,
    pdf_sha256 text,
    storage_mode managed|library|external,
    pdf_path text,
    ingested_at datetime
)
```

#### External references

When Zotero, Mendeley, etc, integration is implemented, each document whose
`storage_mode` is 'library' will have a corresponding row in an
`external_references` table, specifying how to identify/retrieve it from that
external tool.

### Chunks

We _do_ store the actual chunk text here.

```
chunks(
    doc_id uuid,
    chunk_index integer,
    page_start integer,
    page_end integer,
    text_sha256 text,
    text text
)
```

### Proximity alerts (PAs)

Note that a PA row is permanent, even if confirmed or rejected. The CL, if one
is created, points to it durably.

```
proximity_alerts(
    pa_id uuid,
    doc_a_id uuid,
    chunk_a_index integer,
    doc_b_id uuid,
    chunk_b_index integer,
    score real,
    explanation text,
    status pending|confirmed|rejected,
    created_at datetime,
    decided_at datetime
)
```

### Confirmed Links (CLs)

```
confirmed_links (
    cl_id uuid,
    pa_id uuid,
    relationship_type 
        same_claim|
        same_method|
        same_dataset|
        same_theory|
        supporting_evidence|
        contrasting_claim|
        shared_citation|
        shared_background|
        definition_or_concept|
        application_or_example|
        general_topical_overlap,
    created_at datetime
)
```
