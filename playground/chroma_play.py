#!/usr/bin/env python

import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
coll = client.get_or_create_collection("test_collection")
docs=[
    "I'm going to call my lawyer!",
    "I need to do some yardwork.",
    "You'll be hearing from an attorney."
]

# This call is idempotent, since it refers to IDs that already exist in the
# collection (hence updating them without error to their current values).
coll.add(
    documents=docs,
    ids=[f"id{n}" for n in range(len(docs))],
)

newdoc = input("Enter new doc: ")
while newdoc != "done":
    res = coll.query(query_texts=[newdoc], n_results=1)
    print(f"Closest: {res['documents'][0][0]}")
    insertyn = input("Insert into collection? (y/n) ")
    if insertyn and insertyn.lower()[0] == 'y':
        coll.add(
            documents=[newdoc],
            ids=[f"id{coll.count()}"]
        )
    newdoc = input("Enter new doc: ")
