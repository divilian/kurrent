#!/home/stephen/venvs/kurrent/bin/python
#
# Main command-line interface to kurrent.

import argparse
import shutil
import textwrap

from transformers import AutoTokenizer
import polars as pl


def print_wrapped(s: str, width: int | None = None) -> None:
    # Use terminal width if available; otherwise default to 79.
    if width is None:
        width = shutil.get_terminal_size(fallback=(79, 20)).columns

    # Replace existing newlines only if you want a single flowing paragraph.
    s = " ".join(s.splitlines())

    wrapped = textwrap.fill(
        s,
        width=width,
        break_long_words=False,   # don't split words
        break_on_hyphens=False,   # don't split at hyphens
    )
    print(wrapped)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to .csv or .txt file to pre-process."
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        help="Number of chunks to use for each retrieval.",
        default=5
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size of text chunks (in # of tokens).",
        default=1024
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Short phrase indicating what topic the AI is allowed to answer.",
        default=None
    )
    parser.add_argument(
        "--embed-model",
        # Use "sentence-transformers/all-mpnet-base-v2" for slower & better.
        # (768-dim) The default "sentence-transformers/all-MiniLM-L6-v2" has
        # 384-dim vectors.
        choices=["all-mpnet-base-v2","all-MiniLM-L6-v2"],
        default="all-MiniLM-L6-v2",
        help="Which embedding model to use?",
    )
    # Use "sentence-transformers/all-mpnet-base-v2" for slower & better.
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/stephen/local/rag/cache",
        help="Path to dir in which to cache embeddings files.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    llm = LocalLlamaBackend(
        model_path=f"{LLAMA_DIR}/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        llama_cli=f"{LLAMA_DIR}/build/bin/llama-cli",
        n_gpu_layers=10,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        use_fast=True,
    )

    embed_model = get_embed_model(args.embed_model)

    cache_file = get_cache_filepath(
        args.file_path,
        args.embed_model,
        args.chunk_size,
    )

    if cache_file.is_file():
        print("Loading cached embeddings...")
        df = pl.read_parquet(cache_file)
    else:
        path = args.file_path.lower()
        if path.endswith(".csv"):
            num_docs, chunks = load_csv_chunks(args.file_path, args.chunk_size)
        elif path.endswith(".txt"):
            num_docs, chunks = load_txt_chunks(args.file_path, args.chunk_size)
        elif path.endswith(".pdf"):
            num_docs, chunks = load_pdf_chunks(args.file_path, args.chunk_size)
        else:
            raise ValueError("Input file must be .csv, .txt, or .pdf")

        df = pl.DataFrame({'chunk':chunks})

        df = generate_embeddings(embed_model, df)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_file)


    question = input("Ask a question (or 'done'): ")
    while question != "done":

        best_doc_indices = find_best_docs(
            embed_model,
            question,
            df['embedding'].to_list(),
            args.num_docs,
        )
        context="".join(df['chunk'][best_doc_indices])

        answer = perform_rag_query(llm, context, question, args.topic)
        print_wrapped(answer)

        question = input("\nAsk a question (or 'done'): ")
