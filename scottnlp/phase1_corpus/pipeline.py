"""Phase 1 pipeline orchestrator: clean -> chunk -> embed."""

from pathlib import Path

from scottnlp.config import DOCUMENTS, DATA_DIR, OUTPUT_DIR
from scottnlp.phase1_corpus.cleaning import clean_document
from scottnlp.phase1_corpus.chunking import chunk_document, save_chunks, load_chunks, Chunk
from scottnlp.phase1_corpus.embedding import generate_embeddings, save_embeddings, load_embeddings


PHASE1_DIR = OUTPUT_DIR / "phase1"
CLEANED_DIR = PHASE1_DIR / "cleaned"


def run_phase1(
    data_dir: Path = DATA_DIR,
    output_dir: Path = PHASE1_DIR,
    device: str = "cuda:0",
    use_llama: bool = True,
    llama_device: str = "cuda:1",
    force_reclean: bool = False,
    force_rechunk: bool = False,
    force_reembed: bool = False,
) -> dict:
    """Execute the complete Phase 1 pipeline.

    Steps:
    1. Clean each document (save to output_dir/cleaned/)
    2. Chunk all documents (save chunks.jsonl)
    3. Generate embeddings (save embeddings.npy + metadata)

    Returns summary statistics dict.
    """
    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Clean ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Cleaning documents")
    print("=" * 60)
    for doc in DOCUMENTS:
        cleaned_path = cleaned_dir / doc.filename
        if cleaned_path.exists() and not force_reclean:
            print(f"  [skip] {doc.short_title} (already cleaned)")
            continue

        raw = (data_dir / doc.filename).read_text(encoding="utf-8")
        cleaned = clean_document(raw, doc)
        cleaned_path.write_text(cleaned, encoding="utf-8")

        raw_lines = len(raw.split("\n"))
        clean_lines = len(cleaned.split("\n"))
        print(f"  [done] {doc.short_title}: {raw_lines} -> {clean_lines} lines")

    # ── Step 2: Chunk ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Chunking documents")
    print("=" * 60)
    chunks_path = output_dir / "chunks.jsonl"

    if chunks_path.exists() and not force_rechunk:
        print(f"  [skip] Chunks already exist at {chunks_path}")
        all_chunks_dicts = load_chunks(chunks_path)
        print(f"  Loaded {len(all_chunks_dicts)} existing chunks")
    else:
        all_chunks = []
        for doc in DOCUMENTS:
            cleaned_path = cleaned_dir / doc.filename
            cleaned = cleaned_path.read_text(encoding="utf-8")
            chunks = chunk_document(cleaned, doc, use_llama=use_llama, llama_device=llama_device)
            all_chunks.extend(chunks)
            print(f"  {doc.year} {doc.short_title[:40]:40s} -> {len(chunks)} chunks")

        save_chunks(all_chunks, chunks_path)
        all_chunks_dicts = [c.to_dict() for c in all_chunks]

    # ── Step 3: Embed ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating embeddings")
    print("=" * 60)
    emb_path = output_dir / "embeddings.npy"

    if emb_path.exists() and not force_reembed:
        print(f"  [skip] Embeddings already exist at {emb_path}")
        embeddings, _ = load_embeddings(output_dir)
    else:
        embeddings = generate_embeddings(
            all_chunks_dicts,
            device=device,
        )
        save_embeddings(embeddings, all_chunks_dicts, output_dir)

    # ── Summary ────────────────────────────────────────────────────
    chunks_per_doc = {}
    tokens_per_doc = {}
    for c in all_chunks_dicts:
        title = c["doc_title"]
        chunks_per_doc[title] = chunks_per_doc.get(title, 0) + 1
        tokens_per_doc[title] = tokens_per_doc.get(title, 0) + c["token_count"]

    total_tokens = sum(c["token_count"] for c in all_chunks_dicts)

    summary = {
        "num_documents": len(DOCUMENTS),
        "num_chunks": len(all_chunks_dicts),
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": total_tokens / len(all_chunks_dicts) if all_chunks_dicts else 0,
        "embedding_shape": tuple(embeddings.shape),
        "chunks_per_document": chunks_per_doc,
        "tokens_per_document": tokens_per_doc,
    }

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"  Documents: {summary['num_documents']}")
    print(f"  Total chunks: {summary['num_chunks']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Avg tokens/chunk: {summary['avg_tokens_per_chunk']:.1f}")
    print(f"  Embedding shape: {summary['embedding_shape']}")
    print(f"\n  Chunks per document:")
    for title, count in chunks_per_doc.items():
        print(f"    {title[:45]:45s} {count:4d} chunks, {tokens_per_doc[title]:6d} tokens")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 1 pipeline")
    parser.add_argument("--no-llama", action="store_true", help="Skip Llama, use rule-based chunking only")
    parser.add_argument("--device", default="cuda:0", help="Device for Legal-BERT embedding")
    parser.add_argument("--llama-device", default="cuda:1", help="Device for Llama model")
    parser.add_argument("--force-reclean", action="store_true")
    parser.add_argument("--force-rechunk", action="store_true")
    parser.add_argument("--force-reembed", action="store_true")
    args = parser.parse_args()

    run_phase1(
        use_llama=not args.no_llama,
        device=args.device,
        llama_device=args.llama_device,
        force_reclean=args.force_reclean,
        force_rechunk=args.force_rechunk,
        force_reembed=args.force_reembed,
    )
