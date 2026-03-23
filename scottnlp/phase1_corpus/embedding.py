"""Legal-BERT embedding generation for text chunks."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scottnlp.config import LEGAL_BERT_NAME, EMBEDDING_BATCH_SIZE


def generate_embeddings(
    chunks: list,
    model_name: str = LEGAL_BERT_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    device: str = "cuda:0",
    show_progress: bool = True,
) -> np.ndarray:
    """Generate Legal-BERT embeddings for all chunks.

    Uses sentence-transformers to encode chunk.text_with_prefix.
    Returns np.ndarray of shape (num_chunks, 768).
    """
    model = SentenceTransformer(model_name, device=device)

    # Extract texts to encode
    texts = []
    for c in chunks:
        if hasattr(c, "text_with_prefix"):
            texts.append(c.text_with_prefix)
        else:
            texts.append(c["text_with_prefix"])

    print(f"Encoding {len(texts)} chunks with {model_name} on {device}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    chunks: list,
    output_dir: Path,
) -> None:
    """Save embeddings and chunk metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}")

    # Save chunk metadata as JSONL
    chunks_path = output_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            d = c.to_dict() if hasattr(c, "to_dict") else c
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved chunk metadata to {chunks_path}")

    # Save index mapping chunk_id -> row index
    index_path = output_dir / "embedding_index.json"
    index = {}
    for i, c in enumerate(chunks):
        cid = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
        index[cid] = i
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved embedding index to {index_path}")


def load_embeddings(output_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Load previously saved embeddings and chunk metadata."""
    embeddings = np.load(output_dir / "embeddings.npy")

    chunks = []
    with open(output_dir / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return embeddings, chunks
