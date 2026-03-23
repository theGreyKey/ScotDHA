"""spaCy dependency parsing around language-related terms."""

import json
import os
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy


# Target terms for the DHA language analysis
TARGET_TERMS = [
    # Language names
    "English", "Gaelic", "Scots", "Scottish",
    # Gaelic-language forms
    "Gàidhlig", "Gaidhlig", "Beurla", "Albais",
    # Generic language-related terms
    "language", "tongue", "dialect", "speech",
    # Institutional / policy terms
    "bilingual", "multilingual", "monolingual",
    "Bòrd",  # Bòrd na Gàidhlig — key institutional actor
]
# Expanded matching (case-insensitive lemma forms)
TARGET_LEMMAS = {t.lower() for t in TARGET_TERMS}


def _extract_frames_from_doc(doc, chunk: dict, target_lower: set) -> list[dict]:
    """Extract dependency frames from a parsed spaCy doc."""
    frames = []
    for sent in doc.sents:
        for token in sent:
            if token.text.lower() not in target_lower and token.lemma_.lower() not in target_lower:
                continue

            head = token.head
            children = list(token.children)
            subtree = list(token.subtree)
            subtree_text = " ".join(t.text for t in subtree)

            role = _classify_syntactic_role(token)

            modifiers = [
                {"text": c.text, "lemma": c.lemma_, "pos": c.pos_, "dep": c.dep_}
                for c in children
                if c.pos_ in ("ADJ", "ADV") or c.dep_ in ("amod", "advmod", "compound")
            ]

            gov_verb = _find_governing_verb(token)

            frames.append({
                "chunk_id": chunk["chunk_id"],
                "doc_year": chunk["doc_year"],
                "doc_title": chunk["doc_title"],
                "target_term": token.text,
                "target_lemma": token.lemma_.lower(),
                "sentence": sent.text,
                "dep_head": head.text,
                "dep_head_lemma": head.lemma_,
                "dep_head_pos": head.pos_,
                "dep_relation": token.dep_,
                "dep_children": [
                    {"text": c.text, "dep": c.dep_, "pos": c.pos_}
                    for c in children
                ],
                "subtree_text": subtree_text,
                "syntactic_role": role,
                "modifiers": modifiers,
                "governing_verb": gov_verb,
                "token_idx": token.i,
            })
    return frames


def _worker(args):
    """Worker function for multi-GPU parallel processing."""
    gpu_id, chunk_subset, target_terms = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # spaCy trf uses PyTorch which respects CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.set_device(0)  # device 0 within the remapped visible set
    nlp = spacy.load("en_core_web_trf")

    target_lower = {t.lower() for t in target_terms}
    frames = []
    texts = [c["text"] for c in chunk_subset]

    for ci, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        if ci % 100 == 0:
            print(f"  [GPU {gpu_id}] Parsing chunk {ci + 1}/{len(chunk_subset)}...")
        chunk_frames = _extract_frames_from_doc(doc, chunk_subset[ci], target_lower)
        frames.extend(chunk_frames)

    print(f"  [GPU {gpu_id}] Done. Extracted {len(frames)} frames from {len(chunk_subset)} chunks.")
    return frames


def extract_language_dependency_frames(
    chunks: list[dict],
    target_terms: list[str] = TARGET_TERMS,
    nlp=None,
    gpu_ids: list[int] | None = None,
) -> list[dict]:
    """Parse dependency structures around language-related terms.

    Uses multiple GPUs in parallel via multiprocessing. Falls back to single
    GPU if only one is available or nlp is pre-loaded.
    """
    # Single-GPU path (backward compatible)
    if nlp is not None or (gpu_ids is not None and len(gpu_ids) <= 1):
        return _extract_single_gpu(chunks, target_terms, nlp)

    # Detect available GPUs
    if gpu_ids is None:
        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            return _extract_single_gpu(chunks, target_terms, nlp)
        gpu_ids = list(range(n_gpus))

    print(f"Using {len(gpu_ids)} GPUs for parallel dependency parsing: {gpu_ids}")

    # Split chunks evenly across GPUs
    n = len(chunks)
    chunk_splits = []
    for i, gid in enumerate(gpu_ids):
        start = i * n // len(gpu_ids)
        end = (i + 1) * n // len(gpu_ids)
        chunk_splits.append((gid, chunks[start:end], target_terms))

    # Use spawn to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(gpu_ids)) as pool:
        results = pool.map(_worker, chunk_splits)

    frames = []
    for r in results:
        frames.extend(r)

    print(f"Extracted {len(frames)} dependency frames for {len(target_terms)} target terms")
    return frames


def _extract_single_gpu(
    chunks: list[dict],
    target_terms: list[str],
    nlp=None,
) -> list[dict]:
    """Single-GPU extraction using nlp.pipe for batch efficiency."""
    if nlp is None:
        print("Loading spaCy model: en_core_web_trf...")
        nlp = spacy.load("en_core_web_trf")
        print("spaCy loaded.")

    target_lower = {t.lower() for t in target_terms}
    frames = []
    texts = [c["text"] for c in chunks]

    for ci, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        if ci % 100 == 0:
            print(f"  Parsing chunk {ci + 1}/{len(chunks)}...")
        chunk_frames = _extract_frames_from_doc(doc, chunks[ci], target_lower)
        frames.extend(chunk_frames)

    print(f"Extracted {len(frames)} dependency frames for {len(target_terms)} target terms")
    return frames


def _classify_syntactic_role(token) -> str:
    """Classify the syntactic role of a token.

    Returns one of: 'agent', 'patient', 'oblique', 'modifier', 'other'
    """
    dep = token.dep_
    if dep in ("nsubj", "nsubjpass", "csubj"):
        if dep == "nsubjpass":
            return "patient"
        if token.head.tag_ and "Pass" in token.head.morph.get("Voice", []):
            return "patient"
        return "agent"
    elif dep in ("dobj", "obj", "iobj", "pobj"):
        return "patient"
    elif dep in ("pcomp", "prep", "obl"):
        return "oblique"
    elif dep in ("amod", "nmod", "compound", "appos"):
        return "modifier"
    else:
        return "other"


def _find_governing_verb(token) -> dict | None:
    """Walk up the dependency tree to find the governing verb."""
    current = token
    visited = set()
    while current.head != current and current.i not in visited:
        visited.add(current.i)
        current = current.head
        if current.pos_ == "VERB":
            return {
                "text": current.text,
                "lemma": current.lemma_,
                "tag": current.tag_,
            }
    return None


def build_predicate_inventory(dep_frames: list[dict]) -> dict:
    """Build inventory of predicates associated with each language term.

    Groups by (target_term, doc_year) and extracts:
    - Most common governing verbs
    - Most common modifiers
    - Agent vs. patient ratio (agency analysis)

    Returns nested dict: term -> year -> {verbs, modifiers, agency_ratio}
    """
    inventory = {}

    for frame in dep_frames:
        term = frame["target_term"]
        year = frame["doc_year"]
        key = (term, year)

        if key not in inventory:
            inventory[key] = {
                "verbs": [],
                "modifiers": [],
                "roles": [],
            }

        if frame["governing_verb"]:
            inventory[key]["verbs"].append(frame["governing_verb"]["lemma"])
        for mod in frame["modifiers"]:
            inventory[key]["modifiers"].append(mod["lemma"])
        inventory[key]["roles"].append(frame["syntactic_role"])

    # Aggregate
    result = {}
    for (term, year), data in inventory.items():
        if term not in result:
            result[term] = {}

        verb_counts = Counter(data["verbs"]).most_common(10)
        mod_counts = Counter(data["modifiers"]).most_common(10)
        role_counts = Counter(data["roles"])

        agent_count = role_counts.get("agent", 0)
        patient_count = role_counts.get("patient", 0)
        agency_ratio = agent_count / (agent_count + patient_count) if (agent_count + patient_count) > 0 else 0.0

        result[term][year] = {
            "top_verbs": [{"verb": v, "count": c} for v, c in verb_counts],
            "top_modifiers": [{"modifier": m, "count": c} for m, c in mod_counts],
            "role_distribution": dict(role_counts),
            "agency_ratio": agency_ratio,
            "total_occurrences": sum(role_counts.values()),
        }

    return result


def save_dependency_results(
    dep_frames: list[dict],
    predicate_inventory: dict,
    output_dir: Path,
) -> None:
    """Save dependency parsing results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dependency frames as JSONL
    frames_path = output_dir / "dep_frames.jsonl"
    with open(frames_path, "w", encoding="utf-8") as f:
        for frame in dep_frames:
            f.write(json.dumps(frame, ensure_ascii=False) + "\n")
    print(f"Saved {len(dep_frames)} dependency frames to {frames_path}")

    # Predicate inventory
    inv_path = output_dir / "predicate_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(predicate_inventory, f, indent=2, ensure_ascii=False)
    print(f"Saved predicate inventory to {inv_path}")

    # Summary table: agency ratio per term per year
    rows = []
    for term, years in predicate_inventory.items():
        for year, data in years.items():
            rows.append({
                "term": term,
                "year": year,
                "agency_ratio": data["agency_ratio"],
                "total_occurrences": data["total_occurrences"],
                "top_verb": data["top_verbs"][0]["verb"] if data["top_verbs"] else None,
                "top_modifier": data["top_modifiers"][0]["modifier"] if data["top_modifiers"] else None,
            })
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / "agency_summary.csv", index=False)
        print(f"Saved agency summary to {output_dir / 'agency_summary.csv'}")
