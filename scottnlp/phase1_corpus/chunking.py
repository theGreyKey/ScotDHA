"""Structure-aware text chunking with Llama-assisted and rule-based modes."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from scottnlp.config import (
    DocumentMeta,
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    METADATA_TOKEN_BUDGET,
    LEGAL_BERT_NAME,
    LLAMA_MODEL_PATH,
)


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    chunk_id: str
    doc_filename: str
    doc_title: str
    doc_year: int
    doc_type: str
    jurisdiction: str
    section_id: str
    section_title: str
    chunk_index: int
    total_chunks_in_section: int
    text: str
    text_with_prefix: str
    start_line: int
    end_line: int
    token_count: int
    language_focus: list

    def to_dict(self) -> dict:
        return asdict(self)


# ── Tokenizer (lazy-loaded) ───────────────────────────────────────────
_bert_tokenizer = None


def _get_bert_tokenizer():
    global _bert_tokenizer
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained(LEGAL_BERT_NAME)
    return _bert_tokenizer


def _count_tokens(text: str) -> int:
    tok = _get_bert_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ── Section splitting (rule-based) ────────────────────────────────────

def split_into_sections(text: str, section_pattern: str) -> list[dict]:
    """Split document text at section boundaries using regex.

    Returns list of dicts with section_id, section_title, text, start_line, end_line.
    """
    lines = text.split("\n")
    pattern = re.compile(section_pattern)
    sections = []
    current = None

    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            if current is not None:
                current["end_line"] = i - 1
                current["text"] = "\n".join(lines[current["start_line"]:i]).strip()
                if current["text"]:
                    sections.append(current)
            section_id = match.group(0).strip()
            # Try to get a title from the same or next line
            title_text = line.strip()[len(section_id):].strip()
            if not title_text and i + 1 < len(lines):
                title_text = lines[i + 1].strip()
            current = {
                "section_id": section_id,
                "section_title": title_text[:100] if title_text else section_id,
                "start_line": i,
                "end_line": i,
                "text": "",
            }
        elif current is None and line.strip():
            # Text before first section match — create a preamble section
            if not sections and not any(s.get("section_id") == "preamble" for s in sections):
                current = {
                    "section_id": "preamble",
                    "section_title": "Preamble",
                    "start_line": i,
                    "end_line": i,
                    "text": "",
                }

    # Close the last section
    if current is not None:
        current["end_line"] = len(lines) - 1
        current["text"] = "\n".join(lines[current["start_line"]:]).strip()
        if current["text"]:
            sections.append(current)

    # If no sections found, treat entire document as one section
    if not sections:
        sections = [{
            "section_id": "full",
            "section_title": "Full Document",
            "start_line": 0,
            "end_line": len(lines) - 1,
            "text": text.strip(),
        }]

    return sections


# ── Sentence-level sub-splitting ──────────────────────────────────────

def _split_by_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based approach.

    Uses common sentence boundary patterns to avoid loading spaCy for chunking.
    """
    # Split on sentence boundaries: period/question/exclamation followed by space and uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\("])', text)
    # Also split on newline followed by ( for sub-clauses in legal text
    result = []
    for sent in sentences:
        # Further split on double newlines
        parts = sent.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


def split_section_by_tokens(
    section_text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split an over-long section into token-budgeted sub-chunks.

    Uses sentence boundaries where possible, falls back to token-level split.
    """
    sentences = _split_by_sentences(section_text)
    if not sentences:
        return [section_text] if section_text.strip() else []

    chunks = []
    current_sents = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if sent_tokens > max_tokens:
            # Single sentence too long — flush current and force-split this sentence
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents = []
                current_tokens = 0
            # Token-level split
            words = sent.split()
            buf = []
            buf_tokens = 0
            for w in words:
                w_tokens = _count_tokens(w)
                if buf_tokens + w_tokens > max_tokens and buf:
                    chunks.append(" ".join(buf))
                    # Overlap: keep last few words
                    overlap_words = []
                    ot = 0
                    for ow in reversed(buf):
                        owt = _count_tokens(ow)
                        if ot + owt > overlap_tokens:
                            break
                        overlap_words.insert(0, ow)
                        ot += owt
                    buf = overlap_words
                    buf_tokens = ot
                buf.append(w)
                buf_tokens += w_tokens
            if buf:
                chunks.append(" ".join(buf))
            continue

        if current_tokens + sent_tokens > max_tokens and current_sents:
            chunks.append(" ".join(current_sents))
            # Overlap: keep last sentences that fit in overlap budget
            overlap_sents = []
            ot = 0
            for os_ in reversed(current_sents):
                ost = _count_tokens(os_)
                if ot + ost > overlap_tokens:
                    break
                overlap_sents.insert(0, os_)
                ot += ost
            current_sents = overlap_sents
            current_tokens = ot

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


# ── Metadata prefix injection ─────────────────────────────────────────

def inject_metadata_prefix(chunk_text: str, meta: DocumentMeta, section_id: str) -> str:
    """Prepend structured metadata prefix to chunk text."""
    return f"[DOC: {meta.short_title} ({meta.year}) | SEC: {section_id}] {chunk_text}"


# ── Llama-assisted chunking ───────────────────────────────────────────

_llama_model = None
_llama_tokenizer = None


def _load_llama(device: str = "cuda:1"):
    """Load Llama-3.1-8B-Instruct for intelligent chunking."""
    global _llama_model, _llama_tokenizer
    if _llama_model is not None:
        return _llama_model, _llama_tokenizer

    from transformers import AutoModelForCausalLM

    print(f"Loading Llama-3.1-8B-Instruct on {device}...")
    _llama_tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_MODEL_PATH))
    _llama_model = AutoModelForCausalLM.from_pretrained(
        str(LLAMA_MODEL_PATH),
        torch_dtype=torch.float16,
        device_map=device,
    )
    _llama_model.eval()
    print("Llama loaded.")
    return _llama_model, _llama_tokenizer


def _llama_identify_sections(text: str, meta: DocumentMeta, device: str = "cuda:1") -> list[dict]:
    """Use Llama to identify logical section boundaries in a document.

    Sends text in sliding windows and asks Llama to identify section breaks.
    Falls back to rule-based splitting on any failure.
    """
    model, tokenizer = _load_llama(device)

    # For the Llama context, we use a sliding window of ~2000 tokens
    window_size = 2000
    lines = text.split("\n")

    prompt_template = """You are analyzing a legal document: "{title}" ({year}).
Identify logical section boundaries in the following text excerpt.
For each section, output a JSON line with: {{"line_num": <start_line>, "section_id": "<id>", "title": "<title>"}}
Only output JSON lines, nothing else.

Text (starting at line {start_line}):
---
{text_window}
---"""

    all_boundaries = []
    i = 0
    step = 60  # Process ~60 lines at a time

    while i < len(lines):
        window_lines = lines[i:i + step]
        window_text = "\n".join(f"L{i + j}: {ln}" for j, ln in enumerate(window_lines))

        prompt = prompt_template.format(
            title=meta.short_title,
            year=meta.year,
            start_line=i,
            text_window=window_text,
        )

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Parse JSON lines from response
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    boundary = json.loads(line)
                    if "line_num" in boundary:
                        boundary["line_num"] = int(boundary["line_num"])
                        all_boundaries.append(boundary)
                except (json.JSONDecodeError, ValueError):
                    continue

        i += step

    if not all_boundaries:
        return []

    # Convert boundaries to sections
    all_boundaries.sort(key=lambda x: x["line_num"])
    # Deduplicate boundaries that are too close
    deduped = [all_boundaries[0]]
    for b in all_boundaries[1:]:
        if b["line_num"] - deduped[-1]["line_num"] >= 3:
            deduped.append(b)

    sections = []
    for idx, boundary in enumerate(deduped):
        start = boundary["line_num"]
        end = deduped[idx + 1]["line_num"] - 1 if idx + 1 < len(deduped) else len(lines) - 1
        section_text = "\n".join(lines[start:end + 1]).strip()
        if section_text:
            sections.append({
                "section_id": boundary.get("section_id", f"sec_{idx + 1}"),
                "section_title": boundary.get("title", f"Section {idx + 1}"),
                "start_line": start,
                "end_line": end,
                "text": section_text,
            })

    return sections


# ── Main chunking entry point ─────────────────────────────────────────

def chunk_document(
    cleaned_text: str,
    meta: DocumentMeta,
    use_llama: bool = True,
    llama_device: str = "cuda:1",
) -> list[Chunk]:
    """Chunk a single cleaned document into embedding-ready pieces.

    Strategy:
    1. Try Llama-assisted section identification (if use_llama=True)
    2. Fall back to rule-based regex splitting
    3. Sub-split oversized sections by sentence boundaries
    4. Inject metadata prefix for each chunk
    """
    max_tokens = MAX_CHUNK_TOKENS - METADATA_TOKEN_BUDGET

    # Step 1: Get sections
    sections = []
    if use_llama:
        try:
            sections = _llama_identify_sections(cleaned_text, meta, device=llama_device)
            if sections:
                print(f"  Llama found {len(sections)} sections in {meta.short_title}")
        except Exception as e:
            print(f"  Llama failed for {meta.short_title}: {e}, using rule-based")

    # Step 2: Fallback to rule-based
    if not sections:
        sections = split_into_sections(cleaned_text, meta.section_pattern)
        print(f"  Rule-based: {len(sections)} sections in {meta.short_title}")

    # Step 3: Sub-split and create chunks
    chunks = []
    doc_prefix = f"doc{DOCUMENTS_INDEX.get(meta.filename, 0):02d}"

    for sec in sections:
        sec_text = sec["text"]
        sec_tokens = _count_tokens(sec_text)

        if sec_tokens <= max_tokens:
            sub_chunks = [sec_text]
        else:
            sub_chunks = split_section_by_tokens(sec_text, max_tokens, OVERLAP_TOKENS)

        for ci, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{doc_prefix}_{sec['section_id']}_{ci:03d}"
            # Clean chunk_id for filesystem safety
            chunk_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", chunk_id)

            text_with_prefix = inject_metadata_prefix(chunk_text, meta, sec["section_id"])

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_filename=meta.filename,
                doc_title=meta.short_title,
                doc_year=meta.year,
                doc_type=meta.doc_type,
                jurisdiction=meta.jurisdiction,
                section_id=sec["section_id"],
                section_title=sec.get("section_title", sec["section_id"]),
                chunk_index=ci,
                total_chunks_in_section=len(sub_chunks),
                text=chunk_text,
                text_with_prefix=text_with_prefix,
                start_line=sec["start_line"],
                end_line=sec["end_line"],
                token_count=_count_tokens(text_with_prefix),
                language_focus=meta.language_focus,
            ))

    # Update total_chunks_in_section now that we know the count
    sec_counts = {}
    for c in chunks:
        key = c.section_id
        sec_counts[key] = sec_counts.get(key, 0) + 1
    for c in chunks:
        c.total_chunks_in_section = sec_counts[c.section_id]

    return chunks


# Build filename -> index lookup
from scottnlp.config import DOCUMENTS
DOCUMENTS_INDEX = {doc.filename: i + 1 for i, doc in enumerate(DOCUMENTS)}


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Save chunks as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
