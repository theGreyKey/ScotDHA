"""Text cleaning pipeline for OCR-extracted Scottish legal documents."""

import re
from scottnlp.config import DocumentMeta

# ── Known OCR error corrections (whitelist approach) ───────────────────
# Only correct confirmed OCR errors, NOT archaic spellings
OCR_CORRECTIONS = {
    "tbat": "that",
    "eficient": "efficient",
    "Defcent": "Descent",
    "Scollish": "Scottish",
    "Langudges": "Languages",
    "CHaPtER": "CHAPTER",
    "SUPPoRT": "SUPPORT",
    "fOR": "FOR",
    "LANGuAGE": "LANGUAGE",
    "GaELIC": "GAELIC",
    "ScOtS": "SCOTS",
}

# ── legislation.gov.uk boilerplate patterns (Doc 6) ───────────────────
LEGIS_GOV_PATTERNS = [
    re.compile(r"^Changes to legislation:.*", re.IGNORECASE),
    re.compile(r"^Document Generated:.*"),
    re.compile(r".*View outstanding changes\s*$"),
    re.compile(r"^JINDIPERS!?\s*$"),
    re.compile(r"^Gaelic Language \(Scotland\) Act 2005 asp 7\s*$"),
]

# ── Publisher/copyright block patterns ─────────────────────────────────
PUBLISHER_PATTERNS = [
    re.compile(r"^©\s*Crown\s+Copyright", re.IGNORECASE),
    re.compile(r"^Crown\s+[Cc]opyright"),
    re.compile(r"^King'?s\s+Printer", re.IGNORECASE),
    re.compile(r"^Queen'?s\s+Printer", re.IGNORECASE),
    re.compile(r"^Published by TSO", re.IGNORECASE),
    re.compile(r"^ISBN\s+[\d\-]+"),
    re.compile(r"^www\.tsoshop\.co\.uk"),
    re.compile(r"^TSO\s*$"),
    re.compile(r"^tso\s*$"),
    re.compile(r"^PO Box"),
    re.compile(r"^Telephone orders"),
    re.compile(r"^E-mail:"),
    re.compile(r"^Textphone:"),
    re.compile(r"^a Williams Lea"),
    re.compile(r'^"?\d{6}"?\d{6}'),  # Barcode-like strings
    re.compile(r"^AINDEFENSS\s*$"),
    re.compile(r"^Mail, Telephone"),
    re.compile(r"^Online\s*$"),
    re.compile(r"^customer\.services@"),
]

# ── Textual amendment block patterns (Doc 6) ───────────────────────────
AMENDMENT_PATTERNS = [
    re.compile(r"^Textual Amendments\s*$"),
    re.compile(r"^F\d+\s*$"),
    re.compile(r"^S\.\s+\d+.*(?:inserted|substituted|repealed|omitted).*by\s+"),
    re.compile(r"^Words in s\.\s+\d+.*(?:inserted|substituted|repealed).*by\s+"),
    re.compile(r"^s\.\s+\d+.*(?:inserted|substituted|repealed|word).*by\s+\d{4}"),
]


def clean_document(raw_text: str, meta: DocumentMeta) -> str:
    """Master cleaning function for a single document.

    Steps:
    1. Remove boilerplate lines matching meta.boilerplate_patterns
    2. Remove TOC/front matter (lines before meta.toc_end_line)
    3. Remove publisher info at end of document
    4. Remove legislation.gov.uk repeated headers (Doc 6)
    5. Remove textual amendment annotations (Doc 6)
    6. Fix OCR artifacts using whitelist
    7. Rejoin hyphenated line breaks
    8. Normalize whitespace
    """
    lines = raw_text.split("\n")

    # Step 1: Remove document-specific boilerplate
    if meta.boilerplate_patterns:
        compiled = [re.compile(p) for p in meta.boilerplate_patterns]
        lines = [ln for ln in lines if not any(pat.match(ln) for pat in compiled)]

    # Step 2: Strip TOC/front matter
    if meta.toc_end_line is not None:
        lines = lines[meta.toc_end_line:]

    # Step 3: Remove publisher/copyright blocks at end
    lines = _remove_publisher_block(lines)

    # Step 4: Remove legislation.gov.uk boilerplate (primarily Doc 6)
    if meta.filename.startswith("6"):
        lines = _remove_legis_gov_boilerplate(lines)
        lines = _remove_amendment_blocks(lines)

    # Step 5: Fix OCR artifacts
    lines = [_fix_ocr_line(ln) for ln in lines]

    # Step 6: Rejoin hyphenated line breaks
    text = "\n".join(lines)
    text = _rejoin_split_lines(text)

    # Step 7: Normalize whitespace
    text = _normalize_whitespace(text)

    return text


def _remove_publisher_block(lines: list[str]) -> list[str]:
    """Remove publisher/copyright/ISBN blocks from the end of document."""
    # Scan from the end to find where publisher block starts
    cutoff = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - 40, 0), -1):
        line = lines[i].strip()
        if any(pat.match(line) for pat in PUBLISHER_PATTERNS):
            cutoff = i
    # If we found publisher content, also remove trailing blank lines before it
    if cutoff < len(lines):
        while cutoff > 0 and not lines[cutoff - 1].strip():
            cutoff -= 1
        return lines[:cutoff]
    return lines


def _remove_legis_gov_boilerplate(lines: list[str]) -> list[str]:
    """Remove repeated legislation.gov.uk headers throughout document."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if any(pat.match(line) for pat in LEGIS_GOV_PATTERNS):
            i += 1
            continue
        # Also catch multi-line boilerplate blocks
        if line.startswith("before") and "December 2025" in line:
            i += 1
            continue
        if line.startswith("appear in the content and are referenced"):
            i += 1
            continue
        result.append(lines[i])
        i += 1
    return result


def _remove_amendment_blocks(lines: list[str]) -> list[str]:
    """Remove textual amendment annotation blocks from legislation.gov.uk output."""
    result = []
    in_amendment_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Textual Amendments":
            in_amendment_block = True
            continue
        if in_amendment_block:
            # Amendment blocks contain lines like "F2", "S. 1(2)(aa)..."
            if any(pat.match(stripped) for pat in AMENDMENT_PATTERNS):
                continue
            if re.match(r"^F\d+$", stripped):
                continue
            if re.match(r"^S\.S\.I\.\s+\d+", stripped):
                continue
            if re.match(r"^\d{4}/\d+,\s+reg\.", stripped):
                continue
            # Check if this continues the amendment text (indented or starts with specific patterns)
            if stripped and not stripped[0].isupper() and not stripped[0].isdigit() and not stripped.startswith("("):
                continue
            # End of amendment block
            in_amendment_block = False
        result.append(line)
    return result


def _fix_ocr_line(line: str) -> str:
    """Fix known OCR errors in a single line using whitelist."""
    for wrong, correct in OCR_CORRECTIONS.items():
        if wrong in line:
            line = line.replace(wrong, correct)
    return line


def _rejoin_split_lines(text: str) -> str:
    """Rejoin lines split by OCR mid-word at hyphens.

    Pattern: line ending in hyphen + newline + lowercase continuation.
    Example: 'Interpreta-\\ntion' -> 'Interpretation'
    """
    return re.sub(r"-\n([a-z])", r"\1", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines to single blank line, strip trailing spaces."""
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines to 1
    result = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    # Remove leading/trailing blank lines
    text = "\n".join(result).strip()
    return text
