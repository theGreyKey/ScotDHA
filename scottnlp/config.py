"""Central configuration for ScottNLP project."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCOTLAW_DIR = PROJECT_ROOT / "ScotLaw"

LLAMA_MODEL_PATH = Path(os.environ.get("LLAMA_MODEL_PATH", ""))
LEGAL_BERT_NAME = "nlpaueb/legal-bert-base-uncased"

# ── Hyperparameters ────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = 384
OVERLAP_TOKENS = 64
METADATA_TOKEN_BUDGET = 40
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 768

# ── Phase 3: DeepSeek DHA Classification ──────────────────────────────
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_TEMPERATURE = 0.1
DEEPSEEK_MAX_TOKENS = 2048
DEEPSEEK_RATE_LIMIT_RPS = 10
DEEPSEEK_MAX_RETRIES = 3
DEEPSEEK_RETRY_BASE_DELAY = 1.0

DHA_STRATEGIES = [
    "nomination",
    "predication",
    "argumentation",
    "perspectivization",
    "intensification_mitigation",
]

# ── Era Definitions ───────────────────────────────────────────────────
ERA_DEFINITIONS = {
    "pre-devolution (1707-1997)": [1707, 1872, 1992],
    "devolution (1998-2004)": [1998, 2000],
    "gaelic-revival (2005-2022)": [2005, 2010, 2015],
    "modern (2023-2025)": [2023, 2025],
}
YEAR_TO_ERA = {}
for _era, _years in ERA_DEFINITIONS.items():
    for _y in _years:
        YEAR_TO_ERA[_y] = _era

# ── Language Normalization ────────────────────────────────────────────
# Canonical mapping: variant spelling → standard language name
LANG_CANONICAL_MAP = {
    "english": "English",
    "beurla": "English",
    "gaelic": "Gaelic",
    "gàidhlig": "Gaelic",
    "gaidhlig": "Gaelic",
    "scots": "Scots",
    "albais": "Scots",
    # Case variants used in viz layer
    "Gaelic": "Gaelic",
    "Gàidhlig": "Gaelic",
    "GAELIC": "Gaelic",
    "Scots": "Scots",
    "SCOTS": "Scots",
    "English": "English",
    "Beurla": "English",
    "Scottish": "Scottish",
}


@dataclass
class DocumentMeta:
    """Metadata for a single corpus document."""
    filename: str
    short_title: str
    year: int
    doc_type: str              # "act", "treaty", "report", "policy", "plan"
    jurisdiction: str          # "Scotland", "UK", "European"
    language_focus: list        # Which languages this document primarily concerns
    section_pattern: str        # Regex pattern for section boundaries
    toc_end_line: Optional[int] = None
    boilerplate_patterns: list = field(default_factory=list)


DOCUMENTS = [
    DocumentMeta(
        filename="1articles of union1707.txt",
        short_title="Articles of Union",
        year=1707,
        doc_type="treaty",
        jurisdiction="UK",
        language_focus=["English", "Scots"],
        section_pattern=r"^([IVXLC]+)\.\s",
        toc_end_line=None,
    ),
    DocumentMeta(
        filename="2Education (Scotland) Act1872.txt",
        short_title="Education (Scotland) Act",
        year=1872,
        doc_type="act",
        jurisdiction="Scotland",
        language_focus=["English"],
        section_pattern=r"^\d+\.\s",
    ),
    DocumentMeta(
        filename="3European_Charter_for_Regional_or_Minority_Languages 1992.txt",
        short_title="European Charter for Regional or Minority Languages",
        year=1992,
        doc_type="treaty",
        jurisdiction="European",
        language_focus=["Gaelic", "Scots", "English"],
        section_pattern=r"^Article\s+\d+",
    ),
    DocumentMeta(
        filename="4Scotland Act 1998.txt",
        short_title="Scotland Act",
        year=1998,
        doc_type="act",
        jurisdiction="UK",
        language_focus=["English"],
        section_pattern=r"^\d+\s",
        toc_end_line=389,
    ),
    DocumentMeta(
        filename="5Standards in Scotland.txt",
        short_title="Standards in Scotland's Schools etc. Act",
        year=2000,
        doc_type="act",
        jurisdiction="Scotland",
        language_focus=["English", "Gaelic"],
        section_pattern=r"^\d+\s",
        toc_end_line=50,
    ),
    DocumentMeta(
        filename="6Gaelic Language (Scotland) Act 2005.txt",
        short_title="Gaelic Language (Scotland) Act",
        year=2005,
        doc_type="act",
        jurisdiction="Scotland",
        language_focus=["Gaelic", "English"],
        section_pattern=r"^\d+\s",
        boilerplate_patterns=[
            r"^Changes to legislation:.*",
            r"^Document Generated:.*",
            r".*View outstanding changes$",
            r"^JINDIPERS!?$",
        ],
    ),
    DocumentMeta(
        filename="7Report of Ministerial Working Group on the Scots Language 2010.txt",
        short_title="Ministerial Working Group Report on Scots Language",
        year=2010,
        doc_type="report",
        jurisdiction="Scotland",
        language_focus=["Scots", "English"],
        section_pattern=r"^(Chairman|Executive|Discussions|Appendix)",
        toc_end_line=59,
    ),
    DocumentMeta(
        filename="8scots language policy2015.txt",
        short_title="Scots Language Policy",
        year=2015,
        doc_type="policy",
        jurisdiction="Scotland",
        language_focus=["Scots", "English"],
        section_pattern=r"^(Policy Context|Manifesto|Council of Europe|Aims|Actions)",
    ),
    DocumentMeta(
        filename="9National_Gaelic_Language_Plan_ENGLISH2023-28.txt",
        short_title="National Gaelic Language Plan 2023-28",
        year=2023,
        doc_type="plan",
        jurisdiction="Scotland",
        language_focus=["Gaelic", "English"],
        section_pattern=r"^\d+\.\s",
        toc_end_line=36,
    ),
    DocumentMeta(
        filename="10Scottish Languages Act 2025.txt",
        short_title="Scottish Languages Act",
        year=2025,
        doc_type="act",
        jurisdiction="Scotland",
        language_focus=["Gaelic", "Scots", "English"],
        section_pattern=r"^\d+\s",
        toc_end_line=163,
        boilerplate_patterns=[
            r"^AINDEFENSS$",
            r"^tso$",
        ],
    ),
]


def get_doc_meta(filename: str) -> DocumentMeta:
    """Look up metadata by filename."""
    for doc in DOCUMENTS:
        if doc.filename == filename:
            return doc
    raise ValueError(f"No metadata registered for: {filename}")
