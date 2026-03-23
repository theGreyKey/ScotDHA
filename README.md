# ScottNLP

Computational Discourse-Historical Analysis (DHA) of Scottish language policy, 1707--2025.

A four-phase NLP pipeline that traces how English, Gaelic, and Scots are constructed in 318 years of Scottish legal and policy discourse, combining Legal-BERT embeddings, BERTopic topic modeling, spaCy dependency parsing, and LLM-based discourse strategy classification.

## Pipeline Overview

```
data/*.txt  ──>  Phase 1: Clean / Chunk / Embed
                         │
              ┌──────────┴──────────┐
              v                     v
     Phase 2: Topics,        Phase 3: DHA Strategy
     Dep. Parsing,           Classification
     Semantic Networks        (DeepSeek LLM)
              │                     │
              └──────────┬──────────┘
                         v
              Phase 4: Visualization
```

Phases 2 and 3 are independent and can run in parallel after Phase 1 completes.

## Requirements

- Python 3.11+
- GPU: 1+ NVIDIA GPU with CUDA support (tested on 4x RTX 3090)
- [DeepSeek API key](https://platform.deepseek.com/) for Phase 3

### Install dependencies

```bash
pip install -r requirements.txt
pip install bertopic hdbscan umap-learn python-dotenv tqdm matplotlib seaborn
```

### Download models

```bash
python download_models.py
```

This downloads:
- **spaCy** `en_core_web_trf` (~460 MB)
- **Legal-BERT** `nlpaueb/legal-bert-base-uncased` (~440 MB)
- **Llama-3.1-8B-Instruct** (~15 GB, optional) -- used for Phase 1 section boundary detection. The download code is commented out in `download_models.py` by default; uncomment it to enable. If unavailable, Phase 1 falls back to rule-based chunking automatically.

To download Llama manually:

```bash
pip install huggingface_hub
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir model_weights/Meta-Llama-3.1-8B-Instruct
```

> **Note:** Llama model access requires a [Hugging Face account](https://huggingface.co/) and approval of the [Meta Llama license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). Run `huggingface-cli login` before downloading.

### Configure environment

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=sk-your-key-here
LLAMA_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
```

- `DEEPSEEK_API_KEY` -- required for Phase 3 (DHA classification via DeepSeek LLM)
- `LLAMA_MODEL_PATH` -- optional; if unset, Phase 1 falls back to rule-based chunking

## Corpus Data

The `data/` directory is **not included** in this repository. You must download the source PDFs yourself and convert them to `.txt` via the included OCR script.

### Step 1 -- Download PDFs

Download the following documents as PDF and place them in the `ScotLaw/` directory:

| # | Document | Year | PDF Source |
|---|----------|------|------------|
| 1 | Articles of Union | 1707 | [Wikisource](https://en.wikisource.org/wiki/Act_of_Union_1707) |
| 2 | Education (Scotland) Act | 1872 | [legislation.gov.uk](https://www.legislation.gov.uk/ukpga/1872/62/contents/enacted) |
| 3 | European Charter for Regional or Minority Languages | 1992 | [Council of Europe](https://www.coe.int/en/web/european-charter-regional-or-minority-languages) |
| 4 | Scotland Act | 1998 | [legislation.gov.uk](https://www.legislation.gov.uk/ukpga/1998/46/contents) |
| 5 | Standards in Scotland's Schools etc. Act | 2000 | [legislation.gov.uk](https://www.legislation.gov.uk/asp/2000/6/contents) |
| 6 | Gaelic Language (Scotland) Act | 2005 | [legislation.gov.uk](https://www.legislation.gov.uk/asp/2005/7/contents) |
| 7 | Report of Ministerial Working Group on the Scots Language | 2010 | [gov.scot](https://www.gov.scot/publications/report-ministerial-working-group-scots-language/) |
| 8 | Scots Language Policy | 2015 | [gov.scot](https://www.gov.scot/publications/scots-language-policy-english/) |
| 9 | National Gaelic Language Plan 2023--2028 | 2023 | [gaidhlig.scot](https://www.gaidhlig.scot/en/gaelic-language-plans/the-national-gaelic-language-plan/) |
| 10 | Scottish Languages Act | 2025 | [legislation.gov.uk](https://www.legislation.gov.uk/asp/2025/10/contents/enacted) |

### Step 2 -- OCR conversion

```bash
pip install paddleocr pdf2image
python pdf_to_txt.py
```

This uses PaddleOCR (GPU) to extract text from the PDFs in `ScotLaw/` and writes `.txt` files to `data/`. The output filenames must match those expected by `scottnlp/config.py`:

```
data/
├── 1articles of union1707.txt
├── 2Education (Scotland) Act1872.txt
├── 3European_Charter_for_Regional_or_Minority_Languages 1992.txt
├── 4Scotland Act 1998.txt
├── 5Standards in Scotland.txt
├── 6Gaelic Language (Scotland) Act 2005.txt
├── 7Report of Ministerial Working Group on the Scots Language 2010.txt
├── 8scots language policy2015.txt
├── 9National_Gaelic_Language_Plan_ENGLISH2023-28.txt
└── 10Scottish Languages Act 2025.txt
```

If the OCR output filenames differ from the above, rename them accordingly before running the pipeline.

## Usage

Run each phase sequentially (or Phase 2 and 3 in parallel):

```bash
# Phase 1 -- Corpus cleaning, chunking, and Legal-BERT embedding
python -m scottnlp.phase1_corpus.pipeline

# Phase 2 -- BERTopic, dependency parsing, semantic networks
python -m scottnlp.phase2_topics.pipeline

# Phase 3 -- DHA strategy classification via DeepSeek LLM
python -m scottnlp.phase3_dha.pipeline [--sample N] [--skip-classification] [--workers 5]

# Deep-dive analyses (SVO power triangle, identity construction, markedness theory)
python -m scottnlp.deep_analysis_pipeline [--skip-svo] [--skip-identity] [--skip-markedness]

# Phase 4 -- Generate all visualizations
python -m scottnlp.phase4_viz.pipeline
```

Phase 3 uses SHA-256 prompt caching -- safe to re-run without incurring duplicate API costs.

## Project Structure

```
ScottNLP/
├── scottnlp/
│   ├── config.py                    # Central config, document metadata, era definitions
│   ├── deep_analysis_pipeline.py    # Orchestrator for deep-dive analyses
│   ├── phase1_corpus/
│   │   ├── cleaning.py              # OCR correction, boilerplate removal
│   │   ├── chunking.py              # Hybrid Llama + rule-based chunking
│   │   ├── embedding.py             # Legal-BERT sentence embeddings
│   │   └── pipeline.py              # Phase 1 orchestrator
│   ├── phase2_topics/
│   │   ├── topic_modeling.py        # BERTopic with UMAP + HDBSCAN
│   │   ├── dependency_parsing.py    # spaCy language-term dependency extraction
│   │   ├── semantic_networks.py     # Era-specific directed graphs + centrality
│   │   ├── deep_analysis.py         # SVO power triangle, markedness theory
│   │   └── pipeline.py              # Phase 2 orchestrator
│   ├── phase3_dha/
│   │   ├── deepseek_client.py       # API client with caching + rate limiting
│   │   ├── classifier.py            # Concurrent batch classification
│   │   ├── prompts.py               # 5 DHA strategy prompt templates
│   │   ├── deep_analysis.py         # Identity construction trajectory
│   │   └── pipeline.py              # Phase 3 orchestrator
│   └── phase4_viz/
│       ├── visualizations.py        # 9 publication-quality matplotlib figures
│       └── pipeline.py              # Phase 4 orchestrator
├── data/                            # Source .txt documents (user-provided)
├── output/                          # All pipeline outputs (auto-generated)
│   ├── phase1/                      # Cleaned texts, chunks, embeddings
│   ├── phase2/                      # Topics, dep frames, networks, deep analyses
│   ├── phase3/                      # DHA classifications, API cache, deep analyses
│   └── phase4/                      # PNG visualizations (300 DPI)
├── pdf_to_txt.py                    # Optional: PDF to TXT via PaddleOCR
├── download_models.py               # Download spaCy + Legal-BERT models
├── requirements.txt
├── pyproject.toml
└── METHODOLOGY.md                   # Full research methodology & results
```

## Output Artifacts

| Phase | Key Output | Format | Description |
|-------|-----------|--------|-------------|
| 1 | `chunks.jsonl` | JSONL | 783 chunks with metadata prefixes |
| 1 | `embeddings.npy` | NumPy | (783, 768) Legal-BERT embeddings |
| 2 | `dep_frames.jsonl` | JSONL | 2,380 dependency frames around language terms |
| 2 | `network_*.graphml` | GraphML | 4 era-specific semantic networks |
| 2 | `power_triangle.json` | JSON | SVO agent-verb-object analysis |
| 3 | `dha_classifications.jsonl` | JSONL | 3,915 chunk x strategy classifications |
| 3 | `strategy_profiles.json` | JSON | Per-document strategy aggregation |
| 3 | `topos_trajectory.json` | JSON | Diachronic argumentation topos tracking |
| 4 | `*.png` | PNG | 9 publication-quality visualizations |

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for the complete research methodology, theoretical framework, hyperparameter choices, and detailed experimental results.

## Key Findings

1. **All three languages are overwhelmingly framed as patients** (objects of policy), with agency ratios rarely exceeding 0.10.
2. **English is the "invisible hegemon"**: only 33 mentions across 783 chunks spanning 318 years, while Gaelic (832) and Scots (473) are constantly named and characterized.
3. **The 2010 Scots Working Group Report is a turning point**: the dominant argumentative frame shifts from legal authority to cultural advantage.
4. **The 2025 Scottish Languages Act represents "re-juridification"**: legal framing reasserts dominance while incorporating cultural rationale.

## References

- Chalkidis, I. et al. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- Reisigl, M. & Wodak, R. (2001). *Discourse and Discrimination*. Routledge.
- Reisigl, M. & Wodak, R. (2009). The Discourse-Historical Approach. In *Methods of Critical Discourse Analysis* (2nd ed.). Sage.

## License

This project is for academic research purposes.
