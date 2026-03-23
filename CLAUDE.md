# ScottNLP — Claude Code Guide

## Project
Computational DHA analysis of Scottish language policy (1707–2025). Python 3.11 pipeline with 4 phases.

## Structure
```
scottnlp/config.py         # Central config, document metadata
scottnlp/phase1_corpus/    # Cleaning, chunking, embedding
scottnlp/phase2_topics/    # BERTopic, dependency parsing, semantic networks
scottnlp/phase3_dha/       # DeepSeek DHA classification (5 strategies)
scottnlp/phase4_viz/       # Visualization (TODO)
data/                      # 10 source .txt documents
output/phase{1,2,3,4}/     # Pipeline outputs
```

## Running
```bash
# Phase 1
python -m scottnlp.phase1_corpus.pipeline

# Phase 2
python -m scottnlp.phase2_topics.pipeline

# Phase 3
python -m scottnlp.phase3_dha.pipeline [--sample N] [--skip-classification] [--workers 5]

# Phase 4 (not yet implemented)
```

## Key Config
- DeepSeek API key: `.env` file (`DEEPSEEK_API_KEY`)
- LLM model weights: `/workspace/FFprobe/model_weights/`
- Embedding model: `nlpaueb/legal-bert-base-uncased`
- Max chunk tokens: 384, overlap: 64

## Conventions
- All outputs go to `output/phaseN/`
- Phase 3 uses SHA-256 caching; safe to re-run (skips cached calls)
- JSONL for per-item data, JSON for aggregated profiles, CSV for summaries