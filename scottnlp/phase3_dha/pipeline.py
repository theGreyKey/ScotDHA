"""Phase 3 pipeline orchestrator: DHA strategy classification via DeepSeek."""

import json
from collections import defaultdict
from pathlib import Path

from scottnlp.config import DHA_STRATEGIES, OUTPUT_DIR
from scottnlp.phase1_corpus.embedding import load_embeddings
from scottnlp.phase3_dha.classifier import (
    aggregate_strategy_profiles,
    build_strategy_summary,
    classify_all,
    save_classification_results,
)
from scottnlp.phase3_dha.deepseek_client import DeepSeekClient


PHASE1_DIR = OUTPUT_DIR / "phase1"
PHASE2_DIR = OUTPUT_DIR / "phase2"
PHASE3_DIR = OUTPUT_DIR / "phase3"


def _load_dep_frames_grouped(dep_frames_path: Path) -> dict[str, list[dict]]:
    """Load dependency frames from JSONL and group by chunk_id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    if not dep_frames_path.exists():
        print(f"Warning: {dep_frames_path} not found. Proceeding without dep frames.")
        return grouped

    with open(dep_frames_path, "r", encoding="utf-8") as f:
        for line in f:
            frame = json.loads(line)
            grouped[frame.get("chunk_id", "")].append(frame)

    return grouped


def run_phase3(
    phase1_dir: Path = PHASE1_DIR,
    phase2_dir: Path = PHASE2_DIR,
    output_dir: Path = PHASE3_DIR,
    sample_n: int | None = None,
    force_reclassify: bool = False,
    skip_classification: bool = False,
    skip_aggregation: bool = False,
    max_workers: int = 5,
) -> dict:
    """Execute the complete Phase 3 DHA classification pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────
    print("=" * 60)
    print("Loading Phase 1 chunks...")
    print("=" * 60)
    _, chunks = load_embeddings(phase1_dir)
    print(f"Loaded {len(chunks)} chunks")

    print("Loading Phase 2 dependency frames...")
    dep_frames_by_chunk = _load_dep_frames_grouped(phase2_dir / "dep_frames.jsonl")
    print(f"Loaded frames for {len(dep_frames_by_chunk)} chunks")

    summary: dict = {"num_chunks": len(chunks)}
    classifications: list[dict] = []

    # ── Step 1: Classification ────────────────────────────────────
    if not skip_classification:
        print("\n" + "=" * 60)
        print("STEP 1: DHA Strategy Classification via DeepSeek")
        print("=" * 60)

        client = DeepSeekClient(cache_path=output_dir / "api_cache.json")

        if force_reclassify:
            print("Force reclassify: clearing cache...")
            client.clear_cache()

        jsonl_path = output_dir / "dha_classifications.jsonl"
        classifications = classify_all(
            chunks,
            dep_frames_by_chunk,
            client,
            strategies=DHA_STRATEGIES,
            sample_n=sample_n,
            max_workers=max_workers,
            output_jsonl=jsonl_path,
        )

        # Flush any remaining cached results to disk
        client.flush_cache()
        print(f"Saved {len(classifications)} classifications to {jsonl_path}")

        stats = client.stats
        summary.update({
            "total_classifications": len(classifications),
            "api_calls": stats["api_calls"],
            "cache_hits": stats["cache_hits"],
            "api_errors": stats["api_errors"],
        })
    else:
        # Load existing classifications
        jsonl_path = output_dir / "dha_classifications.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                classifications = [json.loads(line) for line in f]
            print(f"Loaded {len(classifications)} existing classifications from {jsonl_path}")
            summary["total_classifications"] = len(classifications)
        else:
            print(f"No existing classifications found at {jsonl_path}. "
                  "Run without --skip-classification first.")
            return summary

    # ── Step 2: Aggregation ───────────────────────────────────────
    if not skip_aggregation:
        print("\n" + "=" * 60)
        print("STEP 2: Aggregating Strategy Profiles")
        print("=" * 60)

        work_chunks = chunks[:sample_n] if sample_n else chunks
        profiles = aggregate_strategy_profiles(classifications, work_chunks)
        summary_df = build_strategy_summary(classifications, work_chunks)

        save_classification_results(classifications, profiles, summary_df, output_dir)

        summary["num_documents_profiled"] = len(profiles)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Per-strategy breakdown
    for strategy in DHA_STRATEGIES:
        strat_results = [c for c in classifications if c.get("strategy_name") == strategy]
        present = sum(1 for c in strat_results if c.get("present"))
        print(f"  {strategy}: {present}/{len(strat_results)} chunks positive")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3 DHA classification pipeline")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Classify only first N chunks (for testing)",
    )
    parser.add_argument(
        "--force-reclassify", action="store_true",
        help="Clear cache and re-classify all chunks",
    )
    parser.add_argument(
        "--skip-classification", action="store_true",
        help="Skip classification, load existing results",
    )
    parser.add_argument(
        "--skip-aggregation", action="store_true",
        help="Skip aggregation step",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of concurrent API threads (default: 5)",
    )
    args = parser.parse_args()

    run_phase3(
        sample_n=args.sample,
        force_reclassify=args.force_reclassify,
        skip_classification=args.skip_classification,
        skip_aggregation=args.skip_aggregation,
        max_workers=args.workers,
    )
