"""Deep Analysis Pipeline: SVO Power Triangle, Identity Construction, Markedness Theory.

Runs all three deep-dive analyses as a single pipeline step.

Usage:
    python -m scottnlp.deep_analysis_pipeline [--skip-svo] [--skip-identity] [--skip-markedness]
"""

import argparse
import time
from pathlib import Path

from scottnlp.config import OUTPUT_DIR

PHASE1_DIR = OUTPUT_DIR / "phase1"
PHASE2_DIR = OUTPUT_DIR / "phase2"
PHASE3_DIR = OUTPUT_DIR / "phase3"


def run_deep_analysis(
    skip_svo: bool = False,
    skip_identity: bool = False,
    skip_markedness: bool = False,
) -> dict:
    """Run all three deep-dive analyses."""
    results = {}
    start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         ScottNLP Deep Analysis Pipeline                 ║")
    print("║  SVO Power Triangle · Identity Construction · Markedness║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Verify input files exist
    deps_path = PHASE2_DIR / "dep_frames.jsonl"
    class_path = PHASE3_DIR / "dha_classifications.jsonl"
    for path in [deps_path, class_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    print(f"\nInput files verified.")

    # Analysis 2: Identity Construction (fastest — no parsing)
    if not skip_identity:
        from scottnlp.phase3_dha.deep_analysis import run_identity_construction_analysis
        print("\n")
        results["identity"] = run_identity_construction_analysis(
            classifications_path=class_path,
            output_dir=PHASE3_DIR / "deep",
        )
    else:
        print("\n[SKIP] Analysis 2: Identity Construction")

    # Analysis 3: Markedness Theory (no parsing)
    if not skip_markedness:
        from scottnlp.phase2_topics.deep_analysis import run_markedness_analysis
        print("\n")
        results["markedness"] = run_markedness_analysis(
            dep_frames_path=deps_path,
            classifications_path=class_path,
            output_dir=PHASE2_DIR / "deep",
        )
    else:
        print("\n[SKIP] Analysis 3: Markedness Theory")

    # Analysis 1: SVO Power Triangle (requires spaCy)
    if not skip_svo:
        from scottnlp.phase2_topics.deep_analysis import run_svo_analysis
        print("\n")
        results["svo"] = run_svo_analysis(
            dep_frames_path=deps_path,
            output_dir=PHASE2_DIR / "deep",
        )
    else:
        print("\n[SKIP] Analysis 1: SVO Power Triangle")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Deep analysis pipeline complete in {elapsed:.1f}s")
    print(f"Outputs:")
    print(f"  Phase 2 deep: {PHASE2_DIR / 'deep'}")
    print(f"  Phase 3 deep: {PHASE3_DIR / 'deep'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ScottNLP Deep Analysis Pipeline"
    )
    parser.add_argument("--skip-svo", action="store_true",
                       help="Skip SVO Power Triangle analysis (requires spaCy)")
    parser.add_argument("--skip-identity", action="store_true",
                       help="Skip Identity Construction Trajectory analysis")
    parser.add_argument("--skip-markedness", action="store_true",
                       help="Skip Markedness Theory analysis")
    args = parser.parse_args()

    run_deep_analysis(
        skip_svo=args.skip_svo,
        skip_identity=args.skip_identity,
        skip_markedness=args.skip_markedness,
    )


if __name__ == "__main__":
    main()
