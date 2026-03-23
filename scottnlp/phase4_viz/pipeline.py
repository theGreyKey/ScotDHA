"""Phase 4 pipeline orchestrator: visualization of DHA analysis results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scottnlp.config import OUTPUT_DIR
from scottnlp.phase4_viz.visualizations import (
    plot_dha_strategies,
    plot_embedding_space,
    plot_language_agency,
    plot_markedness_diverging,
    plot_network_growth,
    plot_semantic_networks,
    plot_svo_sankey,
    plot_topic_heatmap,
    plot_topos_streamgraph,
    setup_style,
)

PHASE1_DIR = OUTPUT_DIR / "phase1"
PHASE2_DIR = OUTPUT_DIR / "phase2"
PHASE3_DIR = OUTPUT_DIR / "phase3"
PHASE4_DIR = OUTPUT_DIR / "phase4"


def run_phase4(
    phase1_dir: Path = PHASE1_DIR,
    phase2_dir: Path = PHASE2_DIR,
    phase3_dir: Path = PHASE3_DIR,
    output_dir: Path = PHASE4_DIR,
    skip_topics: bool = False,
    skip_networks: bool = False,
    skip_dha: bool = False,
    skip_agency: bool = False,
    skip_embeddings: bool = False,
    skip_streamgraph: bool = False,
    skip_sankey: bool = False,
    skip_markedness: bool = False,
    skip_diachronic: bool = False,
) -> dict:
    """Execute the complete Phase 4 visualization pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("=" * 60)
    print("PHASE 4: Visualization")
    print("=" * 60)

    summary: dict = {}

    # ── Viz 1: Topic Heatmap ───────────────────────────────────────
    if not skip_topics:
        print("\nSTEP 1: Topic Evolution Heatmap")
        topics_df = pd.read_csv(phase2_dir / "topics" / "topics_over_time.csv")
        with open(phase2_dir / "topics" / "topic_words.json") as f:
            topic_words = json.load(f)
        plot_topic_heatmap(topics_df, topic_words, output_dir)
        summary["topic_heatmap"] = "saved"

    # ── Viz 2: Semantic Networks ───────────────────────────────────
    if not skip_networks:
        print("\nSTEP 2: Semantic Network Graphs")
        # Load dep_frames for POS-based node filtering
        dep_frames_path = phase2_dir / "dep_frames.jsonl"
        dep_frames = []
        if dep_frames_path.exists():
            with open(dep_frames_path, "r", encoding="utf-8") as f:
                dep_frames = [json.loads(line) for line in f]
        plot_semantic_networks(
            phase2_dir / "networks", output_dir, dep_frames=dep_frames,
        )
        summary["semantic_networks"] = "saved"

    # ── Viz 3: DHA Strategy Distribution ───────────────────────────
    if not skip_dha:
        print("\nSTEP 3: DHA Strategy Distribution")
        strategy_df = pd.read_csv(phase3_dir / "strategy_summary.csv")
        plot_dha_strategies(strategy_df, output_dir)
        summary["dha_strategies"] = "saved"

    # ── Viz 4: Language Agency ─────────────────────────────────────
    if not skip_agency:
        print("\nSTEP 4: Language Agency Chart")
        agency_df = pd.read_csv(phase2_dir / "agency_summary.csv")
        plot_language_agency(agency_df, output_dir)
        summary["language_agency"] = "saved"

    # ── Viz 5: Embedding Space ─────────────────────────────────────
    if not skip_embeddings:
        print("\nSTEP 5: UMAP Embedding Space")
        embeddings = np.load(phase1_dir / "embeddings.npy")
        topic_assignments = pd.read_csv(
            phase2_dir / "topics" / "topic_assignments.csv"
        )
        plot_embedding_space(embeddings, topic_assignments, output_dir)
        summary["embedding_space"] = "saved"

    # ── Viz 6: Topos Streamgraph ─────────────────────────────────
    if not skip_streamgraph:
        print("\nSTEP 6: Topos Streamgraph (Identity Construction Trajectory)")
        with open(phase3_dir / "deep" / "topos_trajectory.json") as f:
            topos_data = json.load(f)
        plot_topos_streamgraph(topos_data, output_dir)
        summary["topos_streamgraph"] = "saved"

    # ── Viz 7: SVO Sankey Diagram ────────────────────────────────
    if not skip_sankey:
        print("\nSTEP 7: SVO Power Triangle Sankey Diagram")
        svo_path = phase2_dir / "deep" / "svo_triples.jsonl"
        svo_triples = []
        if svo_path.exists():
            with open(svo_path, "r", encoding="utf-8") as f:
                svo_triples = [json.loads(line) for line in f]
        plot_svo_sankey(svo_triples, output_dir)
        summary["svo_sankey"] = "saved"

    # ── Viz 8: Markedness Diverging Bar Chart ────────────────────
    if not skip_markedness:
        print("\nSTEP 8: Markedness Asymmetry Diverging Chart")
        with open(phase2_dir / "deep" / "strategy_targeting.json") as f:
            strategy_targeting = json.load(f)
        plot_markedness_diverging(strategy_targeting, output_dir)
        summary["markedness_diverging"] = "saved"

    # ── Viz 9: Network Growth Metrics ─────────────────────────────
    if not skip_diachronic:
        print("\nSTEP 9: Network Growth Metrics")
        plot_network_growth(phase2_dir / "networks", output_dir)
        summary["network_growth"] = "saved"

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  output_dir: {output_dir}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 4 visualization pipeline"
    )
    parser.add_argument("--skip-topics", action="store_true")
    parser.add_argument("--skip-networks", action="store_true")
    parser.add_argument("--skip-dha", action="store_true")
    parser.add_argument("--skip-agency", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-streamgraph", action="store_true")
    parser.add_argument("--skip-sankey", action="store_true")
    parser.add_argument("--skip-markedness", action="store_true")
    parser.add_argument("--skip-diachronic", action="store_true")
    args = parser.parse_args()

    run_phase4(
        skip_topics=args.skip_topics,
        skip_networks=args.skip_networks,
        skip_dha=args.skip_dha,
        skip_agency=args.skip_agency,
        skip_embeddings=args.skip_embeddings,
        skip_streamgraph=args.skip_streamgraph,
        skip_sankey=args.skip_sankey,
        skip_markedness=args.skip_markedness,
        skip_diachronic=args.skip_diachronic,
    )
