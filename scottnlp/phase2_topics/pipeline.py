"""Phase 2 pipeline orchestrator: topic modeling + dependency parsing + semantic networks."""

import gc
import json
from pathlib import Path

import torch

from scottnlp.config import OUTPUT_DIR
from scottnlp.phase1_corpus.embedding import load_embeddings
from scottnlp.phase2_topics.topic_modeling import (
    build_topic_model,
    extract_dynamic_topics,
    analyze_topic_trajectories,
    save_topic_results,
)
from scottnlp.phase2_topics.dependency_parsing import (
    extract_language_dependency_frames,
    build_predicate_inventory,
    save_dependency_results,
)
from scottnlp.phase2_topics.semantic_networks import (
    build_era_networks,
    compare_networks_across_eras,
    save_network_results,
    compute_centrality_metrics,
)


PHASE1_DIR = OUTPUT_DIR / "phase1"
PHASE2_DIR = OUTPUT_DIR / "phase2"


def _free_gpu():
    """Force-release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_phase2(
    phase1_dir: Path = PHASE1_DIR,
    output_dir: Path = PHASE2_DIR,
    min_topic_size: int = 5,
    skip_topics: bool = False,
    skip_deps: bool = False,
    skip_networks: bool = False,
) -> dict:
    """Execute the complete Phase 2 pipeline. Results are overwritten on each run."""
    # ── Load Phase 1 ───────────────────────────────────────────────
    print("=" * 60)
    print("Loading Phase 1 outputs...")
    print("=" * 60)
    embeddings, chunks = load_embeddings(phase1_dir)
    print(f"Loaded {len(chunks)} chunks, embeddings shape {embeddings.shape}")

    summary = {"num_chunks": len(chunks)}

    # ── Step 1: Topic Modeling ─────────────────────────────────────
    if not skip_topics:
        print("\n" + "=" * 60)
        print("STEP 1: BERTopic Dynamic Topic Modeling")
        print("=" * 60)

        topics_dir = output_dir / "topics"
        print("[pipeline] Calling build_topic_model...")
        topic_model, topics = build_topic_model(
            chunks, embeddings, min_topic_size=min_topic_size,
        )
        print("[pipeline] Calling extract_dynamic_topics...")
        topics_over_time = extract_dynamic_topics(topic_model, chunks, topics)
        print("[pipeline] Calling analyze_topic_trajectories...")
        analysis = analyze_topic_trajectories(topics_over_time, topic_model)

        # Free GPU memory BEFORE saving (save doesn't need GPU)
        # Move embedding model to CPU instead of deleting it (BERTopic.save needs the attr)
        print("[pipeline] Moving embedding model to CPU...")
        if hasattr(topic_model, 'embedding_model') and hasattr(topic_model.embedding_model, '_modules'):
            topic_model.embedding_model = topic_model.embedding_model.cpu()
        _free_gpu()
        print("[pipeline] Released GPU memory.")

        print("[pipeline] Calling save_topic_results...")
        save_topic_results(
            topic_model, topics, topics_over_time, analysis, chunks, topics_dir,
        )
        print("[pipeline] save_topic_results done.")

        summary["num_topics"] = len(set(topics)) - (1 if -1 in topics else 0)
        summary["emerging_topics"] = len(analysis["emerging_topics"])
        summary["persistent_topics"] = len(analysis["persistent_topics"])

        del topic_model
        _free_gpu()
        print("Released all topic model memory.")

    # ── Step 2: Dependency Parsing ─────────────────────────────────
    dep_frames = []
    if not skip_deps:
        print("\n" + "=" * 60)
        print("STEP 2: Dependency Parsing around Language Terms")
        print("=" * 60)

        dep_frames = extract_language_dependency_frames(chunks)
        predicate_inv = build_predicate_inventory(dep_frames)
        save_dependency_results(dep_frames, predicate_inv, output_dir)

        summary["num_dep_frames"] = len(dep_frames)
        for term, years in predicate_inv.items():
            for year, data in years.items():
                if data["total_occurrences"] >= 3:
                    print(f"  {term} ({year}): agency={data['agency_ratio']:.2f}, "
                          f"n={data['total_occurrences']}, "
                          f"top verb={data['top_verbs'][0]['verb'] if data['top_verbs'] else 'N/A'}")

    # ── Step 3: Semantic Networks ──────────────────────────────────
    if not skip_networks:
        print("\n" + "=" * 60)
        print("STEP 3: Semantic Network Construction")
        print("=" * 60)

        if not dep_frames:
            frames_path = output_dir / "dep_frames.jsonl"
            if frames_path.exists():
                with open(frames_path) as f:
                    dep_frames = [json.loads(line) for line in f]
                print(f"Loaded {len(dep_frames)} dependency frames from disk")
            else:
                print("No dependency frames available. Run with skip_deps=False first.")
                return summary

        networks_dir = output_dir / "networks"
        era_networks = build_era_networks(dep_frames, min_edge_weight=1)
        comparison = compare_networks_across_eras(era_networks)
        save_network_results(era_networks, comparison, networks_dir)

        for era, G in era_networks.items():
            print(f"  {era}: {len(G)} nodes, {G.number_of_edges()} edges")
            if len(G) > 0:
                centrality = compute_centrality_metrics(G)
                top3 = centrality.head(3)
                for _, row in top3.iterrows():
                    print(f"    Top: {row['node']} (PageRank={row['pagerank']:.4f})")

        summary["num_era_networks"] = len(era_networks)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 2 pipeline")
    parser.add_argument("--min-topic-size", type=int, default=5)
    parser.add_argument("--skip-topics", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-networks", action="store_true")
    args = parser.parse_args()

    run_phase2(
        min_topic_size=args.min_topic_size,
        skip_topics=args.skip_topics,
        skip_deps=args.skip_deps,
        skip_networks=args.skip_networks,
    )
