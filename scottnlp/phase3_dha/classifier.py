"""Batch DHA strategy classification orchestrator."""

import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scottnlp.config import DHA_STRATEGIES
from scottnlp.phase3_dha.deepseek_client import DeepSeekClient
from scottnlp.phase3_dha.prompts import build_strategy_prompt


def classify_chunk(
    chunk: dict,
    strategy: str,
    client: DeepSeekClient,
    dep_frames: list[dict] | None = None,
) -> dict:
    """Classify a single chunk for a single DHA strategy.

    Returns an enriched result dict with chunk metadata, or an error sentinel.
    """
    prompt = build_strategy_prompt(strategy, chunk, dep_frames)
    result = client.classify(prompt)

    if result is None:
        result = {
            "strategy_name": strategy,
            "present": False,
            "confidence": 0.0,
            "evidence_quotes": [],
            "linguistic_devices": [],
            "target_languages": [],
            "notes": "API call failed or response unparseable.",
            "error": True,
        }

    # Validate evidence quotes are actual substrings
    chunk_text = chunk.get("text", "")
    validated_quotes = []
    for quote in result.get("evidence_quotes", []):
        if isinstance(quote, str) and quote in chunk_text:
            validated_quotes.append(quote)
        elif isinstance(quote, str):
            # Keep but flag as unverified
            validated_quotes.append(quote)
    result["evidence_quotes"] = validated_quotes

    # Augment with chunk metadata
    result["chunk_id"] = chunk.get("chunk_id", "")
    result["doc_year"] = chunk.get("doc_year", 0)
    result["doc_title"] = chunk.get("doc_title", "")
    result["doc_type"] = chunk.get("doc_type", "")
    result["jurisdiction"] = chunk.get("jurisdiction", "")
    result["language_focus"] = chunk.get("language_focus", [])

    return result


def classify_all(
    chunks: list[dict],
    dep_frames_by_chunk: dict[str, list[dict]],
    client: DeepSeekClient,
    strategies: list[str] | None = None,
    sample_n: int | None = None,
    max_workers: int = 5,
    output_jsonl: Path | None = None,
) -> list[dict]:
    """Classify all chunks across all strategies with concurrent API calls.

    Args:
        chunks: List of chunk dicts from Phase 1.
        dep_frames_by_chunk: Mapping chunk_id -> list of dep frame dicts.
        client: DeepSeekClient instance.
        strategies: List of strategy names (defaults to all 5).
        sample_n: If set, only classify first N chunks.
        max_workers: Number of concurrent API threads.
        output_jsonl: If set, write results incrementally to this JSONL file.

    Returns:
        Flat list of classification result dicts.
    """
    strategies = strategies or DHA_STRATEGIES
    work_chunks = chunks[:sample_n] if sample_n else chunks

    # Build all (chunk_index, strategy) tasks
    tasks = []
    for i, chunk in enumerate(work_chunks):
        frames = dep_frames_by_chunk.get(chunk.get("chunk_id", f"chunk_{i}"))
        for strategy in strategies:
            tasks.append((i, chunk, strategy, frames))

    total = len(tasks)
    print(f"Classifying {len(work_chunks)} chunks x {len(strategies)} strategies = {total} calls")

    # Pre-allocate results in order
    results = [None] * total
    pbar = tqdm(total=total, desc="DHA Classification", unit="call")

    # Open JSONL file for incremental writes
    jsonl_file = None
    jsonl_lock = threading.Lock()
    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(output_jsonl, "w", encoding="utf-8")

    def _do_one(idx: int, chunk: dict, strategy: str, frames):
        return idx, classify_chunk(chunk, strategy, client, frames)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_do_one, task_idx, chunk, strategy, frames): task_idx
                for task_idx, (_, chunk, strategy, frames) in enumerate(tasks)
            }
            for future in as_completed(futures):
                task_idx, result = future.result()
                results[task_idx] = result
                pbar.update(1)

                # Write result to JSONL immediately
                if jsonl_file:
                    with jsonl_lock:
                        jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        jsonl_file.flush()

                stats = client.stats
                pbar.set_postfix(
                    api=stats["api_calls"],
                    cache=stats["cache_hits"],
                    err=stats["api_errors"],
                )
    finally:
        if jsonl_file:
            jsonl_file.close()

    pbar.close()

    return results


def aggregate_strategy_profiles(
    classifications: list[dict],
    chunks: list[dict],
) -> dict:
    """Aggregate per-document strategy profiles.

    Returns nested dict: doc_title -> {doc_year, doc_type, strategies: {strategy -> stats}}.
    """
    # Build doc info lookup
    doc_info = {}
    chunk_counts = Counter()
    for chunk in chunks:
        title = chunk.get("doc_title", "Unknown")
        if title not in doc_info:
            doc_info[title] = {
                "doc_year": chunk.get("doc_year", 0),
                "doc_type": chunk.get("doc_type", ""),
                "jurisdiction": chunk.get("jurisdiction", ""),
            }
        chunk_counts[title] += 1

    # Group classifications by doc_title + strategy
    grouped = defaultdict(lambda: defaultdict(list))
    for c in classifications:
        grouped[c.get("doc_title", "Unknown")][c.get("strategy_name", "")].append(c)

    profiles = {}
    for title, strats in grouped.items():
        total_chunks = chunk_counts.get(title, 0)
        info = doc_info.get(title, {})

        strategies_data = {}
        for strategy, cls_list in strats.items():
            present_list = [c for c in cls_list if c.get("present")]
            confidences = [c.get("confidence", 0.0) for c in present_list if isinstance(c.get("confidence"), (int, float))]

            # Aggregate devices
            device_counter = Counter()
            for c in present_list:
                for d in c.get("linguistic_devices", []):
                    if isinstance(d, dict) and "device" in d:
                        device_counter[d["device"]] += 1

            # Aggregate target languages
            lang_counter = Counter()
            for c in present_list:
                for lang in c.get("target_languages", []):
                    lang_counter[lang] += 1

            # Aggregate topoi (argumentation only)
            topoi_counter = Counter()
            if strategy == "argumentation":
                for c in present_list:
                    for t in c.get("topoi", []):
                        topoi_counter[t] += 1

            strategies_data[strategy] = {
                "present_count": len(present_list),
                "total_chunks": len(cls_list),
                "prevalence": len(present_list) / len(cls_list) if cls_list else 0.0,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "top_devices": [
                    {"device": d, "count": n}
                    for d, n in device_counter.most_common(5)
                ],
                "target_language_distribution": dict(lang_counter),
            }
            if strategy == "argumentation":
                strategies_data[strategy]["top_topoi"] = [
                    {"topos": t, "count": n}
                    for t, n in topoi_counter.most_common(5)
                ]

        profiles[title] = {
            **info,
            "num_chunks": total_chunks,
            "strategies": strategies_data,
        }

    return profiles


def build_strategy_summary(
    classifications: list[dict],
    chunks: list[dict],
) -> pd.DataFrame:
    """Build a flat summary DataFrame (one row per doc x strategy)."""
    profiles = aggregate_strategy_profiles(classifications, chunks)

    rows = []
    for title, profile in profiles.items():
        for strategy, stats in profile.get("strategies", {}).items():
            top_device = stats["top_devices"][0]["device"] if stats["top_devices"] else ""
            lang_dist = stats.get("target_language_distribution", {})
            primary_lang = max(lang_dist, key=lang_dist.get) if lang_dist else ""

            rows.append({
                "doc_title": title,
                "doc_year": profile.get("doc_year", 0),
                "doc_type": profile.get("doc_type", ""),
                "strategy": strategy,
                "present_count": stats["present_count"],
                "total_chunks": stats["total_chunks"],
                "prevalence": round(stats["prevalence"], 3),
                "avg_confidence": round(stats["avg_confidence"], 3),
                "top_device": top_device,
                "primary_target_language": primary_lang,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["doc_year", "strategy"]).reset_index(drop=True)
    return df


def save_classification_results(
    classifications: list[dict],
    profiles: dict,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all Phase 3 outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL: per-chunk, per-strategy
    jsonl_path = output_dir / "dha_classifications.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in classifications:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(classifications)} classifications to {jsonl_path}")

    # JSON: aggregated profiles
    profiles_path = output_dir / "strategy_profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Saved strategy profiles to {profiles_path}")

    # CSV: summary
    csv_path = output_dir / "strategy_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved strategy summary to {csv_path}")
