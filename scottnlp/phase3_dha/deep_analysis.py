"""Phase 3 Deep-Dive: Identity Construction Trajectory via Argumentation Topoi."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from scottnlp.config import OUTPUT_DIR, ERA_DEFINITIONS, YEAR_TO_ERA

PHASE3_DIR = OUTPUT_DIR / "phase3"
DEEP_OUTPUT_DIR = PHASE3_DIR / "deep"

# Document title → year lookup
DOC_YEAR = {}


def _load_classifications(path: Path) -> list[dict]:
    """Load DHA classifications from JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_topos_trajectory(classifications_path: Path) -> dict:
    """Extract argumentation topoi from DHA classifications and build diachronic trajectory.

    Returns dict with keys: by_year, by_era, total_topoi_counts.
    """
    clss = _load_classifications(classifications_path)

    # Filter argumentation-present classifications
    arg_present = [
        c for c in clss
        if c.get("strategy_name") == "argumentation" and c.get("present")
    ]

    # Total chunks per year (from all classifications, deduplicated by chunk_id per strategy)
    chunks_per_year = Counter()
    seen_chunks = set()
    for c in clss:
        cid = c.get("chunk_id", "")
        if cid not in seen_chunks:
            seen_chunks.add(cid)
            chunks_per_year[c.get("doc_year", 0)] += 1

    # Build doc_year → doc_title mapping
    for c in clss:
        yr = c.get("doc_year", 0)
        if yr not in DOC_YEAR:
            DOC_YEAR[yr] = c.get("doc_title", "")

    # Topoi by year
    by_year = {}
    topoi_by_year = defaultdict(Counter)
    arg_present_by_year = defaultdict(int)

    for c in arg_present:
        yr = c.get("doc_year", 0)
        arg_present_by_year[yr] += 1
        for t in c.get("topoi", []):
            topoi_by_year[yr][t] += 1

    for yr in sorted(topoi_by_year):
        counts = dict(topoi_by_year[yr].most_common())
        total_topoi = sum(counts.values())
        proportions = {t: round(n / arg_present_by_year[yr], 3) for t, n in counts.items()}
        dominant = max(counts, key=counts.get) if counts else None

        by_year[yr] = {
            "doc_title": DOC_YEAR.get(yr, ""),
            "total_chunks": chunks_per_year.get(yr, 0),
            "argumentation_present": arg_present_by_year[yr],
            "topoi_counts": counts,
            "topoi_proportions": proportions,
            "dominant_topos": dominant,
            "total_topoi_instances": total_topoi,
        }

    # Topoi by era
    by_era = {}
    for era, years in ERA_DEFINITIONS.items():
        era_counts = Counter()
        era_arg_present = 0
        era_total_chunks = 0
        for yr in years:
            if yr in topoi_by_year:
                era_counts += topoi_by_year[yr]
            era_arg_present += arg_present_by_year.get(yr, 0)
            era_total_chunks += chunks_per_year.get(yr, 0)

        if era_counts:
            total = sum(era_counts.values())
            proportions = {t: round(n / era_arg_present, 3) if era_arg_present else 0
                          for t, n in era_counts.most_common()}
            dominant = max(era_counts, key=era_counts.get)
        else:
            total = 0
            proportions = {}
            dominant = None

        by_era[era] = {
            "total_chunks": era_total_chunks,
            "argumentation_present": era_arg_present,
            "topoi_counts": dict(era_counts.most_common()),
            "topoi_proportions": proportions,
            "dominant_topos": dominant,
            "total_topoi_instances": total,
        }

    # Overall topos counts
    total_topoi_counts = Counter()
    for yr_counts in topoi_by_year.values():
        total_topoi_counts += yr_counts

    return {
        "by_year": by_year,
        "by_era": by_era,
        "total_topoi_counts": dict(total_topoi_counts.most_common()),
    }


def identify_turning_points(topos_by_year: dict) -> list[dict]:
    """Identify documents where the dominant argumentative frame shifts.

    A turning point occurs when:
    1. The dominant topos changes from the previous document, OR
    2. New topoi emerge that were absent in all earlier documents
    """
    sorted_years = sorted(topos_by_year.keys())
    turning_points = []
    all_seen_topoi = set()

    prev_dominant = None
    for i, yr in enumerate(sorted_years):
        entry = topos_by_year[yr]
        current_dominant = entry.get("dominant_topos")
        current_topoi = set(entry.get("topoi_counts", {}).keys())

        # Find newly emerging topoi
        emerging = current_topoi - all_seen_topoi

        is_shift = prev_dominant is not None and current_dominant != prev_dominant
        has_emerging = len(emerging) > 0

        if is_shift or has_emerging:
            tp = {
                "year": yr,
                "doc_title": entry.get("doc_title", ""),
                "dominant_topos": current_dominant,
            }
            if is_shift:
                tp["previous_dominant"] = prev_dominant
                tp["shift_type"] = "dominant_topos_change"
            if has_emerging:
                tp["emerging_topoi"] = sorted(emerging)
                if not is_shift:
                    tp["shift_type"] = "new_topoi_emergence"
                else:
                    tp["shift_type"] = "dominant_change_and_new_topoi"

            # Compute shift magnitude: proportion change of the old dominant
            if prev_dominant and prev_dominant in entry.get("topoi_counts", {}):
                old_count = entry["topoi_counts"][prev_dominant]
                new_count = entry["topoi_counts"].get(current_dominant, 0)
                tp["magnitude"] = round(
                    (new_count - old_count) / max(new_count, old_count, 1), 3
                )
            turning_points.append(tp)

        all_seen_topoi |= current_topoi
        prev_dominant = current_dominant

    return turning_points


def build_identity_construction_timeline(
    trajectory: dict,
    classifications_path: Path,
) -> dict:
    """Build per-era identity construction profile with evidence.

    For each era computes:
    - dominant_frame, secondary_frame
    - frame_diversity: Shannon entropy of topos distribution
    - law_vs_culture_ratio: law / (law + culture + history + heritage)
    - representative evidence quotes
    """
    clss = _load_classifications(classifications_path)
    arg_present = [
        c for c in clss
        if c.get("strategy_name") == "argumentation" and c.get("present")
    ]

    # Group evidence by era + topos
    evidence_by_era_topos = defaultdict(lambda: defaultdict(list))
    for c in arg_present:
        era = YEAR_TO_ERA.get(c.get("doc_year", 0), "unknown")
        for t in c.get("topoi", []):
            quotes = c.get("evidence_quotes", [])
            if quotes:
                evidence_by_era_topos[era][t].append({
                    "doc_title": c.get("doc_title", ""),
                    "doc_year": c.get("doc_year", 0),
                    "quote": quotes[0][:200],  # First quote, truncated
                })

    timeline = {}
    for era, era_data in trajectory["by_era"].items():
        counts = era_data.get("topoi_counts", {})
        if not counts:
            continue

        sorted_topoi = sorted(counts.items(), key=lambda x: -x[1])
        dominant = sorted_topoi[0][0] if sorted_topoi else None
        secondary = sorted_topoi[1][0] if len(sorted_topoi) > 1 else None

        # Shannon entropy
        total = sum(counts.values())
        probs = [n / total for n in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Law vs culture ratio
        law_count = counts.get("topos_of_law", 0)
        culture_count = (
            counts.get("topos_of_culture", 0)
            + counts.get("topos_of_history", 0)
            + counts.get("topos_of_heritage", 0)
        )
        law_culture_total = law_count + culture_count
        law_vs_culture = round(law_count / law_culture_total, 3) if law_culture_total > 0 else None

        # Evidence: top 2 quotes for dominant topos
        key_evidence = {}
        for topos_name in [dominant, secondary]:
            if topos_name and topos_name in evidence_by_era_topos[era]:
                quotes = evidence_by_era_topos[era][topos_name][:2]
                key_evidence[topos_name] = quotes

        timeline[era] = {
            "dominant_frame": dominant,
            "secondary_frame": secondary,
            "frame_diversity": round(entropy, 3),
            "law_vs_culture_ratio": law_vs_culture,
            "argumentation_prevalence": round(
                era_data["argumentation_present"] / era_data["total_chunks"], 3
            ) if era_data["total_chunks"] > 0 else 0,
            "total_topoi_instances": era_data["total_topoi_instances"],
            "topoi_ranked": [{"topos": t, "count": n} for t, n in sorted_topoi],
            "key_evidence": key_evidence,
        }

    return timeline


def compute_topos_language_matrix(classifications_path: Path) -> pd.DataFrame:
    """Cross-tabulate topoi against target languages.

    Returns DataFrame: rows=topoi, columns=languages, values=counts.
    Tests hypothesis: law topoi target English, culture topoi target Gaelic/Scots.
    """
    clss = _load_classifications(classifications_path)
    arg_present = [
        c for c in clss
        if c.get("strategy_name") == "argumentation" and c.get("present")
    ]

    topos_lang = defaultdict(Counter)
    for c in arg_present:
        for t in c.get("topoi", []):
            for lang in c.get("target_languages", []):
                topos_lang[t][lang] += 1

    # Build DataFrame
    all_topoi = sorted(topos_lang.keys())
    all_langs = sorted(set(
        lang for counts in topos_lang.values() for lang in counts
    ))

    rows = []
    for t in all_topoi:
        row = {"topos": t}
        for lang in all_langs:
            row[lang] = topos_lang[t].get(lang, 0)
        row["total"] = sum(topos_lang[t].values())
        rows.append(row)

    return pd.DataFrame(rows)


def save_identity_construction_results(
    trajectory: dict,
    turning_points: list[dict],
    timeline: dict,
    topos_language_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all Analysis 2 outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full trajectory with embedded turning points and timeline
    full_output = {
        **trajectory,
        "turning_points": turning_points,
        "identity_timeline": timeline,
    }
    traj_path = output_dir / "topos_trajectory.json"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"Saved topos trajectory to {traj_path}")

    # Topos × language matrix
    matrix_path = output_dir / "topos_language_matrix.csv"
    topos_language_df.to_csv(matrix_path, index=False)
    print(f"Saved topos-language matrix to {matrix_path}")

    # Flat summary CSV
    rows = []
    for era, data in timeline.items():
        rows.append({
            "era": era,
            "dominant_topos": data["dominant_frame"],
            "secondary_topos": data["secondary_frame"],
            "law_vs_culture_ratio": data["law_vs_culture_ratio"],
            "frame_diversity": data["frame_diversity"],
            "argumentation_prevalence": data["argumentation_prevalence"],
            "total_topoi_instances": data["total_topoi_instances"],
        })
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "identity_construction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved identity construction summary to {summary_path}")


def run_identity_construction_analysis(
    classifications_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run the full Identity Construction Trajectory analysis.

    Returns the trajectory dict for downstream use.
    """
    if classifications_path is None:
        classifications_path = PHASE3_DIR / "dha_classifications.jsonl"
    if output_dir is None:
        output_dir = DEEP_OUTPUT_DIR

    print("=" * 60)
    print("ANALYSIS 2: Identity Construction Trajectory")
    print("=" * 60)

    # Step 1: Extract topos trajectory
    print("\n[1/4] Extracting topos trajectory...")
    trajectory = extract_topos_trajectory(classifications_path)
    n_topoi = len(trajectory["total_topoi_counts"])
    total_instances = sum(trajectory["total_topoi_counts"].values())
    print(f"  Found {n_topoi} unique topoi across {total_instances} instances")

    # Step 2: Identify turning points
    print("\n[2/4] Identifying turning points...")
    turning_points = identify_turning_points(trajectory["by_year"])
    print(f"  Found {len(turning_points)} turning points:")
    for tp in turning_points:
        shift = tp.get("shift_type", "")
        print(f"    {tp['year']} ({tp['doc_title']}): {shift}")
        if "previous_dominant" in tp:
            print(f"      {tp['previous_dominant']} → {tp['dominant_topos']}")
        if "emerging_topoi" in tp:
            print(f"      New topoi: {tp['emerging_topoi']}")

    # Step 3: Build identity timeline
    print("\n[3/4] Building identity construction timeline...")
    timeline = build_identity_construction_timeline(trajectory, classifications_path)
    for era, data in timeline.items():
        print(f"  {era}:")
        print(f"    Dominant: {data['dominant_frame']} | Secondary: {data['secondary_frame']}")
        print(f"    Law vs Culture ratio: {data['law_vs_culture_ratio']}")
        print(f"    Frame diversity (entropy): {data['frame_diversity']}")

    # Step 4: Compute topos × language matrix
    print("\n[4/4] Computing topos-language cross-tabulation...")
    topos_lang_df = compute_topos_language_matrix(classifications_path)
    print(topos_lang_df.to_string(index=False))

    # Save
    print("\nSaving results...")
    save_identity_construction_results(
        trajectory, turning_points, timeline, topos_lang_df, output_dir
    )

    return {
        "trajectory": trajectory,
        "turning_points": turning_points,
        "timeline": timeline,
        "topos_language_matrix": topos_lang_df,
    }
