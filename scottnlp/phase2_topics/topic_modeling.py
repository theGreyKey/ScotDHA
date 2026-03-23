"""BERTopic dynamic topic modeling over Legal-BERT embeddings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer


def build_topic_model(
    chunks: list[dict],
    embeddings: np.ndarray,
    min_topic_size: int = 5,
    n_neighbors: int = 15,
    n_components: int = 5,
    nr_topics: str | int = "auto",
) -> tuple[BERTopic, np.ndarray]:
    """Build BERTopic model from pre-computed Legal-BERT embeddings.

    Args:
        chunks: list of chunk dicts (must have 'text' field)
        embeddings: pre-computed (N, 768) embedding array
        min_topic_size: minimum documents per topic
        n_neighbors: UMAP n_neighbors
        n_components: UMAP dimensionality
        nr_topics: "auto" or target number of topics

    Returns:
        (fitted BERTopic model, topic assignments array)
    """
    texts = [c["text"] for c in chunks]

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN for clustering (small corpus needs small clusters)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=2,
        metric="euclidean",
        prediction_data=True,
    )

    # Vectorizer to remove very common / very rare words
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )

    # KeyBERTInspired uses embedding model for representation fine-tuning.
    # Use cuda:2 to avoid conflicts with spaCy (cuda:0) in later steps.
    from sentence_transformers import SentenceTransformer
    from scottnlp.config import LEGAL_BERT_NAME
    embedding_model = SentenceTransformer(LEGAL_BERT_NAME, device="cuda:2")
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    print(f"Fitting BERTopic on {len(texts)} chunks...")
    topics, probs = topic_model.fit_transform(texts, embeddings)
    print("fit_transform done.")
    topics = np.array(topics)

    print("Getting topic info...")
    topic_info = topic_model.get_topic_info()
    print(f"\nTopics found: {len(topic_info) - 1} (excluding outlier topic -1)")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))
    print("build_topic_model returning.")

    return topic_model, topics


def extract_dynamic_topics(
    topic_model: BERTopic,
    chunks: list[dict],
    topics: np.ndarray,
) -> pd.DataFrame:
    """Extract topic evolution over time using document years.

    Manually computes topic frequency per year since BERTopic's
    topics_over_time has pandas compatibility issues.

    Returns:
        DataFrame with columns: Topic, Words, Frequency, Timestamp
    """
    # Build topic frequency by year manually
    rows = []
    topic_ids = sorted(set(topics))

    # Get topic words for each topic
    topic_words_map = {}
    for tid in topic_ids:
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        topic_words_map[tid] = ", ".join(w for w, _ in words[:5])

    # Count topic occurrences per year
    year_topic_counts = {}
    for c, t in zip(chunks, topics):
        year = c["doc_year"]
        if t == -1:
            continue
        key = (year, int(t))
        year_topic_counts[key] = year_topic_counts.get(key, 0) + 1

    # Build DataFrame
    for (year, tid), count in sorted(year_topic_counts.items()):
        rows.append({
            "Topic": tid,
            "Words": topic_words_map.get(tid, ""),
            "Frequency": count,
            "Timestamp": str(year),
        })

    topics_over_time = pd.DataFrame(rows)
    print(f"\nDynamic topic modeling: {len(topics_over_time)} data points")
    return topics_over_time


def analyze_topic_trajectories(
    topics_over_time: pd.DataFrame,
    topic_model: BERTopic,
) -> dict:
    """Analyze which topics emerge, persist, or decline across time.

    Returns dict with:
        - emerging_topics: topics that appear only post-1992
        - persistent_topics: present across 3+ time periods
        - declining_topics: present early but disappear
        - topic_summaries: per-topic trajectory description
    """
    years = sorted(topics_over_time["Timestamp"].unique())
    topic_ids = [t for t in topics_over_time["Topic"].unique() if t != -1]

    trajectories = {}
    for tid in topic_ids:
        topic_data = topics_over_time[topics_over_time["Topic"] == tid]
        present_years = sorted(topic_data["Timestamp"].unique())
        freq_by_year = dict(zip(topic_data["Timestamp"], topic_data["Frequency"]))

        # Get topic label
        topic_info = topic_model.get_topic_info()
        label_row = topic_info[topic_info["Topic"] == tid]
        label = label_row["Name"].values[0] if len(label_row) > 0 else f"Topic {tid}"

        trajectories[tid] = {
            "label": label,
            "present_years": present_years,
            "num_periods": len(present_years),
            "first_appearance": min(present_years) if present_years else None,
            "last_appearance": max(present_years) if present_years else None,
            "freq_by_year": freq_by_year,
            "peak_year": max(freq_by_year, key=freq_by_year.get) if freq_by_year else None,
        }

    # Classify trajectories
    analysis = {
        "emerging_topics": [],
        "persistent_topics": [],
        "declining_topics": [],
        "topic_summaries": trajectories,
    }

    for tid, traj in trajectories.items():
        first_yr = traj["first_appearance"]
        last_yr = traj["last_appearance"]

        if first_yr and str(first_yr) >= "1992":
            analysis["emerging_topics"].append({
                "topic_id": tid, "label": traj["label"],
                "first_year": first_yr,
            })
        if traj["num_periods"] >= 3:
            analysis["persistent_topics"].append({
                "topic_id": tid, "label": traj["label"],
                "periods": traj["num_periods"],
            })
        if last_yr and str(last_yr) <= "1998" and first_yr and str(first_yr) <= "1872":
            analysis["declining_topics"].append({
                "topic_id": tid, "label": traj["label"],
                "last_year": last_yr,
            })

    return analysis


def save_topic_results(
    topic_model: BERTopic,
    topics: np.ndarray,
    topics_over_time: pd.DataFrame,
    analysis: dict,
    chunks: list[dict],
    output_dir: Path,
) -> None:
    """Save all topic modeling results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Topic assignments per chunk
    assignments = []
    for i, c in enumerate(chunks):
        assignments.append({
            "chunk_id": c["chunk_id"],
            "doc_year": c["doc_year"],
            "doc_title": c["doc_title"],
            "topic_id": int(topics[i]),
        })
    assignments_df = pd.DataFrame(assignments)
    assignments_df.to_csv(output_dir / "topic_assignments.csv", index=False)
    print(f"Saved topic assignments to {output_dir / 'topic_assignments.csv'}")

    # Topics over time
    topics_over_time.to_csv(output_dir / "topics_over_time.csv", index=False)
    print(f"Saved topics over time to {output_dir / 'topics_over_time.csv'}")

    # Topic info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)

    # Topic words
    topic_words = {}
    for tid in topic_info["Topic"].values:
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        topic_words[str(tid)] = [{"word": w, "score": float(s)} for w, s in words]
    with open(output_dir / "topic_words.json", "w") as f:
        json.dump(topic_words, f, indent=2, ensure_ascii=False)

    # Trajectory analysis
    def _make_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    analysis_serializable = _make_serializable(analysis)
    with open(output_dir / "trajectory_analysis.json", "w") as f:
        json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved trajectory analysis to {output_dir / 'trajectory_analysis.json'}")

    # Skip BERTopic binary model save — it hangs on safetensors serialization.
    # All useful outputs (topic_assignments, topic_words, trajectory_analysis) are
    # already saved as CSV/JSON above, which is sufficient for downstream phases.
    print("Skipped BERTopic binary model save (not needed for downstream).")
