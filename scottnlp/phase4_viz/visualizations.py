"""Phase 4 visualization functions — publication-quality seaborn/matplotlib figures."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import re  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from scottnlp.config import LANG_CANONICAL_MAP  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────

DPI = 300
FORMATS = ["png"]
DOUBLE_COL = (7.0, 5.0)
FULL_PAGE = (7.0, 8.0)

TOPIC_LABELS = {
    0: "Education / Governance",
    1: "Language Rights / Gaelic",
    2: "Union / Constitutional",
    3: "Legal Framework",
}

STRATEGY_LABELS = {
    "nomination": "Nomination",
    "predication": "Predication",
    "argumentation": "Argumentation",
    "perspectivization": "Perspectivization",
    "intensification_mitigation": "Intensification/\nMitigation",
}

STRATEGY_ORDER = [
    "nomination",
    "predication",
    "argumentation",
    "perspectivization",
    "intensification_mitigation",
]

LANGUAGE_MERGE = LANG_CANONICAL_MAP

ERAS = [
    ("pre-devolution_1707-1997", "Pre-devolution\n(1707\u20131997)"),
    ("devolution_1998-2004", "Devolution\n(1998\u20132004)"),
    ("gaelic-revival_2005-2022", "Gaelic Revival\n(2005\u20132022)"),
    ("modern_2023-2025", "Modern\n(2023\u20132025)"),
]

DOC_SHORT_NAMES = {
    1707: "Articles of\nUnion",
    1872: "Education\nAct",
    1992: "European\nCharter",
    1998: "Scotland\nAct",
    2000: "Standards in\nSchools",
    2005: "Gaelic\nAct",
    2010: "Scots WG\nReport",
    2015: "Scots\nPolicy",
    2023: "Gaelic Plan\n2023\u201328",
    2025: "Languages\nAct",
}


# ── Helpers ────────────────────────────────────────────────────────────

def setup_style() -> None:
    """Set global seaborn/matplotlib academic style."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
    })


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as PNG and PDF, then close."""
    for fmt in FORMATS:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.png")


# ── Viz 1: Topic Evolution Heatmap ────────────────────────────────────

def plot_topic_heatmap(
    topics_df: pd.DataFrame,
    topic_words: dict,
    output_dir: Path,
) -> None:
    """Heatmap of topic frequency over time (year × topic)."""
    # Filter to substantive topics (>= 0)
    df = topics_df[topics_df["Topic"] >= 0].copy()

    # Pivot: topic × year
    pivot = df.pivot_table(
        index="Topic", columns="Timestamp", values="Frequency",
        aggfunc="sum", fill_value=0,
    )
    pivot.columns = pivot.columns.astype(int)
    pivot = pivot.sort_index()

    # Build human-readable row labels
    row_labels = []
    for tid in pivot.index:
        label = TOPIC_LABELS.get(tid, f"Topic {tid}")
        # Add top 2 keywords
        words = topic_words.get(str(tid), [])
        if words:
            kws = ", ".join(w["word"] for w in words[:2])
            label = f"{label}\n({kws})"
        row_labels.append(label)
    pivot.index = row_labels

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd", annot=True, fmt="g",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Chunk Frequency", "shrink": 0.8},
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_title("Topic Evolution in Scottish Language Policy (1707\u20132025)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    _save_figure(fig, output_dir, "1_topic_heatmap")


# ── Viz 2: Semantic Network Graphs ────────────────────────────────────

# POS tags to keep in semantic networks (content words only)
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def _build_stopword_nodes(dep_frames: list[dict]) -> set[str]:
    """Build a set of node names that are function words (not content POS)."""
    # Collect POS evidence for each lemma
    lemma_pos: dict[str, Counter] = {}

    for frame in dep_frames:
        # Head
        hl = frame["dep_head_lemma"].lower().strip()
        hp = frame.get("dep_head_pos", "")
        if hl:
            lemma_pos.setdefault(hl, Counter())[hp] += 1

        # Children
        for c in frame.get("dep_children", []):
            cl = c["text"].lower().strip()
            cp = c.get("pos", "")
            if cl:
                lemma_pos.setdefault(cl, Counter())[cp] += 1

        # Modifiers
        for m in frame.get("modifiers", []):
            ml = m["lemma"].lower().strip()
            mp = m.get("pos", "")
            if ml:
                lemma_pos.setdefault(ml, Counter())[mp] += 1

    # A node is a stopword if its most common POS is NOT a content POS
    stopwords = set()
    for lemma, pos_counts in lemma_pos.items():
        if not pos_counts:
            continue
        top_pos = pos_counts.most_common(1)[0][0]
        if top_pos not in _CONTENT_POS:
            stopwords.add(lemma)

    return stopwords


def _remove_overlaps(pos: dict, node_sizes: dict,
                     pr_weights: dict | None = None,
                     iterations: int = 300) -> dict:
    """Iteratively push apart overlapping nodes, then re-center and scale.

    pr_weights: dict mapping node -> PageRank (0-1). High-PR nodes resist
    being pushed away from center.
    """
    pos = {n: list(xy) for n, xy in pos.items()}  # make mutable
    nodes = list(pos.keys())
    if len(nodes) < 2:
        return {n: tuple(xy) for n, xy in pos.items()}

    if pr_weights is None:
        pr_weights = {}

    # Estimate radius — generous padding so labels don't collide
    radii = {}
    for n in nodes:
        s = node_sizes.get(n, 60)
        label_factor = max(1.0, len(n) / 5.5)
        radii[n] = np.sqrt(s) / 85.0 * label_factor

    for it in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                dx = pos[ni][0] - pos[nj][0]
                dy = pos[ni][1] - pos[nj][1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-9
                min_dist = (radii[ni] + radii[nj]) * 1.15  # extra gap
                if dist < min_dist:
                    overlap = (min_dist - dist) / 2.0
                    push = 0.8
                    ux, uy = dx / dist, dy / dist
                    # High-PR nodes move less, low-PR nodes move more
                    wi = 1.0 - pr_weights.get(ni, 0) * 0.85
                    wj = 1.0 - pr_weights.get(nj, 0) * 0.85
                    total = wi + wj + 1e-9
                    pos[ni][0] += ux * overlap * push * (wi / total) * 2
                    pos[ni][1] += uy * overlap * push * (wi / total) * 2
                    pos[nj][0] -= ux * overlap * push * (wj / total) * 2
                    pos[nj][1] -= uy * overlap * push * (wj / total) * 2
                    moved = True
        if not moved:
            break

    # Re-center and scale to fit within [-0.48, 0.48]
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    cx, cy = np.mean(xs), np.mean(ys)
    for n in nodes:
        pos[n][0] -= cx
        pos[n][1] -= cy
    max_extent = max(max(abs(pos[n][0]) for n in nodes),
                     max(abs(pos[n][1]) for n in nodes), 1e-9)
    target = 0.48
    if max_extent > target:
        scale = target / max_extent
        for n in nodes:
            pos[n][0] *= scale
            pos[n][1] *= scale

    return {n: tuple(xy) for n, xy in pos.items()}


def plot_semantic_networks(
    networks_dir: Path,
    output_dir: Path,
    dep_frames: list[dict] | None = None,
    top_n: int = 15,
) -> None:
    """2×2 grid of per-era semantic networks with PageRank sizing."""
    # Build function-word filter from dep_frames
    stopword_nodes = _build_stopword_nodes(dep_frames) if dep_frames else set()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.0))

    for ax, (era_key, era_label) in zip(axes.flat, ERAS):
        graph_path = networks_dir / f"network_{era_key}.graphml"
        cent_path = networks_dir / f"centrality_{era_key}.csv"

        if not graph_path.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(era_label, fontsize=10)
            ax.set_axis_off()
            continue

        G = nx.read_graphml(graph_path)
        cent_df = pd.read_csv(cent_path)

        # Clean node names — drop empty strings
        cent_df["node"] = cent_df["node"].astype(str).str.strip()
        cent_df = cent_df[cent_df["node"].str.len() > 0]

        # Also remove empty-name nodes from the graph itself
        empty_nodes = [n for n in G.nodes() if not str(n).strip()]
        if empty_nodes:
            G.remove_nodes_from(empty_nodes)

        # Remove garbage nodes (whitespace-only, punctuation-only, etc.)
        _garbage_re = re.compile(r"^\W+$|^\s+$")
        cent_df = cent_df[~cent_df["node"].apply(lambda x: bool(_garbage_re.match(x)))]
        garbage_nodes = [n for n in G.nodes() if _garbage_re.match(str(n))]
        if garbage_nodes:
            G.remove_nodes_from(garbage_nodes)

        # Node metrics (needed early for bridge selection)
        pr_map = dict(zip(cent_df["node"], cent_df["pagerank"]))
        bw_map = dict(zip(cent_df["node"], cent_df["betweenness"]))

        # Filter to top-N by PageRank, excluding function words
        cent_df_filtered = cent_df[~cent_df["node"].isin(stopword_nodes)]
        n_show = min(top_n, len(cent_df_filtered))
        top_nodes = cent_df_filtered.nlargest(n_show, "pagerank")["node"].tolist()
        top_set = set(n for n in top_nodes if n in G)

        # Work with undirected view for neighbor/bridge detection
        G_undir = G.to_undirected()

        # Add bridge nodes: if two top nodes share a 1-hop neighbor, include it
        bridge_nodes: set[str] = set()
        top_list = [n for n in top_nodes if n in G_undir]
        for i in range(len(top_list)):
            ni = top_list[i]
            ni_neighbors = set(G_undir.neighbors(ni))
            for j in range(i + 1, len(top_list)):
                nj = top_list[j]
                # Skip if already directly connected
                if G_undir.has_edge(ni, nj):
                    continue
                nj_neighbors = set(G_undir.neighbors(nj))
                shared = (ni_neighbors & nj_neighbors) - top_set - stopword_nodes
                if shared:
                    # Pick the bridge with highest PageRank
                    best = max(shared, key=lambda n: pr_map.get(n, 0))
                    bridge_nodes.add(best)

        all_nodes = top_set | bridge_nodes
        # Use undirected subgraph so all edges between selected nodes appear
        subG = G_undir.subgraph([n for n in all_nodes if n in G_undir]).copy()

        # Remove isolated nodes (degree 0 in the subgraph)
        isolated = [n for n in subG.nodes() if subG.degree(n) == 0]
        if isolated:
            subG.remove_nodes_from(isolated)

        # Inject virtual edges to bind disconnected components together
        virtual_edges: list[tuple[str, str]] = []
        minor_comp_nodes: set[str] = set()
        components = list(nx.connected_components(subG))
        if len(components) > 1:
            # Largest component first
            components.sort(key=len, reverse=True)
            main_comp = components[0]
            main_hub = max(main_comp, key=lambda n: pr_map.get(n, 0))
            for comp in components[1:]:
                minor_comp_nodes |= comp
                comp_hub = max(comp, key=lambda n: pr_map.get(n, 0))
                subG.add_edge(main_hub, comp_hub, weight=0.1)
                virtual_edges.append((main_hub, comp_hub))

        if len(subG) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(era_label, fontsize=10)
            ax.set_axis_off()
            continue

        # Layout — Kamada-Kawai for balanced initial placement, spring for fine-tuning
        pos_init = nx.kamada_kawai_layout(subG, scale=0.6, center=(0, 0))
        k = 2.5 / np.sqrt(max(len(subG), 1))
        pos = nx.spring_layout(
            subG, k=k, iterations=500, seed=42,
            center=(0, 0), scale=0.6, pos=pos_init,
        )

        nodes = list(subG.nodes())
        pr_vals = np.array([pr_map.get(n, 0) for n in nodes])
        bw_vals = np.array([bw_map.get(n, 0) for n in nodes])

        # Normalised PageRank for centering logic
        pr_norm_raw = pr_vals / (pr_vals.max() + 1e-9)
        pr_weight_map = dict(zip(nodes, pr_norm_raw))

        # Scale sizes: power < 0.5 to compress the range between large and small nodes
        pr_norm = pr_norm_raw
        node_sizes = np.power(pr_norm + 0.05, 0.35) * 900 + 60

        # Remove overlaps — high-PR nodes resist being pushed away
        size_map = dict(zip(nodes, node_sizes))
        pos = _remove_overlaps(pos, size_map, pr_weights=pr_weight_map)

        # After overlap removal, apply central gravity to keep nodes cohesive
        # Low-PR nodes move more, high-PR nodes stay put
        # Nodes from smaller (disconnected) components get stronger pull
        all_xs = [pos[n][0] for n in nodes]
        all_ys = [pos[n][1] for n in nodes]
        cx_grav = np.mean(all_xs)
        cy_grav = np.mean(all_ys)
        for i, n in enumerate(nodes):
            base = (1.0 - pr_norm_raw[i]) * 0.35
            # Stronger gravity for disconnected minor-component nodes
            strength = min(base + 0.25, 0.55) if n in minor_comp_nodes else base
            x, y = pos[n]
            pos[n] = (
                x * (1 - strength) + cx_grav * strength,
                y * (1 - strength) + cy_grav * strength,
            )

        # Color by betweenness — light palette for text readability
        cmap = plt.cm.YlGnBu
        if bw_vals.max() > 0:
            norm = mcolors.Normalize(vmin=0, vmax=bw_vals.max())
            node_colors = [cmap(norm(v) * 0.45 + 0.08) for v in bw_vals]
        else:
            node_colors = [cmap(0.2)] * len(nodes)

        # Remove virtual edges before drawing
        for ve in virtual_edges:
            if subG.has_edge(*ve):
                subG.remove_edge(*ve)

        # Draw edges — width, alpha and color scaled by weight
        edge_weights = np.array([
            subG[u][v].get("weight", 1) for u, v in subG.edges()
        ], dtype=float)
        if len(edge_weights) > 0 and edge_weights.max() > 0:
            w_norm = edge_weights / edge_weights.max()
            # Top edges (w_norm > 0.7): moderate straight lines
            # Other edges: thinner curved lines for elegance
            edge_widths = w_norm * 2.0 + 0.6          # 0.6 – 2.6
            edge_alphas = w_norm * 0.25 + 0.35         # 0.35 – 0.60
            edge_colors = [
                (0.40 - 0.15 * wn, 0.40 - 0.15 * wn, 0.40 - 0.15 * wn)
                for wn in w_norm
            ]  # rgb from ~(0.40) to ~(0.25)
        else:
            w_norm = np.array([0.5] * len(subG.edges()))
            edge_widths = [0.8] * len(subG.edges())
            edge_alphas = [0.40] * len(subG.edges())
            edge_colors = ["#666666"] * len(subG.edges())

        # All edges drawn as curves for visual consistency
        for idx, (u, v) in enumerate(subG.edges()):
            nx.draw_networkx_edges(
                subG, pos, ax=ax,
                edgelist=[(u, v)],
                width=edge_widths[idx],
                alpha=float(edge_alphas[idx]),
                arrows=True, arrowsize=0.1,
                edge_color=[edge_colors[idx]],
                connectionstyle="arc3,rad=0.12",
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            subG, pos, ax=ax, nodelist=nodes,
            node_size=node_sizes, node_color=node_colors,
            edgecolors="#888888", linewidths=0.3,
        )

        # Label ALL nodes
        for i, n in enumerate(nodes):
            x, y = pos[n]
            fsize = np.sqrt(pr_norm[i]) * 4.5 + 5.5
            rgba = node_colors[i] if isinstance(node_colors[i], tuple) else (0.5, 0.5, 0.5, 1)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if brightness < 0.45 else "#222222"
            ax.text(
                x, y, n, ha="center", va="center",
                fontsize=fsize, fontfamily="serif", color=txt_color,
                fontweight="bold" if pr_norm[i] > 0.3 else "normal",
            )

        ax.set_title(era_label, fontsize=9)
        ax.set_axis_off()

    fig.suptitle(
        "Semantic Networks of Scottish Language Policy by Era",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.95])
    _save_figure(fig, output_dir, "2_semantic_networks")


# ── Viz 3: DHA Strategy Distribution ──────────────────────────────────

def plot_dha_strategies(
    strategy_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Stacked bar chart of DHA strategy prevalence per document."""
    df = strategy_df.copy()
    df = df.sort_values("doc_year")

    # Pivot: doc_year × strategy → prevalence
    pivot = df.pivot_table(
        index="doc_year", columns="strategy",
        values="prevalence", aggfunc="first", fill_value=0.0,
    )
    # Reorder columns
    ordered_cols = [s for s in STRATEGY_ORDER if s in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot.columns = [STRATEGY_LABELS.get(c, c) for c in pivot.columns]

    # X-tick labels: year + short name
    x_labels = [f"{y}\n{DOC_SHORT_NAMES.get(y, '')}" for y in pivot.index]

    colors = sns.color_palette("colorblind", len(pivot.columns))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    pivot.plot(
        kind="bar", stacked=True, ax=ax,
        color=colors, edgecolor="white", linewidth=0.5,
        width=0.75,
    )

    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Strategy Prevalence")
    ax.set_title("DHA Strategy Distribution Across Scottish Language Policy Documents")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, -0.22),
        ncol=3, frameon=False, fontsize=7,
    )
    sns.despine(ax=ax)

    _save_figure(fig, output_dir, "3_dha_strategies")


# ── Viz 4: Language Agency Chart ──────────────────────────────────────

def plot_language_agency(
    agency_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Dual-panel chart: agency ratio (top) + occurrence count (bottom)."""
    df = agency_df.copy()

    # Merge language variants
    df["language"] = df["term"].map(LANGUAGE_MERGE)
    df = df[df["language"].isin(["Gaelic", "Scots", "English"])]

    # Weighted average per language per year
    def _weighted_agg(g: pd.DataFrame) -> pd.Series:
        total_occ = g["total_occurrences"].sum()
        if total_occ == 0:
            return pd.Series({"agency_ratio": 0.0, "total_occurrences": 0})
        wa = np.average(g["agency_ratio"], weights=g["total_occurrences"])
        return pd.Series({"agency_ratio": wa, "total_occurrences": total_occ})

    merged = (
        df.groupby(["language", "year"])
        .apply(_weighted_agg, include_groups=False)
        .reset_index()
    )

    # Use era-based grouping for cleaner x-axis
    era_bounds = [(1707, 1997), (1998, 2004), (2005, 2022), (2023, 2025)]
    era_names = ["Pre-devolution", "Devolution", "Gaelic Revival", "Modern"]

    def _year_to_era(y: int) -> str:
        for (lo, hi), name in zip(era_bounds, era_names):
            if lo <= y <= hi:
                return name
        return "Unknown"

    merged["era"] = merged["year"].apply(_year_to_era)

    # Aggregate per language per era (weighted)
    era_agg = []
    for (lang, era), g in merged.groupby(["language", "era"]):
        total = g["total_occurrences"].sum()
        if total == 0:
            continue
        wa = np.average(g["agency_ratio"], weights=g["total_occurrences"])
        era_agg.append({
            "language": lang, "era": era,
            "agency_ratio": wa, "total_occurrences": int(total),
        })
    era_df = pd.DataFrame(era_agg)
    era_df["era"] = pd.Categorical(era_df["era"], categories=era_names, ordered=True)

    palette = {"Gaelic": "#0173B2", "Scots": "#DE8F05", "English": "#029E73"}
    bar_width = 0.22
    era_positions = np.arange(len(era_names))
    languages = ["English", "Gaelic", "Scots"]

    fig, (ax_ratio, ax_count) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), height_ratios=[3, 2],
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # ── Top panel: Agency Ratio ──
    for j, lang in enumerate(languages):
        ldf = era_df[era_df["language"] == lang].set_index("era")
        vals, ns = [], []
        for era in era_names:
            if era in ldf.index:
                row = ldf.loc[era]
                vals.append(row["agency_ratio"])
                ns.append(int(row["total_occurrences"]))
            else:
                vals.append(0)
                ns.append(0)

        x_pos = era_positions + (j - 1) * bar_width
        bars = ax_ratio.bar(
            x_pos, vals, width=bar_width,
            color=palette[lang], edgecolor="white", linewidth=0.5,
            label=lang,
        )
        for bar, n, v in zip(bars, ns, vals):
            if n > 0:
                ax_ratio.annotate(
                    f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    fontsize=6, ha="center", va="bottom",
                )

    # Auto-scale y to data with some headroom
    max_ratio = era_df["agency_ratio"].max() if len(era_df) > 0 else 0.5
    ax_ratio.set_ylim(0, max(max_ratio * 1.4, 0.2))
    ax_ratio.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_ratio.set_ylabel("Agency Ratio\n(0 = patient, 1 = agent)", fontsize=8)
    ax_ratio.set_title("Language Agency Across Historical Eras", fontsize=10)
    ax_ratio.legend(title="", frameon=True, framealpha=0.9, fontsize=8)
    sns.despine(ax=ax_ratio)

    # ── Bottom panel: Occurrence Count ──
    for j, lang in enumerate(languages):
        ldf = era_df[era_df["language"] == lang].set_index("era")
        counts = []
        for era in era_names:
            if era in ldf.index:
                counts.append(int(ldf.loc[era, "total_occurrences"]))
            else:
                counts.append(0)

        x_pos = era_positions + (j - 1) * bar_width
        bars = ax_count.bar(
            x_pos, counts, width=bar_width,
            color=palette[lang], edgecolor="white", linewidth=0.5,
        )
        for bar, c in zip(bars, counts):
            if c > 0:
                ax_count.annotate(
                    str(c), (bar.get_x() + bar.get_width() / 2, c),
                    textcoords="offset points", xytext=(0, 3),
                    fontsize=6, ha="center", va="bottom",
                )

    ax_count.set_xticks(era_positions)
    ax_count.set_xticklabels(era_names, fontsize=8)
    ax_count.set_ylabel("Total Occurrences", fontsize=8)
    ax_count.set_xlabel("")
    sns.despine(ax=ax_count)

    _save_figure(fig, output_dir, "4_language_agency")


# ── Viz 5: UMAP Embedding Space ──────────────────────────────────────

def plot_embedding_space(
    embeddings: np.ndarray,
    topic_assignments: pd.DataFrame,
    output_dir: Path,
) -> None:
    """2D UMAP projection of Legal-BERT embeddings, colored by topic."""
    import warnings
    import umap
    from scipy.spatial.distance import pdist, squareform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            metric="cosine", random_state=42,
        )
        coords = reducer.fit_transform(embeddings)

    years = topic_assignments["doc_year"].values
    topics = topic_assignments["topic_id"].values
    titles = topic_assignments["doc_title"].values

    # Color by topic (distinct categorical colors)
    topic_colors = {
        -1: "#BBBBBB",
        0: "#0173B2",
        1: "#DE8F05",
        2: "#029E73",
        3: "#CC78BC",
    }

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    # Plot each topic group
    for tid in [-1, 0, 1, 2, 3]:
        mask = topics == tid
        if not mask.any():
            continue
        label = TOPIC_LABELS.get(tid, "Noise") if tid != -1 else "Noise"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=topic_colors[tid],
            s=14 if tid != -1 else 6,
            alpha=0.75 if tid != -1 else 0.35,
            label=label, zorder=3 if tid != -1 else 1,
        )

    # Annotate each document's sub-cluster with its year
    # Group by (doc_title, topic_id) to find sub-clusters
    df_ann = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "year": years, "topic": topics, "title": titles,
    })
    annotated_positions = []
    for (title, tid), grp in df_ann.groupby(["title", "topic"]):
        if tid == -1 or len(grp) < 2:
            continue
        cx, cy = grp["x"].mean(), grp["y"].mean()
        yr = int(grp["year"].iloc[0])
        # Avoid overlapping annotations
        too_close = False
        for px, py in annotated_positions:
            if abs(cx - px) < 0.8 and abs(cy - py) < 0.5:
                too_close = True
                break
        if too_close:
            continue
        annotated_positions.append((cx, cy))
        color = topic_colors.get(tid, "#444444")
        ax.annotate(
            str(yr), (cx, cy),
            textcoords="offset points", xytext=(0, -10),
            fontsize=6, ha="center", color=color,
            fontweight="bold", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    ax.legend(
        title="Topic", loc="upper left",
        frameon=True, framealpha=0.9, fontsize=7,
    )

    # Semantic axis labels with arrow indicators
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw axis arrows with labels outside the plot
    ax.annotate(
        "Earlier legislation", xy=(0.0, -0.04), xycoords="axes fraction",
        fontsize=7, color="#555555", ha="left",
    )
    ax.annotate(
        "Recent legislation  \u2192", xy=(1.0, -0.04), xycoords="axes fraction",
        fontsize=7, color="#555555", ha="right",
    )
    ax.annotate(
        "Governance / Procedural", xy=(-0.02, 0.0), xycoords="axes fraction",
        fontsize=7, color="#555555", ha="right", va="bottom", rotation=90,
    )
    ax.annotate(
        "Language-specific policy  \u2192", xy=(-0.02, 1.0), xycoords="axes fraction",
        fontsize=7, color="#555555", ha="right", va="top", rotation=90,
    )
    ax.set_title("Legal-BERT Embedding Space of Scottish Language Policy Corpus")
    sns.despine(ax=ax)

    _save_figure(fig, output_dir, "5_embedding_space")


# ── Viz 6: ThemeRiver / Streamgraph — Topos Trajectory ────────────────

# Curated topos colour palette — warm academic tones, high contrast
TOPOS_COLORS = {
    "topos_of_law":        "#1B4F72",  # deep navy
    "topos_of_culture":    "#C0392B",  # crimson
    "topos_of_advantage":  "#27AE60",  # emerald
    "topos_of_authority":  "#E67E22",  # tangerine
    "topos_of_numbers":    "#8E44AD",  # amethyst
    "topos_of_history":    "#A0522D",  # sienna
    "topos_of_threat":     "#E74C3C",  # vermilion
    "topos_of_definition": "#2980B9",  # cerulean
    "topos_of_burden":     "#95A5A6",  # silver
    "topos_of_rights":     "#16A085",  # teal
    "topos_of_heritage":   "#D4AC0D",  # old gold
}

TOPOS_LABELS = {
    "topos_of_law":        "Law",
    "topos_of_culture":    "Culture",
    "topos_of_advantage":  "Advantage",
    "topos_of_authority":  "Authority",
    "topos_of_numbers":    "Numbers",
    "topos_of_history":    "History",
    "topos_of_threat":     "Threat",
    "topos_of_definition": "Definition",
    "topos_of_burden":     "Burden",
    "topos_of_rights":     "Rights",
    "topos_of_heritage":   "Heritage",
}


def plot_topos_streamgraph(
    topos_data: dict,
    output_dir: Path,
) -> None:
    """Streamgraph (ThemeRiver) of topos proportions across 1707–2025.

    X-axis uses equi-spaced document positions (not linear years) to avoid
    the 1707–1872 gap overwhelming the 1992–2025 detail.
    """
    from scipy.interpolate import PchipInterpolator

    by_year = topos_data["by_year"]
    years_raw = sorted(int(y) for y in by_year.keys())
    n_docs = len(years_raw)

    # Equi-spaced positions: 0, 1, 2, ... n_docs-1
    x_pos = np.arange(n_docs, dtype=float)

    # Collect all topoi ranked by total count
    total_counts = topos_data.get("total_topoi_counts", {})
    all_topoi = sorted(total_counts.keys(), key=lambda t: total_counts[t], reverse=True)

    # Build proportions matrix (topoi × documents)
    proportions = {}
    for topos in all_topoi:
        proportions[topos] = []
        for y in years_raw:
            year_data = by_year[str(y)]
            proportions[topos].append(
                year_data.get("topoi_proportions", {}).get(topos, 0.0)
            )

    # Smooth interpolation on equi-spaced grid
    x_smooth = np.linspace(0, n_docs - 1, 400)

    smoothed = {}
    for topos in all_topoi:
        y_raw = np.array(proportions[topos])
        if n_docs >= 4:
            # PCHIP: monotone, no overshoot
            interp = PchipInterpolator(x_pos, y_raw)
            y_smooth = interp(x_smooth)
            y_smooth = np.clip(y_smooth, 0, None)
        else:
            y_smooth = np.interp(x_smooth, x_pos, y_raw)
        smoothed[topos] = y_smooth

    # Stack for streamgraph
    y_stack = np.array([smoothed[t] for t in all_topoi])

    # Symmetric baseline (ThemeRiver / silhouette)
    total = y_stack.sum(axis=0)
    n = len(all_topoi)
    baseline = np.zeros_like(x_smooth)
    for i in range(n):
        baseline -= (n - i) * y_stack[i]
    baseline /= n
    baseline -= baseline.min()
    center_offset = baseline + total / 2
    baseline -= center_offset.mean()

    # ── Draw ──
    fig, ax = plt.subplots(figsize=(11.0, 5.5))

    bottoms = baseline.copy()
    handles = []
    labels_out = []
    for i, topos in enumerate(all_topoi):
        tops = bottoms + y_stack[i]
        color = TOPOS_COLORS.get(topos, f"C{i}")
        label = TOPOS_LABELS.get(topos, topos.replace("topos_of_", "").title())
        h = ax.fill_between(
            x_smooth, bottoms, tops,
            color=color, alpha=0.88, linewidth=0.4, edgecolor="white",
            label=label,
        )
        handles.append(h)
        labels_out.append(label)
        bottoms = tops

    # Era background shading (mapped to equi-spaced positions)
    era_bounds_year = [(1707, 1997), (1998, 2004), (2005, 2022), (2023, 2025)]
    era_colors_bg = ["#DCEEFB", "#D5F5E3", "#FDEBD0", "#FADBD8"]
    era_names_bg = ["Pre-devolution", "Devolution", "Gaelic Revival", "Modern"]

    def _year_to_xpos(yr: int) -> float:
        """Map a year to the equi-spaced position (interpolated)."""
        if yr <= years_raw[0]:
            return 0.0
        if yr >= years_raw[-1]:
            return float(n_docs - 1)
        for j in range(len(years_raw) - 1):
            if years_raw[j] <= yr <= years_raw[j + 1]:
                frac = (yr - years_raw[j]) / (years_raw[j + 1] - years_raw[j])
                return j + frac
        return float(n_docs - 1)

    for (y0, y1), bg_col, era_nm in zip(era_bounds_year, era_colors_bg, era_names_bg):
        xp0 = _year_to_xpos(y0) - 0.3
        xp1 = _year_to_xpos(y1) + 0.3
        ax.axvspan(xp0, xp1, alpha=0.18, color=bg_col, zorder=0)
        # Era label at top
        ax.text(
            (xp0 + xp1) / 2, 1.02, era_nm,
            transform=ax.get_xaxis_transform(),
            fontsize=7, ha="center", va="bottom",
            color="#555555", fontstyle="italic",
        )

    # Turning-point annotations
    turning_points = topos_data.get("turning_points", [])
    tp_positions = []
    for tp in turning_points:
        if tp.get("shift_type") in ("dominant_topos_change", "dominant_change_and_new_topoi"):
            yr = tp["year"]
            xp = _year_to_xpos(yr)
            tp_positions.append((xp, yr, tp["dominant_topos"]))

    # Stagger vertically, alternating top/bottom of the stream
    for ti, (xp, yr, dom_topos) in enumerate(tp_positions):
        ax.axvline(xp, color="#444444", linewidth=0.7, linestyle="--", alpha=0.5, zorder=5)
        dom = TOPOS_LABELS.get(dom_topos, dom_topos)
        # Alternate between top (0.92) and bottom positions
        y_frac = 0.92 if ti % 2 == 0 else 0.08
        va = "top" if ti % 2 == 0 else "bottom"
        ax.text(
            xp, y_frac, f"{yr} → {dom}",
            transform=ax.get_xaxis_transform(),
            fontsize=7, ha="center", va=va,
            color="#333333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#AAAAAA",
                      alpha=0.9, linewidth=0.5),
            zorder=6,
        )

    # X-axis: equi-spaced, labelled with "Year\nDoc short name"
    doc_labels = []
    for y in years_raw:
        short = DOC_SHORT_NAMES.get(y, "")
        # Flatten multiline short names to single line
        short_flat = short.replace("\n", " ")
        doc_labels.append(f"{y}\n{short_flat}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(doc_labels, fontsize=6.5, ha="center")
    ax.set_xlim(-0.5, n_docs - 0.5)

    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("Topos Proportion", fontsize=9)

    ax.set_title(
        "Diachronic Evolution of Argumentative Topoi in Scottish Language Policy\n"
        "(Identity Construction Trajectory, 1707–2025)",
        fontsize=10, pad=18,
    )

    # Legend
    ax.legend(
        handles[::-1], labels_out[::-1],
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5, frameon=True, framealpha=0.9,
        title="Topos", title_fontsize=8,
    )
    sns.despine(ax=ax, left=True)
    fig.subplots_adjust(left=0.04, right=0.83, top=0.85, bottom=0.14)

    _save_figure(fig, output_dir, "6_topos_streamgraph")


# ── Viz 7: Sankey Diagram — SVO Power Flow ────────────────────────────

def _build_sankey_data(svo_triples: list[dict]) -> tuple[list, list, list, list, list]:
    """Aggregate SVO triples into Sankey flows: Agent → Verb → Patient language."""

    # Agent category mapping for cleaner labels
    agent_labels = {
        "AGENTLESS": "Agentless\n(hidden agent)",
        "UNRESOLVED": "Unresolved\nreference",
        "INSTITUTIONAL": "Institutional\nactor",
        "OTHER": "Other",
        "PERSON_GROUP": "Person /\ngroup",
        "LEGAL_PROCESS": "Legal\nprocess",
        "ABSTRACT": "Abstract\nentity",
        "NO_VERB": "No verb\n(nominal)",
    }

    # Only keep triples with a governing verb and a resolved target language
    filtered = []
    for t in svo_triples:
        lang = t.get("target_language")
        if not lang:
            # Try to infer from target_term
            term = (t.get("target_term") or "").lower()
            if "gaelic" in term or "gàidhlig" in term or "bòrd" in term:
                lang = "Gaelic"
            elif "scots" in term:
                lang = "Scots"
            elif "english" in term or "beurla" in term:
                lang = "English"
            else:
                continue
        filtered.append({
            "agent": t.get("agent_category", "UNRESOLVED"),
            "verb": (t.get("governing_verb") or {}).get("lemma", None) if isinstance(t.get("governing_verb"), dict) else t.get("governing_verb"),
            "language": lang,
        })

    if not filtered:
        return [], [], [], [], []

    # Count flows: agent → verb
    from collections import Counter as Ctr
    agent_verb = Ctr()
    verb_lang = Ctr()
    for row in filtered:
        v = row["verb"]
        if v is None:
            continue
        agent_verb[(row["agent"], v)] += 1
        verb_lang[(v, row["language"])] += 1

    # Select top verbs by frequency
    verb_totals = Ctr()
    for (_, v), c in agent_verb.items():
        verb_totals[v] += c
    top_verbs = [v for v, _ in verb_totals.most_common(12)]

    # Select top agent categories (exclude tiny ones)
    agent_totals = Ctr()
    for (a, v), c in agent_verb.items():
        if v in top_verbs:
            agent_totals[a] += c
    top_agents = [a for a, _ in agent_totals.most_common(6)]

    # Build node list
    nodes = []
    node_idx = {}

    for a in top_agents:
        label = agent_labels.get(a, a)
        node_idx[("agent", a)] = len(nodes)
        nodes.append(label)
    for v in top_verbs:
        node_idx[("verb", v)] = len(nodes)
        nodes.append(v)
    for lang in ["Gaelic", "Scots", "English"]:
        node_idx[("lang", lang)] = len(nodes)
        nodes.append(lang)

    sources, targets, values = [], [], []
    # Agent → Verb links
    for (a, v), c in agent_verb.items():
        if a in top_agents and v in top_verbs:
            sources.append(node_idx[("agent", a)])
            targets.append(node_idx[("verb", v)])
            values.append(c)
    # Verb → Language links
    for (v, lang), c in verb_lang.items():
        if v in top_verbs and lang in ["Gaelic", "Scots", "English"]:
            sources.append(node_idx[("verb", v)])
            targets.append(node_idx[("lang", lang)])
            values.append(c)

    # Node colors
    node_colors = []
    for key in list(node_idx.keys()):
        kind = key[0]
        if kind == "agent":
            node_colors.append("#888888")
        elif kind == "verb":
            node_colors.append("#5A9BD5")
        else:
            lang = key[1]
            if lang == "Gaelic":
                node_colors.append("#0173B2")
            elif lang == "Scots":
                node_colors.append("#DE8F05")
            else:
                node_colors.append("#029E73")

    return nodes, sources, targets, values, node_colors


def plot_svo_sankey(
    svo_triples: list[dict],
    output_dir: Path,
) -> None:
    """Sankey-style alluvial diagram of SVO power flows.

    Uses matplotlib to draw an alluvial/Sankey diagram without plotly.
    Left column: Agent categories. Middle column: Top verbs. Right column: Target languages.
    """
    from collections import Counter as Ctr

    # Agent category mapping
    agent_labels = {
        "AGENTLESS": "Agentless (hidden)",
        "UNRESOLVED": "Unresolved ref.",
        "INSTITUTIONAL": "Institutional",
        "OTHER": "Other",
        "PERSON_GROUP": "Person / group",
        "LEGAL_PROCESS": "Legal process",
        "ABSTRACT": "Abstract entity",
        "NO_VERB": "No verb",
    }

    agent_colors = {
        "AGENTLESS": "#C0392B",     # vivid crimson — hidden power
        "UNRESOLVED": "#E67E22",    # warm orange
        "INSTITUTIONAL": "#27AE60", # rich green
        "OTHER": "#8E44AD",         # deep purple
        "PERSON_GROUP": "#D35400",  # burnt sienna
        "LEGAL_PROCESS": "#2980B9", # ocean blue
        "ABSTRACT": "#7F8C8D",      # slate
        "NO_VERB": "#BDC3C7",       # silver
    }

    lang_colors = {
        "Gaelic": "#1A5276",
        "Scots": "#D4AC0D",
        "English": "#1E8449",
    }

    # Process triples
    filtered = []
    for t in svo_triples:
        lang = t.get("target_language")
        if not lang:
            term = (t.get("target_term") or "").lower()
            if "gaelic" in term or "gàidhlig" in term or "bòrd" in term:
                lang = "Gaelic"
            elif "scots" in term:
                lang = "Scots"
            elif "english" in term or "beurla" in term:
                lang = "English"
            else:
                continue
        verb_info = t.get("governing_verb")
        if isinstance(verb_info, dict):
            verb = verb_info.get("lemma")
        elif isinstance(verb_info, str):
            verb = verb_info
        else:
            verb = None
        if verb is None:
            continue
        filtered.append({
            "agent": t.get("agent_category", "UNRESOLVED"),
            "verb": verb,
            "language": lang,
        })

    # Count flows
    agent_verb = Ctr()
    verb_lang = Ctr()
    for row in filtered:
        agent_verb[(row["agent"], row["verb"])] += 1
        verb_lang[(row["verb"], row["language"])] += 1

    # Top verbs & agents
    verb_totals = Ctr()
    for (_, v), c in agent_verb.items():
        verb_totals[v] += c
    top_verbs = [v for v, _ in verb_totals.most_common(10)]

    agent_totals = Ctr()
    for (a, v), c in agent_verb.items():
        if v in top_verbs:
            agent_totals[a] += c
    top_agents = [a for a, _ in agent_totals.most_common(6) if _ > 2]

    languages = ["Gaelic", "Scots", "English"]

    # Column positions
    col_x = [0.0, 0.45, 0.90]  # left, middle, right

    # Compute node heights (proportional to total flow)
    total_flow = sum(c for (a, v), c in agent_verb.items()
                     if a in top_agents and v in top_verbs)
    if total_flow == 0:
        return

    bar_height_total = 0.85  # fraction of figure height for bars
    gap_frac = 0.015  # gap between bars as fraction

    def _layout_column(items: list, totals: dict) -> list:
        """Return list of (item, y_bottom, height) for a column."""
        total = sum(totals.get(it, 0) for it in items)
        if total == 0:
            return []
        gap = gap_frac * len(items)
        usable = bar_height_total - gap
        out = []
        y = (1.0 - bar_height_total) / 2.0
        for it in items:
            h = (totals.get(it, 0) / total) * usable
            out.append((it, y, h))
            y += h + gap_frac
        return out

    # Agent column totals
    agent_col_totals = {}
    for a in top_agents:
        agent_col_totals[a] = sum(c for (aa, v), c in agent_verb.items()
                                  if aa == a and v in top_verbs)

    # Verb column totals
    verb_col_totals = {}
    for v in top_verbs:
        verb_col_totals[v] = sum(c for (a, vv), c in agent_verb.items()
                                 if vv == v and a in top_agents)

    # Language column totals
    lang_col_totals = {}
    for lang in languages:
        lang_col_totals[lang] = sum(c for (v, ll), c in verb_lang.items()
                                    if ll == lang and v in top_verbs)

    agent_layout = _layout_column(top_agents, agent_col_totals)
    verb_layout = _layout_column(top_verbs, verb_col_totals)
    lang_layout = _layout_column(languages, lang_col_totals)

    fig, ax = plt.subplots(figsize=(12.0, 7.5))
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_axis_off()

    bar_width = 0.07

    # Helper: draw a curved band between two bars
    def _draw_band(x0, y0_bot, h0, x1, y1_bot, h1, color, alpha=0.35):
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.path as mpath

        # Build cubic Bezier path for the band
        cx0 = x0 + bar_width
        cx1 = x1
        mid_x = (cx0 + cx1) / 2.0

        # Top edge (from top of source to top of target)
        top_src = y0_bot + h0
        top_tgt = y1_bot + h1
        # Bottom edge
        bot_src = y0_bot
        bot_tgt = y1_bot

        verts = [
            (cx0, bot_src),
            (mid_x, bot_src),
            (mid_x, bot_tgt),
            (cx1, bot_tgt),
            (cx1, top_tgt),
            (mid_x, top_tgt),
            (mid_x, top_src),
            (cx0, top_src),
            (cx0, bot_src),
        ]
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.LINETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CLOSEPOLY,
        ]
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(
            path, facecolor=color, edgecolor="none",
            alpha=alpha, zorder=1,
        )
        ax.add_patch(patch)

    # Draw agent bars
    agent_cursors = {}
    for a, y, h in agent_layout:
        color = agent_colors.get(a, "#888888")
        rect = mpatches.FancyBboxPatch(
            (col_x[0], y), bar_width, h,
            boxstyle="round,pad=0.003", facecolor=color,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.add_patch(rect)
        label = agent_labels.get(a, a)
        ax.text(col_x[0] - 0.01, y + h / 2, label,
                ha="right", va="center", fontsize=7, fontfamily="serif")
        agent_cursors[a] = y  # cursor for band stacking

    # Draw verb bars — gradient tint from steel blue to indigo
    verb_cursors_left = {}
    verb_cursors_right = {}
    verb_bar_colors = {}
    n_verbs = len(verb_layout)
    for vi, (v, y, h) in enumerate(verb_layout):
        # gradient from steel-blue to indigo
        t = vi / max(n_verbs - 1, 1)
        r = int(0x34 + (0x5B - 0x34) * (1 - t))
        g = int(0x49 + (0x8D - 0x49) * (1 - t))
        b = int(0x5E + (0xC8 - 0x5E) * (1 - t))
        vcolor = f"#{r:02X}{g:02X}{b:02X}"
        verb_bar_colors[v] = vcolor
        rect = mpatches.FancyBboxPatch(
            (col_x[1], y), bar_width, h,
            boxstyle="round,pad=0.003", facecolor=vcolor,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(col_x[1] + bar_width / 2, y + h / 2, v,
                ha="center", va="center", fontsize=7, fontfamily="serif",
                color="white", fontweight="bold")
        verb_cursors_left[v] = y
        verb_cursors_right[v] = y

    # Draw language bars
    lang_cursors = {}
    for lang, y, h in lang_layout:
        color = lang_colors.get(lang, "#888888")
        rect = mpatches.FancyBboxPatch(
            (col_x[2], y), bar_width, h,
            boxstyle="round,pad=0.003", facecolor=color,
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(col_x[2] + bar_width + 0.01, y + h / 2, lang,
                ha="left", va="center", fontsize=8, fontfamily="serif",
                fontweight="bold", color=color)
        lang_cursors[lang] = y

    # Draw bands: agent → verb
    for a, a_y, a_h in agent_layout:
        a_total = agent_col_totals[a]
        if a_total == 0:
            continue
        for v, v_y, v_h in verb_layout:
            flow = agent_verb.get((a, v), 0)
            if flow == 0:
                continue
            band_h_src = (flow / a_total) * a_h
            v_total = verb_col_totals[v]
            band_h_tgt = (flow / v_total) * v_h if v_total > 0 else 0

            _draw_band(
                col_x[0], agent_cursors[a], band_h_src,
                col_x[1], verb_cursors_left[v], band_h_tgt,
                color=agent_colors.get(a, "#888888"), alpha=0.45,
            )
            agent_cursors[a] += band_h_src
            verb_cursors_left[v] += band_h_tgt

    # Draw bands: verb → language
    for v, v_y, v_h in verb_layout:
        v_total = verb_col_totals[v]
        if v_total == 0:
            continue
        for lang, l_y, l_h in lang_layout:
            flow = verb_lang.get((v, lang), 0)
            if flow == 0:
                continue
            band_h_src = (flow / v_total) * v_h
            l_total = lang_col_totals[lang]
            band_h_tgt = (flow / l_total) * l_h if l_total > 0 else 0

            _draw_band(
                col_x[1], verb_cursors_right[v], band_h_src,
                col_x[2], lang_cursors[lang], band_h_tgt,
                color=lang_colors.get(lang, "#888888"), alpha=0.45,
            )
            verb_cursors_right[v] += band_h_src
            lang_cursors[lang] += band_h_tgt

    # Column headers
    headers = [
        (col_x[0] + bar_width / 2, "Agent Category"),
        (col_x[1] + bar_width / 2, "Governing Verb"),
        (col_x[2] + bar_width / 2, "Target Language"),
    ]
    for hx, label in headers:
        ax.text(hx, 0.98, label, ha="center", va="bottom",
                fontsize=9, fontweight="bold", fontfamily="serif")

    ax.set_title(
        "SVO Power Triangle: Agency Flows in Scottish Language Policy Discourse",
        fontsize=11, pad=15, fontfamily="serif",
    )

    fig.tight_layout()
    _save_figure(fig, output_dir, "7_svo_sankey")


# ── Viz 8: Diverging Bar Chart — Markedness Asymmetry ─────────────────

def plot_markedness_diverging(
    strategy_targeting: dict,
    output_dir: Path,
) -> None:
    """Diverging butterfly bar chart showing DHA strategy targeting asymmetry.

    Centre axis = 0. Right = minority language frequency. Left = English.
    Visually demonstrates the extreme asymmetry of English "invisibility".
    """
    strategies_order = [
        "predication",
        "nomination",
        "argumentation",
        "perspectivization",
        "intensification_mitigation",
    ]

    strategy_display = {
        "predication": "Predication",
        "nomination": "Nomination",
        "argumentation": "Argumentation",
        "perspectivization": "Perspectivization",
        "intensification_mitigation": "Intensification /\nMitigation",
    }

    rows = []
    for strat in strategies_order:
        if strat not in strategy_targeting:
            continue
        strat_data = strategy_targeting[strat]
        eng = strat_data.get("English", {}).get("count", 0)
        gael = strat_data.get("Gaelic", {}).get("count", 0)
        scots = strat_data.get("Scots", {}).get("count", 0)
        minority = gael + scots
        ratio = minority / eng if eng > 0 else float("inf")
        rows.append({
            "strategy": strategy_display.get(strat, strat),
            "english": eng,
            "gaelic": gael,
            "scots": scots,
            "minority": minority,
            "ratio": ratio,
        })

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    y_pos = np.arange(len(rows))
    bar_height = 0.55

    for i, row in enumerate(rows):
        # English (left — negative direction)
        ax.barh(
            i, -row["english"], height=bar_height,
            color="#029E73", edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        # Gaelic (right)
        ax.barh(
            i, row["gaelic"], height=bar_height,
            color="#0173B2", edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        # Scots (right, stacked on top of Gaelic)
        ax.barh(
            i, row["scots"], left=row["gaelic"], height=bar_height,
            color="#DE8F05", edgecolor="white", linewidth=0.5,
            zorder=3,
        )

        # Ratio annotation
        if row["ratio"] != float("inf"):
            ratio_text = f"{row['ratio']:.1f}×"
        else:
            ratio_text = "∞×"
        max_x = row["gaelic"] + row["scots"]
        ax.text(
            max_x + 8, i, ratio_text,
            ha="left", va="center", fontsize=8,
            fontweight="bold", color="#333333",
        )

        # Count labels
        if row["english"] > 0:
            ax.text(
                -row["english"] / 2, i, str(row["english"]),
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold",
            )
        if row["gaelic"] > 10:
            ax.text(
                row["gaelic"] / 2, i, str(row["gaelic"]),
                ha="center", va="center", fontsize=6.5, color="white",
            )
        if row["scots"] > 10:
            ax.text(
                row["gaelic"] + row["scots"] / 2, i, str(row["scots"]),
                ha="center", va="center", fontsize=6.5, color="white",
            )

    # Centre line
    ax.axvline(0, color="#333333", linewidth=1.2, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["strategy"] for r in rows], fontsize=8)
    ax.invert_yaxis()

    # X-axis: absolute values on both sides
    max_val = max(r["gaelic"] + r["scots"] for r in rows)
    max_eng = max(r["english"] for r in rows)
    xlim = max(max_val, max_eng) * 1.3
    ax.set_xlim(-xlim, xlim)

    # Custom tick labels (absolute values)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(abs(t))) for t in ticks], fontsize=7)

    # Direction labels — place inside the axes area using axes fraction
    ax.text(
        0.25, 1.02, "\u2190 English",
        ha="center", va="bottom", fontsize=9,
        color="#029E73", fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.75, 1.02, "Minority Languages \u2192",
        ha="center", va="bottom", fontsize=9,
        color="#555555", fontweight="bold",
        transform=ax.transAxes,
    )

    # Ratio header
    ax.text(
        xlim * 1.05, -0.8, "Ratio",
        ha="left", va="center", fontsize=8,
        fontweight="bold", color="#333333",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#0173B2", label="Gaelic"),
        mpatches.Patch(facecolor="#DE8F05", label="Scots"),
        mpatches.Patch(facecolor="#029E73", label="English"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=7.5, frameon=True, framealpha=0.9,
    )

    ax.set_title(
        "DHA Strategy Targeting Asymmetry: Minority Languages vs. English\n"
        "(Markedness Theory — English Invisibility in Policy Discourse)",
        fontsize=10, pad=25,
    )
    ax.set_xlabel("Frequency of Strategy Application (chunk count)", fontsize=8)
    sns.despine(ax=ax)

    _save_figure(fig, output_dir, "8_markedness_diverging")


# ── Viz 9: Network Growth Metrics ─────────────────────────────────────

def plot_network_growth(
    networks_dir: Path,
    output_dir: Path,
) -> None:
    """Multi-panel chart quantifying semantic network evolution across eras.

    Shows nodes, edges, density, hub dominance (top PageRank), and
    average betweenness — replacing the redundant diachronic network graph.
    """
    import json as _json

    era_keys = [
        "pre-devolution_1707-1997",
        "devolution_1998-2004",
        "gaelic-revival_2005-2022",
        "modern_2023-2025",
    ]
    era_labels_short = [
        "Pre-devolution\n(1707–1997)",
        "Devolution\n(1998–2004)",
        "Gaelic Revival\n(2005–2022)",
        "Modern\n(2023–2025)",
    ]

    # Collect statistics
    rows = []
    for ek in era_keys:
        cent_path = networks_dir / f"centrality_{ek}.csv"
        graph_path = networks_dir / f"network_{ek}.graphml"
        if not cent_path.exists():
            continue
        cdf = pd.read_csv(cent_path)
        cdf["node"] = cdf["node"].astype(str).str.strip()
        cdf = cdf[cdf["node"].str.len() > 0]

        G = nx.read_graphml(graph_path) if graph_path.exists() else None
        n_nodes = len(G) if G else len(cdf)
        n_edges = G.number_of_edges() if G else 0
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

        top_row = cdf.nlargest(1, "pagerank").iloc[0]
        rows.append({
            "era": ek,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "top_pagerank": top_row["pagerank"],
            "top_node": top_row["node"],
            "avg_betweenness": cdf["betweenness"].mean(),
            "hub_dominance": top_row["pagerank"] / cdf["pagerank"].mean() if cdf["pagerank"].mean() > 0 else 0,
        })

    df = pd.DataFrame(rows)
    x = np.arange(len(df))

    # ── 2×2 panel layout ──
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    era_color = ["#2166AC", "#4DAF4A", "#FF7F00", "#E31A1C"]

    # Panel A: Nodes & Edges (dual-axis bar)
    ax1 = axes[0, 0]
    w = 0.35
    bars_n = ax1.bar(x - w / 2, df["n_nodes"], w, color=era_color, alpha=0.85,
                     edgecolor="white", linewidth=0.5, label="Nodes")
    ax1_r = ax1.twinx()
    bars_e = ax1_r.bar(x + w / 2, df["n_edges"], w, color=era_color, alpha=0.45,
                       edgecolor="white", linewidth=0.5, hatch="//", label="Edges")
    for i, (n, e) in enumerate(zip(df["n_nodes"], df["n_edges"])):
        ax1.text(x[i] - w / 2, n + 3, str(n), ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax1_r.text(x[i] + w / 2, e + 3, str(e), ha="center", va="bottom", fontsize=7, color="#555555")
    ax1.set_ylabel("Nodes (N)", fontsize=8)
    ax1_r.set_ylabel("Edges (E)", fontsize=8, color="#555555")
    ax1.set_title("A. Network Scale", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(era_labels_short, fontsize=6.5)
    # Combined legend
    ax1.legend([bars_n, bars_e], ["Nodes", "Edges"], loc="upper left", fontsize=7, framealpha=0.9)
    sns.despine(ax=ax1, right=False)

    # Panel B: Density
    ax2 = axes[0, 1]
    ax2.plot(x, df["density"], marker="o", markersize=8, linewidth=2.5,
             color="#2166AC", zorder=3)
    ax2.fill_between(x, 0, df["density"], alpha=0.15, color="#2166AC")
    for i, d in enumerate(df["density"]):
        ax2.text(x[i], d + 0.0005, f"{d:.4f}", ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel("Graph Density (ρ)", fontsize=8)
    ax2.set_title("B. Network Density", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(era_labels_short, fontsize=6.5)
    sns.despine(ax=ax2)

    # Panel C: Hub Dominance (top PageRank / mean PageRank)
    ax3 = axes[1, 0]
    bars_h = ax3.bar(x, df["hub_dominance"], color=era_color, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
    for i, (hd, tn) in enumerate(zip(df["hub_dominance"], df["top_node"])):
        ax3.text(x[i], hd + 0.3, f"{hd:.1f}×", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax3.text(x[i], hd / 2, f'"{tn}"', ha="center", va="center", fontsize=6.5,
                 color="white", fontstyle="italic")
    ax3.set_ylabel("Hub Dominance\n(top PageRank / mean)", fontsize=8)
    ax3.set_title("C. Hub Dominance", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(era_labels_short, fontsize=6.5)
    sns.despine(ax=ax3)

    # Panel D: Avg Betweenness Centrality
    ax4 = axes[1, 1]
    ax4.plot(x, df["avg_betweenness"], marker="s", markersize=8, linewidth=2.5,
             color="#E31A1C", zorder=3)
    ax4.fill_between(x, 0, df["avg_betweenness"], alpha=0.15, color="#E31A1C")
    for i, b in enumerate(df["avg_betweenness"]):
        ax4.text(x[i], b + 0.0001, f"{b:.4f}", ha="center", va="bottom", fontsize=7)
    ax4.set_ylabel("Avg. Betweenness Centrality", fontsize=8)
    ax4.set_title("D. Bridging Structure", fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(era_labels_short, fontsize=6.5)
    sns.despine(ax=ax4)

    fig.suptitle(
        "Semantic Network Structural Evolution in Scottish Language Policy (1707–2025)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.96])
    _save_figure(fig, output_dir, "9_network_growth")
