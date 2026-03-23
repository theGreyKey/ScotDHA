"""Semantic network construction and centrality analysis."""

import json
from pathlib import Path

import networkx as nx
import pandas as pd


def build_semantic_network(
    dep_frames: list[dict],
    doc_year: int | None = None,
    min_edge_weight: int = 2,
) -> nx.DiGraph:
    """Build directed semantic network from dependency parse frames.

    Nodes: lemmatized lexical items
    Edges: dependency relations, weighted by co-occurrence frequency

    Args:
        dep_frames: list of dependency frame dicts
        doc_year: filter to frames from this year only (None = all years)
        min_edge_weight: minimum frequency to include an edge
    """
    if doc_year is not None:
        dep_frames = [f for f in dep_frames if f["doc_year"] == doc_year]

    G = nx.DiGraph()
    edge_weights = {}

    for frame in dep_frames:
        target = frame["target_lemma"]
        head_lemma = frame["dep_head_lemma"].lower()
        dep_rel = frame["dep_relation"]

        # Add target -> head edge
        edge_key = (target, head_lemma, dep_rel)
        edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

        # Add children -> target edges
        for child in frame["dep_children"]:
            child_lemma = child["text"].lower()
            child_rel = child["dep"]
            child_key = (child_lemma, target, child_rel)
            edge_weights[child_key] = edge_weights.get(child_key, 0) + 1

        # Add modifier edges
        for mod in frame["modifiers"]:
            mod_lemma = mod["lemma"].lower()
            mod_rel = mod["dep"]
            mod_key = (mod_lemma, target, mod_rel)
            edge_weights[mod_key] = edge_weights.get(mod_key, 0) + 1

    # Build graph, filtering by minimum weight
    for (src, tgt, rel), weight in edge_weights.items():
        if weight >= min_edge_weight and src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += weight
                G[src][tgt]["relations"].append(rel)
            else:
                G.add_edge(src, tgt, weight=weight, relations=[rel])

    # Add node attributes
    for node in G.nodes():
        G.nodes[node]["total_degree"] = G.degree(node, weight="weight")

    if doc_year is not None:
        G.graph["year"] = doc_year
    G.graph["num_frames"] = len(dep_frames)

    return G


def compute_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """Compute centrality metrics for all nodes.

    Returns DataFrame with columns: node, pagerank, betweenness, in_degree, out_degree
    """
    if len(G) == 0:
        return pd.DataFrame(columns=["node", "pagerank", "betweenness", "in_degree", "out_degree"])

    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")

    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "pagerank": pagerank.get(node, 0),
            "betweenness": betweenness.get(node, 0),
            "in_degree": G.in_degree(node, weight="weight"),
            "out_degree": G.out_degree(node, weight="weight"),
        })

    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
    return df


def compare_networks_across_eras(
    networks: dict[str, nx.DiGraph],
) -> dict:
    """Compare semantic networks across time periods.

    Args:
        networks: dict mapping era label -> DiGraph

    Returns dict with:
        - centrality_shifts: how node centrality changes over eras
        - emerging_edges: new connections appearing in later eras
        - density_changes: network density per era
    """
    eras = sorted(networks.keys())
    if len(eras) < 2:
        return {"centrality_shifts": {}, "emerging_edges": [], "density_changes": {}}

    # Compute centrality per era
    centrality_by_era = {}
    for era in eras:
        G = networks[era]
        if len(G) > 0:
            centrality_by_era[era] = dict(nx.pagerank(G, weight="weight"))
        else:
            centrality_by_era[era] = {}

    # Track centrality shifts for key nodes
    all_nodes = set()
    for c in centrality_by_era.values():
        all_nodes.update(c.keys())

    centrality_shifts = {}
    for node in all_nodes:
        trajectory = {}
        for era in eras:
            trajectory[era] = centrality_by_era.get(era, {}).get(node, 0)
        centrality_shifts[node] = trajectory

    # Find emerging edges (present in later eras but not earlier)
    emerging_edges = []
    if len(eras) >= 2:
        early_edges = set(networks[eras[0]].edges()) if eras[0] in networks else set()
        for era in eras[1:]:
            G = networks.get(era)
            if G is None:
                continue
            for edge in G.edges():
                if edge not in early_edges:
                    emerging_edges.append({
                        "source": edge[0],
                        "target": edge[1],
                        "first_era": era,
                        "weight": G[edge[0]][edge[1]].get("weight", 1),
                    })

    # Network density per era
    density_changes = {}
    for era in eras:
        G = networks[era]
        density_changes[era] = {
            "num_nodes": len(G),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G) if len(G) > 1 else 0,
        }

    return {
        "centrality_shifts": centrality_shifts,
        "emerging_edges": emerging_edges,
        "density_changes": density_changes,
    }


def build_era_networks(
    dep_frames: list[dict],
    era_definitions: dict[str, list[int]] | None = None,
    min_edge_weight: int = 1,
) -> dict[str, nx.DiGraph]:
    """Build semantic networks per historical era.

    Default eras:
        - pre-devolution: 1707-1997
        - devolution: 1998-2004
        - gaelic-revival: 2005-2022
        - modern: 2023-2025
    """
    if era_definitions is None:
        era_definitions = {
            "pre-devolution (1707-1997)": [1707, 1872, 1992],
            "devolution (1998-2004)": [1998, 2000],
            "gaelic-revival (2005-2022)": [2005, 2010, 2015],
            "modern (2023-2025)": [2023, 2025],
        }

    # Map year -> era
    year_to_era = {}
    for era_label, years in era_definitions.items():
        for y in years:
            year_to_era[y] = era_label

    networks = {}
    for era_label in era_definitions:
        era_frames = [f for f in dep_frames if year_to_era.get(f["doc_year"]) == era_label]
        if era_frames:
            networks[era_label] = build_semantic_network(
                era_frames, doc_year=None, min_edge_weight=min_edge_weight
            )
        else:
            networks[era_label] = nx.DiGraph()

    return networks


def save_network_results(
    networks: dict[str, nx.DiGraph],
    comparison: dict,
    output_dir: Path,
) -> None:
    """Save semantic network results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each network as GraphML (convert list attrs to strings)
    for era_label, G in networks.items():
        safe_name = era_label.replace(" ", "_").replace("(", "").replace(")", "")
        path = output_dir / f"network_{safe_name}.graphml"
        G_copy = G.copy()
        for u, v, data in G_copy.edges(data=True):
            if "relations" in data and isinstance(data["relations"], list):
                data["relations"] = "|".join(data["relations"])
        nx.write_graphml(G_copy, path)

    # Save centrality tables
    for era_label, G in networks.items():
        if len(G) > 0:
            df = compute_centrality_metrics(G)
            safe_name = era_label.replace(" ", "_").replace("(", "").replace(")", "")
            df.to_csv(output_dir / f"centrality_{safe_name}.csv", index=False)

    # Save comparison
    # Need to serialize carefully
    comparison_safe = {
        "density_changes": comparison["density_changes"],
        "emerging_edges": comparison["emerging_edges"],
        "centrality_shifts": {
            k: {str(era): float(v) for era, v in traj.items()}
            for k, traj in comparison["centrality_shifts"].items()
        },
    }
    with open(output_dir / "network_comparison.json", "w") as f:
        json.dump(comparison_safe, f, indent=2)

    print(f"Saved {len(networks)} network graphs to {output_dir}")
