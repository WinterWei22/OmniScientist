"""Knowledge graph tools for biomedical triple files and PrimeKG-style CSVs."""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

_kg_cache: dict[str, nx.MultiDiGraph] = {}
_metapath_graph_cache: dict[tuple[str, bool], nx.DiGraph] = {}


def _edge_items(edge_data: Any) -> list[dict[str, Any]]:
    if edge_data is None:
        return []
    if isinstance(edge_data, dict) and edge_data and all(
        isinstance(v, dict) for v in edge_data.values()
    ):
        return list(edge_data.values())
    if isinstance(edge_data, dict):
        return [edge_data]
    return [edge_data]


def _graph_stats(G: nx.MultiDiGraph, kg_path: str) -> dict:
    relations = sorted({str(data.get("relation", "unknown")) for _, _, data in G.edges(data=True)})
    return {
        "success": True,
        "query_info": {
            "kg_path": kg_path,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        },
        "result": {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "relation_types": relations,
            "relation_count": len(relations),
        },
    }


def _format_primekg_node_id(entity_type: str | None, entity_id: str | None) -> str:
    entity_id = (entity_id or "").strip()
    entity_type = (entity_type or "").strip()
    if not entity_id:
        return ""
    return f"{entity_type}:{entity_id}" if entity_type else entity_id


def _infer_node_type(node_id: str) -> str:
    """Infer a node type from ``type:id`` identifiers used by triple files."""

    prefix, separator, _identifier = node_id.partition(":")
    return prefix if separator and prefix else "unknown"


def _node_type(G: nx.MultiDiGraph, node_id: str) -> str:
    """Return the stored node type, falling back to the identifier prefix."""

    node_data = G.nodes[node_id]
    return str(node_data.get("type") or _infer_node_type(node_id))


def _build_metapath_search_graph(G: nx.MultiDiGraph, bidirectional: bool) -> nx.DiGraph:
    """Build the relation-collapsed graph used by ``extract_metapaths``."""

    simple_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        rel = str(data.get("relation", "unknown"))
        if simple_G.has_edge(u, v):
            simple_G[u][v]["relations"].add(rel)
        else:
            simple_G.add_edge(u, v, relations={rel})
        if bidirectional:
            rev_rel = f"INV_{rel}"
            if simple_G.has_edge(v, u):
                simple_G[v][u]["relations"].add(rev_rel)
            else:
                simple_G.add_edge(v, u, relations={rev_rel})
    return simple_G


def _get_metapath_search_graph(kg_path: str, G: nx.MultiDiGraph, bidirectional: bool) -> nx.DiGraph:
    """Get or build the cached search graph used for metapath enumeration."""

    cache_key = (os.path.abspath(kg_path), bidirectional)
    cached = _metapath_graph_cache.get(cache_key)
    if cached is not None:
        return cached
    simple_G = _build_metapath_search_graph(G, bidirectional)
    _metapath_graph_cache[cache_key] = simple_G
    return simple_G


def _reverse_shortest_path_lengths(G: nx.DiGraph, target: str) -> dict[str, int]:
    """Return shortest directed distances to ``target`` in edge count."""

    if target not in G:
        return {}
    reverse_G = G.reverse(copy=False)
    return {node: int(distance) for node, distance in nx.single_source_shortest_path_length(reverse_G, target).items()}


def _default_kg_paths() -> list[str]:
    return [
        "./data/omniInfra_data/data_lake/primekg.csv",
        "./data/omniInfra_data/data_lake/kg.csv",
        "./data/kg.csv",
    ]


def _two_sweep_diameter_estimate(G: nx.Graph, rng: random.Random) -> int:
    """Estimate graph diameter with a few linear-time BFS sweeps."""

    if G.number_of_nodes() <= 1:
        return 0

    start = rng.choice(list(G.nodes()))
    best = 0
    current = start
    for _ in range(3):
        lengths = nx.single_source_shortest_path_length(G, current)
        if not lengths:
            return best
        farthest_node, farthest_distance = max(lengths.items(), key=lambda item: item[1])
        best = max(best, int(farthest_distance))
        current = farthest_node
    return best


def load_biomedical_kg(
    kg_path: str,
    format: str = "csv",
    delimiter: str = "\t",
    has_header: bool = False,
    schema: str = "auto",
    use_cache: bool = True,
) -> dict:
    """Load a biomedical knowledge graph from a triple file or PrimeKG-style CSV."""

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": {"kg_path": kg_path}}
    if format not in {"csv", "tsv"}:
        return {
            "success": False,
            "error": f"Unsupported format '{format}'. Use 'csv' or 'tsv'.",
            "query_info": {"kg_path": kg_path, "format": format},
        }
    if schema not in {"auto", "triples", "primekg"}:
        return {
            "success": False,
            "error": f"Unsupported schema '{schema}'. Use 'auto', 'triples', or 'primekg'.",
            "query_info": {"kg_path": kg_path, "schema": schema},
        }

    cache_key = os.path.abspath(kg_path)
    if use_cache and cache_key in _kg_cache:
        return _graph_stats(_kg_cache[cache_key], kg_path)

    effective_delimiter = delimiter
    if format == "tsv":
        effective_delimiter = "\t"

    G = nx.MultiDiGraph()
    try:
        with open(kg_path, encoding="utf-8") as fh:
            first_line = fh.readline()
            if not first_line:
                return {"success": False, "error": "KG file is empty", "query_info": {"kg_path": kg_path}}

            if effective_delimiter not in first_line and "," in first_line:
                effective_delimiter = ","

            first_parts = [part.strip().lstrip("\ufeff") for part in next(csv.reader([first_line], delimiter=effective_delimiter))]
            header_lower = [part.lower() for part in first_parts]
            primekg_header = {"x_id", "x_type", "relation", "y_id", "y_type"}.issubset(header_lower)
            active_schema = "primekg" if schema == "auto" and primekg_header else schema
            active_has_header = has_header or active_schema == "primekg"

            if active_schema == "primekg":
                if not primekg_header:
                    return {
                        "success": False,
                        "error": "PrimeKG schema requires x_id, x_type, relation, y_id, y_type columns.",
                        "query_info": {"kg_path": kg_path, "schema": schema},
                    }
                reader = csv.DictReader(fh, fieldnames=first_parts, delimiter=effective_delimiter)
                for row in reader:
                    head_id = _format_primekg_node_id(row.get("x_type"), row.get("x_id"))
                    tail_id = _format_primekg_node_id(row.get("y_type"), row.get("y_id"))
                    relation = (row.get("relation") or row.get("display_relation") or "").strip()
                    if not head_id or not tail_id or not relation:
                        continue
                    G.add_node(head_id, type=(row.get("x_type") or "").strip() or None, source_id=(row.get("x_id") or "").strip() or None, name=(row.get("x_name") or "").strip() or None)
                    G.add_node(tail_id, type=(row.get("y_type") or "").strip() or None, source_id=(row.get("y_id") or "").strip() or None, name=(row.get("y_name") or "").strip() or None)
                    G.add_edge(head_id, tail_id, relation=relation, display_relation=(row.get("display_relation") or "").strip() or relation)
            else:
                lines = fh if active_has_header else [first_line, *fh]
                reader = csv.reader(lines, delimiter=effective_delimiter)
                for parts in reader:
                    if len(parts) < 3:
                        continue
                    head, relation, tail = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    if head and relation and tail:
                        G.add_node(head, type=_infer_node_type(head))
                        G.add_node(tail, type=_infer_node_type(tail))
                        G.add_edge(head, tail, relation=relation)
    except Exception as exc:
        return {"success": False, "error": f"Failed to parse KG file: {exc}", "query_info": {"kg_path": kg_path, "schema": schema}}

    if use_cache:
        _kg_cache[cache_key] = G
        _metapath_graph_cache.pop((cache_key, False), None)
        _metapath_graph_cache.pop((cache_key, True), None)
    return _graph_stats(G, kg_path)


def random_walk_with_restart(
    kg_path: str,
    seed_nodes: list[str],
    restart_prob: float = 0.7,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    top_k: int = 200,
    return_subgraph: bool = True,
) -> dict:
    """Run Random Walk with Restart on a biomedical knowledge graph."""

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes}}
    if not seed_nodes:
        return {"success": False, "error": "seed_nodes must be a non-empty list", "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes}}
    if not (0 < restart_prob < 1):
        return {"success": False, "error": f"restart_prob must be in (0, 1), got {restart_prob}", "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes, "restart_prob": restart_prob}}
    if top_k < 1:
        return {"success": False, "error": f"top_k must be >= 1, got {top_k}", "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes, "top_k": top_k}}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": {"kg_path": kg_path}}

    nodes = list(G.nodes())
    node2idx = {node: idx for idx, node in enumerate(nodes)}
    seed_indices = [node2idx[s] for s in seed_nodes if s in node2idx]
    missing_seeds = [s for s in seed_nodes if s not in node2idx]
    if not seed_indices:
        return {"success": False, "error": f"None of the seed nodes found in KG. Missing: {missing_seeds}", "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes}}

    weighted = defaultdict(int)
    for u, v, _data in G.edges(data=True):
        key = (node2idx[u], node2idx[v])
        weighted[key] += 1

    n = len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for (u, v), weight in weighted.items():
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([float(weight), float(weight)])
    adj = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1.0

    p = np.zeros(n, dtype=np.float64)
    p[seed_indices] = 1.0 / len(seed_indices)
    restart = np.zeros(n, dtype=np.float64)
    restart[seed_indices] = restart_prob / len(seed_indices)
    walk_prob = 1.0 - restart_prob

    delta = float("inf")
    for it in range(max_iter):
        p_next = walk_prob * adj.dot(p / degrees) + restart
        delta = float(np.abs(p_next - p).sum())
        p = p_next
        if delta < epsilon:
            break
    else:
        it = max_iter - 1

    seed_set = set(seed_indices)
    candidate_indices = np.array([idx for idx in range(n) if idx not in seed_set], dtype=int)
    if candidate_indices.size == 0:
        return {
            "success": True,
            "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes, "restart_prob": restart_prob, "top_k": top_k, "missing_seeds": missing_seeds or None},
            "result": {"stationary_probs": {}, "convergence": {"iterations": it + 1, "delta": delta}, "node_count_in_subgraph": 0, "edge_count_in_subgraph": 0},
        }

    top_k_actual = min(top_k, candidate_indices.size)
    top_local = np.argpartition(-p[candidate_indices], top_k_actual - 1)[:top_k_actual]
    top_global = candidate_indices[top_local][np.argsort(-p[candidate_indices][top_local])]
    stationary_probs = {nodes[i]: float(p[i]) for i in top_global if float(p[i]) > 0}

    result: dict[str, Any] = {
        "stationary_probs": stationary_probs,
        "convergence": {"iterations": it + 1, "delta": delta},
    }
    if return_subgraph:
        sub_nodes = set(seed_indices) | set(int(i) for i in top_global.tolist())
        edges = []
        for u_idx in sub_nodes:
            for v_idx in sub_nodes:
                edge_data = G.get_edge_data(nodes[u_idx], nodes[v_idx])
                for ed in _edge_items(edge_data):
                    if "relation" in ed:
                        edges.append({"head": nodes[u_idx], "relation": ed.get("relation", "unknown"), "tail": nodes[v_idx]})
        result["subgraph"] = {"nodes": [nodes[i] for i in sorted(sub_nodes)], "edges": edges}
        result["node_count_in_subgraph"] = len(sub_nodes)
        result["edge_count_in_subgraph"] = len(edges)

    return {
        "success": True,
        "query_info": {"kg_path": kg_path, "seed_nodes": seed_nodes, "restart_prob": restart_prob, "top_k": top_k, "missing_seeds": missing_seeds or None},
        "result": result,
    }


def extract_khop_subgraph(
    kg_path: str,
    seed_nodes: list[str],
    k: int = 2,
    relation_filter: list[str] | None = None,
    bidirectional: bool = True,
) -> dict:
    """Extract the k-hop neighbourhood subgraph around a set of seed entities."""

    query_info = {
        "kg_path": kg_path,
        "seed_nodes": seed_nodes,
        "k": k,
        "bidirectional": bidirectional,
        "relation_filter": relation_filter,
    }
    if not isinstance(kg_path, str) or not kg_path:
        return {"success": False, "error": "kg_path must be a non-empty string", "query_info": query_info}
    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": query_info}
    if not isinstance(seed_nodes, list) or not seed_nodes or not all(isinstance(seed, str) and seed for seed in seed_nodes):
        return {"success": False, "error": "seed_nodes must be a non-empty list of non-empty strings", "query_info": query_info}
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        return {"success": False, "error": f"k must be a non-negative integer, got {k!r}", "query_info": query_info}
    if relation_filter is not None and (
        not isinstance(relation_filter, list)
        or not all(isinstance(relation, str) and relation for relation in relation_filter)
    ):
        return {
            "success": False,
            "error": "relation_filter must be None or a list of non-empty strings",
            "query_info": query_info,
        }
    if not isinstance(bidirectional, bool):
        return {"success": False, "error": "bidirectional must be a boolean", "query_info": query_info}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": {"kg_path": kg_path}}

    valid_seeds = list(dict.fromkeys(seed for seed in seed_nodes if seed in G))
    missing_seeds = list(dict.fromkeys(seed for seed in seed_nodes if seed not in G))
    if not valid_seeds:
        return {"success": False, "error": f"None of the seed nodes found in KG. Missing: {missing_seeds}", "query_info": query_info}

    relation_filter_set = set(relation_filter) if relation_filter is not None else None
    visited = set(valid_seeds)
    frontier = set(valid_seeds)
    hops_distribution: dict[int, int] = {0: len(valid_seeds)}

    for hop in range(1, k + 1):
        new_frontier: set[str] = set()
        for node in frontier:
            neighbors = set(G.successors(node))
            if bidirectional:
                neighbors |= set(G.predecessors(node))
            for nb in neighbors:
                if nb in visited:
                    continue
                if relation_filter_set is not None:
                    directions = [(node, nb)]
                    if bidirectional:
                        directions.append((nb, node))
                    if not any(
                        str(edge_data.get("relation", "unknown")) in relation_filter_set
                        for src, dst in directions
                        for edge_data in _edge_items(G.get_edge_data(src, dst))
                    ):
                        continue
                new_frontier.add(nb)
        if not new_frontier:
            break
        visited |= new_frontier
        frontier = new_frontier
        hops_distribution[hop] = len(new_frontier)

    edges = []
    for u, v, edge_data in G.subgraph(visited).edges(data=True):
        relation = str(edge_data.get("relation", "unknown"))
        if relation_filter_set is None or relation in relation_filter_set:
            edges.append({"head": u, "relation": relation, "tail": v})
    edges.sort(key=lambda edge: (edge["head"], edge["relation"], edge["tail"]))

    query_info["missing_seeds"] = missing_seeds or None
    return {
        "success": True,
        "query_info": query_info,
        "result": {"subgraph": {"nodes": sorted(visited), "edges": edges}, "node_count": len(visited), "edge_count_in_subgraph": len(edges), "hops_distribution": hops_distribution},
    }


def extract_metapaths(
    kg_path: str,
    head_entity: str,
    tail_entity: str,
    max_length: int = 4,
    max_paths: int = 100,
    bidirectional: bool = True,
) -> dict:
    """Find metapath patterns and their path instances connecting two entities.

    A metapath is a typed walk schema: an ordered node-type sequence together
    with the relation traversed between each pair of adjacent node types. For
    example, ``drug -[binds]-> gene/protein -[associated_with]-> disease``
    distinguishes the same relation sequence used over a different set of
    node types. When ``bidirectional`` is True, reverse traversals are
    represented with an ``INV_`` prefix.

    The returned ``pattern`` is the typed schema. ``relation_pattern`` and
    ``node_type_pattern`` expose its two components separately for downstream
    filtering. ``instance_count`` counts concrete relation paths, while
    ``examples`` contains up to three representative node-ID paths.
    """

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if max_length < 1:
        return {"success": False, "error": f"max_length must be >= 1, got {max_length}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_length": max_length}}
    if max_paths < 1:
        return {"success": False, "error": f"max_paths must be >= 1, got {max_paths}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_paths": max_paths}}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": {"kg_path": kg_path}}
    if head_entity not in G:
        return {"success": False, "error": f"head_entity '{head_entity}' not found in KG", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if tail_entity not in G:
        return {"success": False, "error": f"tail_entity '{tail_entity}' not found in KG", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if head_entity == tail_entity:
        return {"success": False, "error": "head_entity and tail_entity must be different", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}

    simple_G = _get_metapath_search_graph(kg_path, G, bidirectional)

    min_distance_to_tail = _reverse_shortest_path_lengths(simple_G, tail_entity)

    pattern_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    pattern_examples: dict[tuple[tuple[str, ...], tuple[str, ...]], list[list[str]]] = {}
    total_found = 0
    search_exhausted = True

    def path_generator() -> Any:
        nonlocal search_exhausted

        if head_entity not in min_distance_to_tail or min_distance_to_tail[head_entity] > max_length:
            return

        path = [head_entity]
        visited = {head_entity}

        def dfs(node: str, depth: int) -> Any:
            nonlocal search_exhausted
            remaining_steps = max_length - depth
            min_remaining = min_distance_to_tail.get(node)
            if min_remaining is None or min_remaining > remaining_steps:
                return
            if node == tail_entity:
                yield list(path)
                return
            for neighbor in simple_G.successors(node):
                if neighbor in visited:
                    continue
                next_min_remaining = min_distance_to_tail.get(neighbor)
                if next_min_remaining is None or next_min_remaining > remaining_steps - 1:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                yield from dfs(neighbor, depth + 1)
                path.pop()
                visited.remove(neighbor)
                if not search_exhausted:
                    return

        yield from dfs(head_entity, 0)

    try:
        for path_nodes in path_generator():
            relation_options = []
            for i in range(len(path_nodes) - 1):
                data = simple_G.get_edge_data(path_nodes[i], path_nodes[i + 1]) or {}
                relation_options.append(sorted(data.get("relations", {"unknown"})))
            node_type_pattern = tuple(_node_type(G, node_id) for node_id in path_nodes)
            for rel_seq in product(*relation_options):
                if total_found >= max_paths:
                    search_exhausted = False
                    break
                relation_pattern = tuple(rel_seq)
                key = (node_type_pattern, relation_pattern)
                pattern_counts[key] = pattern_counts.get(key, 0) + 1
                pattern_examples.setdefault(key, [])
                if len(pattern_examples[key]) < 3:
                    pattern_examples[key].append(path_nodes)
                total_found += 1
            if not search_exhausted:
                break
    except nx.NetworkXNoPath:
        pass

    metapath_patterns = [
        {
            "pattern": " ".join(
                [
                    pattern[0][0],
                    *[
                        f"-[{relation}]-> {node_type}"
                        for relation, node_type in zip(pattern[1], pattern[0][1:])
                    ],
                ]
            ),
            "relation_pattern": " → ".join(pattern[1]),
            "node_type_pattern": list(pattern[0]),
            "length": len(pattern[1]),
            "instance_count": count,
            "examples": pattern_examples.get(pattern, []),
        }
        for pattern, count in sorted(pattern_counts.items(), key=lambda item: -item[1])
    ]

    return {
        "success": True,
        "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_length": max_length, "max_paths": max_paths, "bidirectional": bidirectional},
        "result": {"metapath_patterns": metapath_patterns, "total_paths_found": total_found, "search_exhausted": search_exhausted},
    }


def inspect_metapath_length_limits(
    kg_paths: list[str] | None = None,
    sample_pairs: int = 64,
    random_seed: int = 0,
) -> dict:
    """Inspect practical ``extract_metapaths.max_length`` limits for one or more KG files.

    The exact longest simple path of a general cyclic graph is NP-hard to compute,
    so this function reports:
    1. a structural upper bound: ``largest_weak_component_size - 1``
    2. an approximate undirected diameter of the largest component
    3. the maximum sampled shortest-path distance within that component

    The structural upper bound is a safe "cannot exceed" limit for any simple
    path wholly contained in the largest connected region. The diameter metrics
    are usually a better guide for choosing a practical ``max_length``.
    """

    target_paths = kg_paths or _default_kg_paths()
    rng = random.Random(random_seed)
    reports = []

    for kg_path in target_paths:
        if not os.path.isfile(kg_path):
            reports.append(
                {
                    "kg_path": kg_path,
                    "exists": False,
                    "success": False,
                    "error": f"KG file not found: {kg_path}",
                }
            )
            continue

        load_result = load_biomedical_kg(kg_path)
        if not load_result["success"]:
            reports.append(
                {
                    "kg_path": kg_path,
                    "exists": True,
                    "success": False,
                    "error": load_result.get("error", "Failed to load KG"),
                }
            )
            continue

        G = _kg_cache.get(os.path.abspath(kg_path))
        if G is None:
            reports.append(
                {
                    "kg_path": kg_path,
                    "exists": True,
                    "success": False,
                    "error": "Graph not found in cache after loading",
                }
            )
            continue

        simple_undirected = nx.Graph()
        simple_undirected.add_nodes_from(G.nodes())
        simple_undirected.add_edges_from((u, v) for u, v in G.edges())
        weak_components = sorted(nx.connected_components(simple_undirected), key=len, reverse=True)

        if not weak_components:
            reports.append(
                {
                    "kg_path": kg_path,
                    "exists": True,
                    "success": True,
                    "result": {
                        "node_count": 0,
                        "edge_count": 0,
                        "component_count": 0,
                        "largest_component_nodes": 0,
                        "theoretical_simple_path_upper_bound": 0,
                        "approx_diameter_largest_component": 0,
                        "sampled_max_shortest_path": 0,
                        "sampled_pair_count": 0,
                        "exact_longest_path_if_dag": 0,
                        "note": "Empty graph.",
                    },
                }
            )
            continue

        largest_component_nodes = weak_components[0]
        largest_subgraph = simple_undirected.subgraph(largest_component_nodes).copy()
        largest_component_size = largest_subgraph.number_of_nodes()
        theoretical_upper_bound = max(largest_component_size - 1, 0)

        approx_diameter = _two_sweep_diameter_estimate(largest_subgraph, rng)

        component_node_list = list(largest_component_nodes)
        if len(component_node_list) < 2:
            sampled_max_shortest_path = 0
            sampled_pair_count = 0
        else:
            max_possible_pairs = len(component_node_list) * (len(component_node_list) - 1) // 2
            sampled_pair_count = min(sample_pairs, max_possible_pairs)
            sampled_max_shortest_path = 0
            seen_pairs: set[tuple[str, str]] = set()
            while len(seen_pairs) < sampled_pair_count:
                u, v = rng.sample(component_node_list, 2)
                pair = (u, v) if u <= v else (v, u)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                try:
                    distance = nx.shortest_path_length(largest_subgraph, source=pair[0], target=pair[1])
                except nx.NetworkXNoPath:
                    continue
                sampled_max_shortest_path = max(sampled_max_shortest_path, int(distance))

        if G.number_of_nodes() <= 50000 and nx.is_directed_acyclic_graph(G):
            exact_longest_path_if_dag = len(nx.dag_longest_path(G)) - 1
        else:
            exact_longest_path_if_dag = None

        reports.append(
            {
                "kg_path": kg_path,
                "exists": True,
                "success": True,
                "result": {
                    "node_count": G.number_of_nodes(),
                    "edge_count": G.number_of_edges(),
                    "component_count": len(weak_components),
                    "largest_component_nodes": largest_component_size,
                    "theoretical_simple_path_upper_bound": theoretical_upper_bound,
                    "approx_diameter_largest_component": approx_diameter,
                    "sampled_max_shortest_path": sampled_max_shortest_path,
                    "sampled_pair_count": sampled_pair_count,
                    "exact_longest_path_if_dag": exact_longest_path_if_dag,
                    "note": (
                        "Use approx_diameter_largest_component or sampled_max_shortest_path as a practical "
                        "max_length guide. theoretical_simple_path_upper_bound is only a structural upper bound."
                    ),
                },
            }
        )

    return {
        "success": True,
        "query_info": {"kg_paths": target_paths, "sample_pairs": sample_pairs, "random_seed": random_seed},
        "result": {"files": reports},
    }


def traverse_metapath(
    kg_path: str,
    head_entity: str,
    metapath: list[str] | None = None,
    max_results: int = 50,
    bidirectional: bool = True,
    pattern: dict[str, Any] | None = None,
) -> dict:
    """Traverse the KG following an exact relation/type pattern from a head entity.

    ``metapath`` keeps the original interface: it is a non-empty list of
    relation names.  ``pattern`` may be one item from
    ``extract_metapaths(...)["result"]["metapath_patterns"]``.  When supplied,
    its relation pattern is used as the metapath and its node type pattern is
    enforced at every traversal layer.  A path item always remains the
    three-field ``head``/``relation``/``tail`` mapping.
    """

    query_info: dict[str, Any] = {
        "kg_path": kg_path,
        "head_entity": head_entity,
        "metapath": metapath,
        "max_results": max_results,
        "bidirectional": bidirectional,
    }

    if not isinstance(kg_path, str) or not kg_path:
        return {"success": False, "error": "kg_path must be a non-empty string", "query_info": query_info}
    if not isinstance(head_entity, str) or not head_entity:
        return {"success": False, "error": "head_entity must be a non-empty string", "query_info": query_info}
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        return {"success": False, "error": f"max_results must be a positive integer, got {max_results!r}", "query_info": query_info}
    if not isinstance(bidirectional, bool):
        return {"success": False, "error": "bidirectional must be a boolean", "query_info": query_info}

    pattern_node_types: list[str] | None = None
    pattern_relations: list[str] | None = None
    if pattern is not None:
        if not isinstance(pattern, dict):
            return {"success": False, "error": "pattern must be a dictionary returned by extract_metapaths", "query_info": query_info}
        raw_relations = pattern.get("relation_pattern")
        if isinstance(raw_relations, str):
            pattern_relations = [item.strip() for item in raw_relations.split("→")]
        elif isinstance(raw_relations, list):
            pattern_relations = raw_relations
        else:
            return {"success": False, "error": "pattern.relation_pattern must be a string or list of strings", "query_info": query_info}
        if not pattern_relations or not all(isinstance(item, str) and item for item in pattern_relations):
            return {"success": False, "error": "pattern.relation_pattern must contain non-empty strings", "query_info": query_info}
        raw_node_types = pattern.get("node_type_pattern")
        if not isinstance(raw_node_types, list) or not all(isinstance(item, str) and item for item in raw_node_types):
            return {"success": False, "error": "pattern.node_type_pattern must be a list of non-empty strings", "query_info": query_info}
        pattern_node_types = raw_node_types
        if len(pattern_node_types) != len(pattern_relations) + 1:
            return {"success": False, "error": "pattern.node_type_pattern must contain exactly one more item than relation_pattern", "query_info": query_info}
        if metapath is not None and metapath != pattern_relations:
            return {"success": False, "error": "metapath does not match pattern.relation_pattern", "query_info": query_info}
        metapath = pattern_relations

    if not isinstance(metapath, list) or not metapath or not all(isinstance(item, str) and item for item in metapath):
        return {"success": False, "error": "metapath must be a non-empty list of non-empty strings", "query_info": query_info}
    if any(item == "INV_" for item in metapath):
        return {"success": False, "error": "reverse relation steps must include a relation after 'INV_'", "query_info": query_info}
    if not bidirectional and any(item.startswith("INV_") for item in metapath):
        return {"success": False, "error": "bidirectional must be true when metapath contains INV_ relations", "query_info": query_info}
    query_info["metapath"] = metapath
    if pattern_node_types is not None:
        query_info["node_type_pattern"] = pattern_node_types

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": query_info}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": query_info}
    if head_entity not in G:
        return {"success": False, "error": f"head_entity '{head_entity}' not found in KG", "query_info": query_info}
    if pattern_node_types is not None and _node_type(G, head_entity) != pattern_node_types[0]:
        return {
            "success": False,
            "error": f"head_entity '{head_entity}' has type '{_node_type(G, head_entity)}', expected '{pattern_node_types[0]}'",
            "query_info": query_info,
        }

    fwd: dict[str, set[tuple[str, str]]] = defaultdict(set)
    rev: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        rel = str(data.get("relation", "unknown"))
        fwd[u].add((v, rel))
        rev[v].add((u, rel))

    frontier: list[list[dict[str, str]]] = [[]]
    frontier_counts = [1]
    failed_relation_step: int | None = None
    for step_index, rel_step in enumerate(metapath):
        use_reverse = bidirectional and rel_step.startswith("INV_")
        rel_key = rel_step[4:] if use_reverse else rel_step
        next_frontier: list[list[dict[str, str]]] = []
        for path in frontier:
            current = path[-1]["tail"] if path else head_entity
            visited = {head_entity}
            visited.update(triple["tail"] for triple in path)
            candidates = rev.get(current, set()) if use_reverse else fwd.get(current, set())
            expected_type = pattern_node_types[step_index + 1] if pattern_node_types is not None else None
            for nb, actual_rel in sorted(candidates):
                if actual_rel != rel_key or nb in visited:
                    continue
                if expected_type is not None and _node_type(G, nb) != expected_type:
                    continue
                next_frontier.append(path + [{"head": current, "relation": rel_step, "tail": nb}])
        frontier = next_frontier
        frontier_counts.append(len(frontier))
        if not frontier:
            failed_relation_step = step_index
            break

    tail_examples: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    tail_counts: dict[str, int] = defaultdict(int)
    for path in frontier:
        if not path:
            continue
        tail = path[-1]["tail"]
        tail_counts[tail] += 1
        if len(tail_examples[tail]) < 3:
            tail_examples[tail].append(path)

    ordered_tails = sorted(tail_counts, key=lambda tail: (-tail_counts[tail], tail))
    tail_entities = [{"entity": tail, "path_count": tail_counts[tail], "paths": tail_examples[tail]} for tail in ordered_tails[:max_results]]

    query_info.update(
        {
            "frontier_counts": frontier_counts,
            "matched_depth": len(metapath) if failed_relation_step is None else failed_relation_step,
            "failed_relation_step": failed_relation_step,
        }
    )

    return {
        "success": True,
        "query_info": query_info,
        "result": {"tail_entities": tail_entities, "total_reached": len(ordered_tails), "path_count": sum(tail_counts.values())},
    }


def extract_enclosing_subgraph(
    kg_path: str,
    head_entity: str,
    tail_entity: str,
    max_hops: int = 3,
    max_nodes_per_hop: int = 200,
    remove_direct_link: bool = True,
    bidirectional: bool = True,
) -> dict:
    """Extract the enclosing subgraph around a head-tail entity pair."""

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if max_hops < 1:
        return {"success": False, "error": f"max_hops must be >= 1, got {max_hops}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_hops": max_hops}}
    if max_nodes_per_hop < 1:
        return {"success": False, "error": f"max_nodes_per_hop must be >= 1, got {max_nodes_per_hop}", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_nodes_per_hop": max_nodes_per_hop}}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": {"kg_path": kg_path}}
    if head_entity not in G:
        return {"success": False, "error": f"head_entity '{head_entity}' not found in KG", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if tail_entity not in G:
        return {"success": False, "error": f"tail_entity '{tail_entity}' not found in KG", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}
    if head_entity == tail_entity:
        return {"success": False, "error": "head_entity and tail_entity must be different", "query_info": {"head_entity": head_entity, "tail_entity": tail_entity}}

    def neighbors(node: str) -> list[str]:
        if bidirectional:
            return list(set(G.successors(node)) | set(G.predecessors(node)))
        return list(G.successors(node))

    dist_h: dict[str, int] = {head_entity: 0}
    dist_t: dict[str, int] = {tail_entity: 0}
    frontier_h: set[str] = {head_entity}
    frontier_t: set[str] = {tail_entity}

    for hop in range(1, max_hops + 1):
        new_h = {nb for node in frontier_h for nb in neighbors(node) if nb not in dist_h}
        if len(new_h) > max_nodes_per_hop:
            new_h = set(random.sample(list(new_h), max_nodes_per_hop))
        for nb in new_h:
            dist_h[nb] = hop
        frontier_h = new_h

        new_t = {nb for node in frontier_t for nb in neighbors(node) if nb not in dist_t}
        if len(new_t) > max_nodes_per_hop:
            new_t = set(random.sample(list(new_t), max_nodes_per_hop))
        for nb in new_t:
            dist_t[nb] = hop
        frontier_t = new_t

        if not frontier_h and not frontier_t:
            break

    all_visited = set(dist_h) | set(dist_t)
    node_labels = {
        node: {
            "dist_to_head": dist_h.get(node) if dist_h.get(node, max_hops + 1) <= max_hops else None,
            "dist_to_tail": dist_t.get(node) if dist_t.get(node, max_hops + 1) <= max_hops else None,
        }
        for node in all_visited
    }

    edges = []
    for u in all_visited:
        for v in all_visited:
            for ed in _edge_items(G.get_edge_data(u, v)):
                if remove_direct_link and ((u == head_entity and v == tail_entity) or (u == tail_entity and v == head_entity)):
                    continue
                edges.append({"head": u, "relation": ed.get("relation", "unknown"), "tail": v})

    head_reachable = {n for n in all_visited if n in dist_h}
    tail_reachable = {n for n in all_visited if n in dist_t}
    return {
        "success": True,
        "query_info": {"head_entity": head_entity, "tail_entity": tail_entity, "max_hops": max_hops, "max_nodes_per_hop": max_nodes_per_hop, "remove_direct_link": remove_direct_link, "bidirectional": bidirectional},
        "result": {
            "subgraph": {"nodes": sorted(all_visited), "edges": edges},
            "node_labels": node_labels,
            "node_count": len(all_visited),
            "edge_count": len(edges),
            "head_reachable_count": len(head_reachable),
            "tail_reachable_count": len(tail_reachable),
            "intersection_count": len(head_reachable & tail_reachable),
        },
    }


def compute_pagerank(
    kg_path: str,
    damping: float = 0.85,
    top_k: int = 100,
    relation_filter: list[str] | None = None,
    bidirectional: bool = True,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> dict:
    """Compute global PageRank centrality on the biomedical knowledge graph."""

    if not os.path.isfile(kg_path):
        return {"success": False, "error": f"KG file not found: {kg_path}", "query_info": {"kg_path": kg_path}}
    if not (0 < damping < 1):
        return {"success": False, "error": f"damping must be in (0, 1), got {damping}", "query_info": {"kg_path": kg_path, "damping": damping}}
    if top_k < 1:
        return {"success": False, "error": f"top_k must be >= 1, got {top_k}", "query_info": {"kg_path": kg_path, "top_k": top_k}}

    load_result = load_biomedical_kg(kg_path)
    if not load_result["success"]:
        return load_result
    G = _kg_cache.get(os.path.abspath(kg_path))
    if G is None:
        return {"success": False, "error": "Graph not in cache", "query_info": {"kg_path": kg_path}}

    pr_G = nx.DiGraph()
    pr_G.add_nodes_from(G.nodes())
    relation_filter_set = set(relation_filter) if relation_filter is not None else None
    for u, v, data in G.edges(data=True):
        rel = str(data.get("relation", "unknown"))
        if relation_filter_set is not None and rel not in relation_filter_set:
            continue
        pr_G.add_edge(u, v)
        if bidirectional:
            pr_G.add_edge(v, u)

    if pr_G.number_of_edges() == 0:
        return {"success": False, "error": "Graph has no edges after filtering", "query_info": {"kg_path": kg_path, "relation_filter": relation_filter}}

    try:
        scores = nx.pagerank(pr_G, alpha=damping, max_iter=max_iter, tol=tolerance)
    except nx.PowerIterationFailedConvergence as exc:
        return {"success": False, "error": f"PageRank did not converge within {max_iter} iterations: {exc}", "query_info": {"kg_path": kg_path, "damping": damping, "max_iter": max_iter}}

    sorted_nodes = sorted(scores.items(), key=lambda item: -item[1])
    ranking = [{"rank": idx, "node": node, "score": round(score, 8)} for idx, (node, score) in enumerate(sorted_nodes[:top_k], start=1)]
    score_array = np.array(list(scores.values()), dtype=np.float64)
    stats = {
        "min": round(float(score_array.min()), 8),
        "max": round(float(score_array.max()), 8),
        "mean": round(float(score_array.mean()), 8),
        "median": round(float(np.median(score_array)), 8),
    }

    return {
        "success": True,
        "query_info": {"kg_path": kg_path, "damping": damping, "top_k": top_k, "relation_filter": relation_filter, "bidirectional": bidirectional, "graph_nodes": pr_G.number_of_nodes(), "graph_edges": pr_G.number_of_edges()},
        "result": {"ranking": ranking, "statistics": stats},
    }
