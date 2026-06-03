# Copyright 2024 The OPRO Authors (TE extension)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OPRO optimization loop for the Traffic Engineering (TE) problem.

This module adapts the OPRO meta-prompt optimizer (originally designed for
TSP) to Traffic Engineering routing.  A *solution* is represented as a list
of path indices — one per OD pair (in canonical source-first order) — that
specifies which pre-computed k-shortest path to use for each commodity.

Evaluation uses a lightweight edge-utilisation calculation (no LP/Gurobi),
making the loop fast enough to run inside a MetaRL episode.

Supported objectives:
  * ``min_max_link_util``  — minimise maximum link utilisation (lower = better)
  * ``total_flow``         — maximise total routed flow (greedy, bounded by
                             residual edge capacities)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data-structure helpers
# ---------------------------------------------------------------------------

def build_te_structures(graph, node_names, paths_dict):
    """Build compact commodity / path lists from MetaRL topology kwargs.

    Returns
    -------
    commodities : list of (commod_id, s_idx, t_idx, [path_edge_lists])
        ``path_edge_lists[i]`` is the list of ``(u, v)`` tuples for local path
        *i* of this OD pair.
    edge_cap : dict {(u, v): float}
        Edge capacities from the graph.
    od_pair_labels : list of str
        Human-readable labels "OD_{id}: ({s}→{t})" for the meta-prompt.
    od_pair_paths_text : list of str
        Human-readable path descriptions for the meta-prompt.
    """
    commodities = []
    od_pair_labels = []
    od_pair_paths_text = []
    commod_id = 0

    for s_idx, s in enumerate(node_names):
        for t_idx, t in enumerate(node_names):
            if s == t:
                continue
            paths = paths_dict.get((s, t), [])
            if not paths:
                continue

            path_edge_lists = []
            path_strs = []
            for k, path in enumerate(paths):
                edges = list(_path_to_edges(path))
                path_edge_lists.append(edges)
                path_strs.append(f"P{k}=[{','.join(str(n) for n in path)}]")

            commodities.append((commod_id, s_idx, t_idx, path_edge_lists))
            od_pair_labels.append(f"OD_{commod_id}: ({s}→{t})")
            od_pair_paths_text.append(
                f"  OD_{commod_id} ({s}→{t}): " + ", ".join(path_strs)
            )
            commod_id += 1

    edge_cap = {
        (u, v): float(data.get("capacity", 0.0))
        for u, v, data in graph.edges(data=True)
    }
    return commodities, edge_cap, od_pair_labels, od_pair_paths_text


def _path_to_edges(path):
    """Yield (u, v) pairs for consecutive nodes in *path*."""
    it = iter(path)
    prev = next(it)
    for node in it:
        yield (prev, node)
        prev = node


# ---------------------------------------------------------------------------
# Routing evaluation
# ---------------------------------------------------------------------------

def evaluate_routing(routing_indices, tm, commodities, edge_cap, objective):
    """Evaluate a single-path routing assignment.

    Parameters
    ----------
    routing_indices : list[int]
        One path index per commodity (0-indexed, relative to that OD pair's
        local path list).  Values exceeding the path count wrap around (modulo).
    tm : np.ndarray, shape (N, N)
        Traffic demand matrix.
    commodities : list
        As returned by :func:`build_te_structures`.
    edge_cap : dict {(u, v): float}
        Edge capacities.
    objective : str
        ``'min_max_link_util'`` or ``'total_flow'``.

    Returns
    -------
    float
        MLU (lower is better) or total routed flow (higher is better).
    """
    if objective == "min_max_link_util":
        return _eval_mlu(routing_indices, tm, commodities, edge_cap)
    elif objective == "total_flow":
        return _eval_total_flow(routing_indices, tm, commodities, edge_cap)
    else:
        raise ValueError(f"Unknown objective: {objective!r}")


def _eval_mlu(routing_indices, tm, commodities, edge_cap):
    """Compute max link utilisation for a single-path routing."""
    edge_flow: dict = defaultdict(float)
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        for u, v in path_edge_lists[local_idx]:
            edge_flow[(u, v)] += demand

    mlu = 0.0
    for (u, v), cap in edge_cap.items():
        if cap > 0.0:
            mlu = max(mlu, edge_flow.get((u, v), 0.0) / cap)
    return mlu


def _eval_total_flow(routing_indices, tm, commodities, edge_cap):
    """Compute greedy total flow for a single-path routing.

    OD pairs are processed largest-demand-first.  Each demand is routed on
    its chosen path up to the remaining bottleneck capacity on that path.
    """
    remaining_cap = dict(edge_cap)
    total_flow = 0.0

    sorted_commods = sorted(
        commodities, key=lambda c: -float(tm[c[1], c[2]])
    )
    for commod_id, s_idx, t_idx, path_edge_lists in sorted_commods:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        edges = path_edge_lists[local_idx]
        if not edges:
            continue
        bottleneck = min(remaining_cap.get((u, v), 0.0) for u, v in edges)
        allocated = min(demand, max(0.0, bottleneck))
        if allocated > 0.0:
            total_flow += allocated
            for u, v in edges:
                remaining_cap[(u, v)] = remaining_cap.get((u, v), 0.0) - allocated

    return total_flow


# ---------------------------------------------------------------------------
# Initial solution generators
# ---------------------------------------------------------------------------

def _make_random_routing(num_od_pairs, num_paths_per_od):
    """Return a random per-OD path index assignment."""
    return [
        int(np.random.randint(0, max(1, n)))
        for n in num_paths_per_od
    ]


def _make_shortest_path_routing(num_od_pairs):
    """Route all OD pairs on path 0 (assumed to be the shortest path)."""
    return [0] * num_od_pairs


# ---------------------------------------------------------------------------
# Meta-prompt generation and output parsing
# ---------------------------------------------------------------------------

def _tm_summary(tm, node_names, commodities, max_show=30):
    """Build a compact string for the traffic matrix (non-zero pairs only)."""
    lines = []
    for commod_id, s_idx, t_idx, _ in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand > 0.0:
            lines.append(
                f"  OD_{commod_id} ({node_names[s_idx]}→{node_names[t_idx]}): {demand:.1f}"
            )
    if len(lines) > max_show:
        shown = lines[:max_show]
        shown.append(f"  ... ({len(lines) - max_show} more OD pairs omitted)")
        return "\n".join(shown)
    return "\n".join(lines) if lines else "  (all demands are zero)"


def _edge_summary(graph, max_show=20):
    """Build a compact edge-capacity summary string."""
    edges_sorted = sorted(
        graph.edges(data=True),
        key=lambda e: -float(e[2].get("capacity", 0.0)),
    )
    lines = []
    for u, v, data in edges_sorted[:max_show]:
        cap = data.get("capacity", 0.0)
        lines.append(f"  ({u},{v}): cap={cap:.0f}")
    if len(list(graph.edges())) > max_show:
        lines.append(f"  ... ({graph.number_of_edges() - max_show} more edges)")
    return "\n".join(lines)


def gen_meta_prompt_te(
    old_value_pairs_set,
    graph,
    node_names,
    tm,
    commodities,
    od_pair_labels,
    od_pair_paths_text,
    max_num_pairs=5,
    objective="min_max_link_util",
):
    """Generate the OPRO meta-prompt for traffic engineering.

    Parameters
    ----------
    old_value_pairs_set : set of (routing_str, score)
        Previously evaluated (routing, objective) pairs.
    graph : networkx.DiGraph
    node_names : list
    tm : np.ndarray
    commodities : list
    od_pair_labels : list[str]
    od_pair_paths_text : list[str]
    max_num_pairs : int
        Maximum number of exemplar pairs to include.
    objective : str

    Returns
    -------
    str
        The full meta-prompt.
    """
    # Sort pairs — for minimisation (MLU) ascending score is better;
    # show worst first (descending) so the best is last.
    pairs = list(old_value_pairs_set)
    if objective == "min_max_link_util":
        pairs.sort(key=lambda x: -x[1])   # descending: worst → best at end
        better_direction = "lower"
        obj_label = "MLU (Maximum Link Utilisation)"
    else:
        pairs.sort(key=lambda x: x[1])    # ascending: worst → best at end
        better_direction = "higher"
        obj_label = "Total Flow"

    pairs = pairs[-max_num_pairs:]  # keep only the most recent/best slice

    # ── Topology section ───────────────────────────────────────────────────
    n_nodes = len(node_names)
    n_edges = graph.number_of_edges()
    topo_str = (
        f"Nodes: {n_nodes} (indexed {node_names[0]}–{node_names[-1]})\n"
        f"Directed edges (total {n_edges}), top-{min(20, n_edges)} by capacity:\n"
        + _edge_summary(graph)
    )

    # ── Traffic Matrix section ─────────────────────────────────────────────
    tm_str = _tm_summary(tm, node_names, commodities)

    # ── Paths section (trimmed for large networks) ─────────────────────────
    max_paths_show = 40
    if len(od_pair_paths_text) <= max_paths_show:
        paths_str = "\n".join(od_pair_paths_text)
    else:
        paths_str = (
            "\n".join(od_pair_paths_text[:max_paths_show])
            + f"\n  ... ({len(od_pair_paths_text) - max_paths_show} more OD pairs)"
        )

    # ── Previous solutions section ─────────────────────────────────────────
    if pairs:
        prev_str = ""
        for routing_str, score in pairs:
            score_fmt = f"{score:.4f}"
            prev_str += f"\n<routing> {routing_str} </routing>\n{obj_label}: {score_fmt}\n"
        prev_str = prev_str.strip()
    else:
        prev_str = "(none yet)"

    # ── Task description ───────────────────────────────────────────────────
    num_od = len(commodities)
    max_paths_per_od = max(
        (len(path_edge_lists) for _, _, _, path_edge_lists in commodities),
        default=1,
    )

    prompt = f"""You are solving a Traffic Engineering routing optimisation problem.

=== Network ===
{topo_str}

=== Traffic Demands ===
{tm_str}

=== Available Paths ===
Each OD pair has up to {max_paths_per_od} paths (0-indexed). Path nodes are listed in order:
{paths_str}

=== Objective ===
{obj_label}.  A {better_direction} value is better.

=== Previous Routings ===
Previous routings and their {obj_label} scores are shown below,
sorted from worst to best ({better_direction} is better):

{prev_str}

=== Task ===
Provide a new routing that is DIFFERENT from all routings above and achieves
a {better_direction} {obj_label} than any routing listed.

The routing is a comma-separated list of {num_od} integers, one per OD pair
in the same order as "=== Available Paths ===" above. Each integer is the
0-based path index for that OD pair (use 0 if only one path exists).

Wrap the routing in <routing> ... </routing> tags.
Example format: <routing> 0,1,0,2,1,0,1 </routing>
""".strip()

    return prompt


def extract_routing(input_string, num_od_pairs):
    """Parse the LLM output to extract a routing (list of path indices).

    Parameters
    ----------
    input_string : str
        Raw LLM output.
    num_od_pairs : int
        Expected number of path-index values.

    Returns
    -------
    list[int] or None
        Parsed routing or ``None`` if parsing failed.
    """
    if not input_string:
        return None

    # Try to find content between <routing> ... </routing>
    match = re.search(r"<routing>(.*?)</routing>", input_string, re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1)
    else:
        # Fallback: look for a comma-separated sequence of integers
        raw = input_string

    tokens = []
    for tok in re.split(r"[,\s]+", raw.strip()):
        tok = tok.strip().strip(".,;[](){}")
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            tokens.append(int(tok))

    if len(tokens) != num_od_pairs:
        return None

    return tokens


# ---------------------------------------------------------------------------
# Main OPRO optimization loop
# ---------------------------------------------------------------------------

def build_edge_allocation(routing_indices, tm, commodities, node_names):
    """Build an edge-allocation dict from a routing assignment.

    Parameters
    ----------
    routing_indices : list[int]
    tm : np.ndarray
    commodities : list
    node_names : list

    Returns
    -------
    dict { (s_name, t_name): { (u, v): flow } }
        Same structure as returned by DPSolver / PathOptimalSolver.
    """
    allocation = defaultdict(lambda: defaultdict(float))
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        local_idx = routing_indices[commod_id] % len(path_edge_lists)
        s_name = node_names[s_idx]
        t_name = node_names[t_idx]
        for u, v in path_edge_lists[local_idx]:
            allocation[(s_name, t_name)][(u, v)] += demand
    return {k: dict(v) for k, v in allocation.items()}


def run_opro_te(
    tm,
    graph,
    node_names,
    paths_dict,
    call_optimizer_server_func: Callable,
    objective: str = "min_max_link_util",
    num_steps: int = 50,
    num_decode_per_step: int = 4,
    max_num_pairs: int = 5,
    num_starting_points: int = 3,
    verbose: bool = True,
):
    """Run the OPRO optimization loop for Traffic Engineering.

    Parameters
    ----------
    tm : np.ndarray, shape (N, N)
        Traffic demand matrix for this episode.
    graph : networkx.DiGraph
    node_names : list
    paths_dict : dict
        MetaRL-format path dictionary ``{(s, t): [[node, ...], ...]}``
    call_optimizer_server_func : callable
        A function ``f(prompt: str) -> list[str]`` that makes a single LLM
        call and returns a list of generated strings.
    num_decode_per_step : int
        How many times to call *call_optimizer_server_func* per OPRO step;
        all returned strings are pooled for routing extraction.
    objective : str
    num_steps : int
    max_num_pairs : int
    num_starting_points : int
    verbose : bool

    Returns
    -------
    best_score : float
        Objective value of the best routing found.
    best_edge_allocation : dict
        ``{(s_name, t_name): {(u, v): flow}}`` for the best routing found.
    """
    # Build internal data structures
    commodities, edge_cap, od_pair_labels, od_pair_paths_text = build_te_structures(
        graph, node_names, paths_dict
    )
    num_od = len(commodities)
    if num_od == 0:
        return 0.0, {}

    num_paths_per_od = [
        len(path_edge_lists) for _, _, _, path_edge_lists in commodities
    ]

    # ---------- initialise with a few starting points ----------------------
    old_value_pairs_set: set = set()  # {(routing_str, score)}
    best_score: Optional[float] = None
    best_routing: Optional[List[int]] = None

    # Always include the shortest-path routing as a baseline
    init_routings = [_make_shortest_path_routing(num_od)]
    for _ in range(num_starting_points - 1):
        init_routings.append(_make_random_routing(num_od, num_paths_per_od))

    for routing in init_routings:
        score = evaluate_routing(routing, tm, commodities, edge_cap, objective)
        routing_str = ",".join(str(x) for x in routing)
        old_value_pairs_set.add((routing_str, score))
        if best_score is None or _is_better(score, best_score, objective):
            best_score = score
            best_routing = routing[:]

    if verbose:
        print(f"[OPRO-TE] Initial best {objective}: {best_score:.4f}")

    # ---------- OPRO loop --------------------------------------------------
    for i_step in range(num_steps):
        meta_prompt = gen_meta_prompt_te(
            old_value_pairs_set,
            graph,
            node_names,
            tm,
            commodities,
            od_pair_labels,
            od_pair_paths_text,
            max_num_pairs=max_num_pairs,
            objective=objective,
        )

        if verbose:
            print(f"\n[OPRO-TE] Step {i_step + 1}/{num_steps}")

        raw_outputs = []
        for _ in range(num_decode_per_step):
            result = call_optimizer_server_func(meta_prompt)
            if isinstance(result, str):
                raw_outputs.append(result)
            else:
                raw_outputs.extend(result)

        for raw_out in raw_outputs:
            routing = extract_routing(raw_out, num_od)
            if routing is None:
                if verbose:
                    print("  [parse] failed to parse LLM output, skipping.")
                continue

            score = evaluate_routing(routing, tm, commodities, edge_cap, objective)
            routing_str = ",".join(str(x) for x in routing)
            old_value_pairs_set.add((routing_str, score))

            if _is_better(score, best_score, objective):
                best_score = score
                best_routing = routing[:]
                if verbose:
                    print(f"  [update] new best {objective}: {best_score:.4f}")

    if verbose:
        print(f"[OPRO-TE] Final best {objective}: {best_score:.4f}")

    best_edge_allocation = build_edge_allocation(
        best_routing, tm, commodities, node_names
    ) if best_routing else {}

    return best_score, best_edge_allocation


def _is_better(score, current_best, objective):
    """Return True if *score* is strictly better than *current_best*."""
    if objective == "min_max_link_util":
        return score < current_best
    else:  # total_flow
        return score > current_best
