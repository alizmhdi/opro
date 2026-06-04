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
import sys
import os
from collections import defaultdict
from typing import List, Optional

import numpy as np

# Allow importing prompt_utils from the OPRO package
_OPRO_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _OPRO_PKG_DIR not in sys.path:
    sys.path.insert(0, _OPRO_PKG_DIR)

from opro.prompt_utils import call_vllm_server_single_prompt  # noqa: E402


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

def evaluate_routing(routing_splits, tm, commodities, edge_cap, objective):
    """Evaluate a multi-path split-ratio routing.

    Parameters
    ----------
    routing_splits : list[list[float]]
        For each commodity, a list of fractions (one per available path)
        that sum to 1.0.  Each fraction specifies what portion of the demand
        is routed through the corresponding path.
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
        return _eval_mlu(routing_splits, tm, commodities, edge_cap)
    elif objective == "total_flow":
        return _eval_total_flow(routing_splits, tm, commodities, edge_cap)
    else:
        raise ValueError(f"Unknown objective: {objective!r}")


def _eval_mlu(routing_splits, tm, commodities, edge_cap):
    """Compute max link utilisation for a split-ratio routing."""
    edge_flow: dict = defaultdict(float)
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        demand = float(tm[s_idx, t_idx])
        if demand <= 0.0:
            continue
        fracs = routing_splits[commod_id]
        for path_idx, frac in enumerate(fracs):
            if frac <= 0.0 or path_idx >= len(path_edge_lists):
                continue
            for u, v in path_edge_lists[path_idx]:
                edge_flow[(u, v)] += frac * demand

    mlu = 0.0
    for (u, v), cap in edge_cap.items():
        if cap > 0.0:
            mlu = max(mlu, edge_flow.get((u, v), 0.0) / cap)
    return mlu


def _eval_total_flow(routing_splits, tm, commodities, edge_cap):
    """Compute greedy total flow for a split-ratio routing.

    OD pairs are processed largest-demand-first.  For each OD pair, demand
    is split across paths proportionally and each slice is routed up to the
    remaining bottleneck capacity on that path.
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
        fracs = routing_splits[commod_id]
        for path_idx, frac in enumerate(fracs):
            if frac <= 0.0 or path_idx >= len(path_edge_lists):
                continue
            path_demand = frac * demand
            edges = path_edge_lists[path_idx]
            if not edges:
                continue
            bottleneck = min(remaining_cap.get((u, v), 0.0) for u, v in edges)
            allocated = min(path_demand, max(0.0, bottleneck))
            if allocated > 0.0:
                total_flow += allocated
                for u, v in edges:
                    remaining_cap[(u, v)] = remaining_cap.get((u, v), 0.0) - allocated

    return total_flow


# ---------------------------------------------------------------------------
# Initial solution generators
# ---------------------------------------------------------------------------

def _make_random_routing(num_od_pairs, num_paths_per_od):
    """Return a random split-ratio routing using Dirichlet sampling."""
    return [
        list(np.random.dirichlet(np.ones(max(1, n))))
        for n in num_paths_per_od
    ]


def _make_shortest_path_routing(num_paths_per_od):
    """Route all OD pairs entirely on path 0 (split ratio 1.0 on path 0)."""
    return [[1.0] + [0.0] * (n - 1) for n in num_paths_per_od]


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
    num_paths,
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
        pairs.sort(key=lambda x: -x[1])
        better_direction = "lower"
        obj_label = "MLU"
    else:
        pairs.sort(key=lambda x: x[1])
        better_direction = "higher"
        obj_label = "TotalFlow"

    pairs = pairs[-max_num_pairs:]

    # ── Compact paths summary: OD_id(s→t) P0:Nh,P1:Nh,... ────────────────
    paths_lines = []
    for commod_id, s_idx, t_idx, path_edge_lists in commodities:
        hops = ",".join(f"P{i}:{len(e)}h" for i, e in enumerate(path_edge_lists))
        paths_lines.append(f"OD_{commod_id}({node_names[s_idx]}->{node_names[t_idx]}):{hops}")
    paths_str = " | ".join(paths_lines)

    # ── Traffic demands (non-zero only) ────────────────────────────────────
    tm_str = _tm_summary(tm, node_names, commodities)

    # ── Previous solutions ─────────────────────────────────────────────────
    if pairs:
        prev_str = ""
        for routing_str, score in pairs:
            prev_str += f"\n<routing>{routing_str}</routing> {obj_label}:{score:.4f}\n"
        prev_str = prev_str.strip()
    else:
        prev_str = "(none yet)"

    # ── Task description ───────────────────────────────────────────────────
    num_od = len(commodities)

    prompt = f"""Traffic Engineering routing optimisation.
Objective: {obj_label}. {better_direction} is better.
There are EXACTLY {num_od} OD pairs. Each has EXACTLY {num_paths} paths (P0..P{num_paths - 1}).
Paths (OD_id(src->dst):P0:hops,P1:hops,...):
{paths_str}
Demands (OD_id src->dst: demand):
{tm_str}
Previous routings (worst to best):
{prev_str}
Task: output a routing with {better_direction} {obj_label} than all above.
Format: {num_od} groups separated by "|"; each group = {num_paths} comma-separated fractions (normalised to sum=1).
Do NOT output any explanation or reasoning. Output ONLY the splits immediately after <routing>.
<routing>""".strip()

    return prompt


def extract_routing(input_string, num_od_pairs):
    """Parse the LLM output to extract a split-ratio routing.

    Parameters
    ----------
    input_string : str
        Raw LLM output.
    num_od_pairs : int
        Expected number of OD pairs.

    Returns
    -------
    list[list[float]] or None
        Normalised split ratios (one list of fractions per OD pair, summing
        to 1.0), or ``None`` if parsing failed.
    """
    if not input_string:
        return None

    # Try to find content between <routing> ... </routing>
    match = re.search(r"<routing>(.*?)</routing>", input_string, re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
    else:
        # Closing tag missing (output truncated or model omitted it).
        # Extract everything after the opening <routing> tag.
        open_match = re.search(r"<routing>(.*)", input_string, re.DOTALL | re.IGNORECASE)
        raw = open_match.group(1).strip() if open_match else input_string.strip()

    od_parts = [p.strip() for p in re.split(r"\|", raw)]
    # Accept outputs with extra groups (LLM repetition) by truncating;
    # reject only if there are fewer groups than needed.
    if len(od_parts) < num_od_pairs:
        return None
    od_parts = od_parts[:num_od_pairs]

    routing_splits = []
    for part in od_parts:
        fracs = []
        for tok in re.split(r"[,\s]+", part.strip()):
            tok = tok.strip().strip(".,;[](){}")
            if not tok:
                continue
            try:
                fracs.append(float(tok))
            except ValueError:
                pass
        if not fracs:
            return None
        total = sum(fracs)
        if total <= 0.0:
            return None
        routing_splits.append([f / total for f in fracs])

    return routing_splits


# ---------------------------------------------------------------------------
# Main OPRO optimization loop
# ---------------------------------------------------------------------------

def build_edge_allocation(routing_splits, tm, commodities, node_names):
    """Build an edge-allocation dict from a split-ratio routing.

    Parameters
    ----------
    routing_splits : list[list[float]]
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
        fracs = routing_splits[commod_id]
        s_name = node_names[s_idx]
        t_name = node_names[t_idx]
        for path_idx, frac in enumerate(fracs):
            if frac <= 0.0 or path_idx >= len(path_edge_lists):
                continue
            for u, v in path_edge_lists[path_idx]:
                allocation[(s_name, t_name)][(u, v)] += frac * demand
    return {k: dict(v) for k, v in allocation.items()}


def run_opro_te(
    tm,
    graph,
    node_names,
    paths_dict,
    objective: str = "min_max_link_util",
    num_steps: int = 50,
    num_decode_per_step: int = 4,
    max_num_pairs: int = 5,
    num_starting_points: int = 3,
    verbose: bool = True,
    vllm_base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_decode_tokens: int = 1024,
    temperature: float = 1.0,
    api_key: str = "EMPTY",
    num_paths: int = 4,
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
    vllm_base_url : str
        Base URL of the vLLM OpenAI-compatible server.
    model_name : str
        Model name as served by vLLM.
    max_decode_tokens : int
        Maximum tokens to generate per LLM call.
    temperature : float
        Sampling temperature.
    api_key : str
        API key for the vLLM server.
    num_decode_per_step : int
        How many LLM samples to draw per OPRO step;
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


    # Build the vLLM client once and reuse it across all steps/samples.
    from openai import OpenAI as _OpenAIClient  # noqa: PLC0415
    _llm_client = _OpenAIClient(base_url=vllm_base_url, api_key=api_key)

    num_paths_per_od = [
        len(path_edge_lists) for _, _, _, path_edge_lists in commodities
    ]

    # Cap max_decode_tokens to what's actually needed for split-ratio output:
    # Compute the minimum tokens needed to output a complete routing.
    # Each fraction like "0.3," is ~4 tokens; add overhead for tags and separators.
    _min_needed_tokens = num_od * num_paths * 6 + 128
    _max_allowed_tokens = _min_needed_tokens
    if max_decode_tokens < _min_needed_tokens:
        if verbose:
            print(f"[OPRO-TE] Raising max_decode_tokens from {max_decode_tokens} "
                  f"to {_min_needed_tokens} (minimum needed for {num_od} OD pairs x {num_paths} paths)")
        max_decode_tokens = _min_needed_tokens
    elif max_decode_tokens > _max_allowed_tokens:
        if verbose:
            print(f"[OPRO-TE] Capping max_decode_tokens from {max_decode_tokens} "
                  f"to {_max_allowed_tokens} (based on {num_od} OD pairs x {num_paths} paths)")
        max_decode_tokens = _max_allowed_tokens

    # ---------- initialise with a few starting points ----------------------
    old_value_pairs_set: set = set()  # {(routing_str, score)}
    best_score: Optional[float] = None
    best_routing: Optional[List[List[float]]] = None

    # Always include the shortest-path routing as a baseline
    init_routings = [_make_shortest_path_routing(num_paths_per_od)]
    for _ in range(num_starting_points - 1):
        init_routings.append(_make_random_routing(num_od, num_paths_per_od))

    for routing in init_routings:
        score = evaluate_routing(routing, tm, commodities, edge_cap, objective)
        routing_str = "|".join(",".join(f"{f:.1f}" for f in fracs) for fracs in routing)
        old_value_pairs_set.add((routing_str, score))
        if best_score is None or _is_better(score, best_score, objective):
            best_score = score
            best_routing = [list(fracs) for fracs in routing]

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
            num_paths=num_paths,
            max_num_pairs=max_num_pairs,
            objective=objective,
        )

        if verbose:
            print(f"\n[OPRO-TE] Step {i_step + 1}/{num_steps}")

        # All num_decode_per_step samples are requested in a single HTTP call
        # via the OpenAI `n` parameter; vLLM batches them on the GPU.
        raw_outputs = call_vllm_server_single_prompt(
            meta_prompt,
            base_url=vllm_base_url,
            model=model_name,
            max_decode_steps=max_decode_tokens,
            temperature=temperature,
            api_key=api_key,
            n=num_decode_per_step,
            client=_llm_client,
        )
        print(raw_outputs)
        if isinstance(raw_outputs, str):
            raw_outputs = [raw_outputs]

        for raw_out in raw_outputs:
            routing = extract_routing(raw_out, num_od)
            if routing is None:
                if verbose:
                    print("  [parse] failed to parse LLM output, skipping.")
                continue

            score = evaluate_routing(routing, tm, commodities, edge_cap, objective)
            routing_str = "|".join(",".join(f"{f:.1f}" for f in fracs) for fracs in routing)
            old_value_pairs_set.add((routing_str, score))

            if _is_better(score, best_score, objective):
                best_score = score
                best_routing = [list(fracs) for fracs in routing]
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
