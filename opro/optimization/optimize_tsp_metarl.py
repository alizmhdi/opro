# Copyright 2024 The OPRO Authors (MetaRL TSP extension)
"""OPRO optimization loop for TSP — callable from MetaRL adversarial search."""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_OPRO_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _OPRO_PKG_DIR not in sys.path:
    sys.path.insert(0, _OPRO_PKG_DIR)

from opro.prompt_utils import call_vllm_server_single_prompt  # noqa: E402


def evaluate_distance(x, y, trace, num_decimals=0):
    dis = 0.0
    try:
        for i in range(len(trace) - 1):
            id0, id1 = trace[i], trace[i + 1]
            dis += float(np.hypot(x[id0] - x[id1], y[id0] - y[id1]))
    except Exception:
        return -1
    id0, id1 = trace[-1], trace[0]
    dis += float(np.hypot(x[id0] - x[id1], y[id0] - y[id1]))
    if num_decimals > 0:
        dis = round(dis, num_decimals)
    else:
        dis = int(dis)
    return dis


def gen_meta_prompt(old_value_pairs_set, x, y, max_num_pairs=10):
    old_value_pairs = sorted(old_value_pairs_set, key=lambda item: -item[1])[-max_num_pairs:]
    exemplars = ""
    for trace, dis in old_value_pairs:
        exemplars += f"\n<trace> {trace} </trace>\nlength:\n{dis}\n"
    meta_prompt = "You are given a list of points with coordinates below:\n"
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i:
            meta_prompt += ", "
        meta_prompt += f"({i}): ({xi}, {yi})"
    meta_prompt += (
        ".\n\nBelow are some previous traces and their lengths. The traces are "
        "arranged in descending order based on their lengths, where lower values "
        "are better."
    )
    meta_prompt += "\n\n"
    meta_prompt += exemplars.strip()
    meta_prompt += "\n\n"
    meta_prompt += (
        "Give me a new trace that is different from all traces above, and has a "
        "length lower than any of the above. The trace should traverse all points "
        "exactly once, starting at city 0. "
        "Output only the final solution: start with '<trace>' and end with '</trace>'. "
        "Do not include any explanation, reasoning, or other text outside those tags."
    )
    return meta_prompt.strip()


def extract_trace(input_string):
    """Parse ``<trace> i,j,k,... </trace>`` from LLM output."""
    start_string, end_string = "<trace>", "</trace>"
    if start_string not in input_string:
        return None
    body = input_string.split(start_string, 1)[1]
    if end_string in body:
        body = body.split(end_string, 1)[0]
    parsed = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.append(int(part))
        except ValueError:
            continue
    return parsed if parsed else None


def nearest_neighbor_seed(x, y, num_points, num_decimals=0):
    gt_sol = [0]
    remaining = list(range(1, num_points))
    min_dis = 0.0
    while remaining:
        min_p, min_cur = -1, -1.0
        for p in remaining:
            cur = float(np.hypot(x[p] - x[gt_sol[-1]], y[p] - y[gt_sol[-1]]))
            if min_p == -1 or cur < min_cur:
                min_p, min_cur = p, cur
        gt_sol.append(min_p)
        min_dis += min_cur
        remaining.remove(min_p)
    min_dis += float(np.hypot(x[0] - x[gt_sol[-1]], y[0] - y[gt_sol[-1]]))
    if num_decimals > 0:
        min_dis = round(min_dis, num_decimals)
    else:
        min_dis = int(min_dis)
    return gt_sol, min_dis


def random_starting_tours(num_points, count):
    tours = []
    nodes = list(range(1, num_points))
    attempts = 0
    while len(tours) < count and attempts < count * 20:
        attempts += 1
        perm = np.random.permutation(nodes).tolist()
        tours.append([0] + perm)
    return tours


def _coords_to_xy(coords, num_decimals=0):
    coords = np.asarray(coords, dtype=np.float64)
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    if num_decimals > 0:
        x = [round(v, num_decimals) for v in x]
        y = [round(v, num_decimals) for v in y]
    else:
        x = [int(v) for v in x]
        y = [int(v) for v in y]
    return x, y


def run_opro_tsp(
    coords,
    num_steps: int = 50,
    num_decode_per_step: int = 8,
    max_num_pairs: int = 10,
    num_starting_points: int = 5,
    early_stop_patience: int = 0,
    verbose: bool = False,
    vllm_base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_decode_tokens: int = 1024,
    temperature: float = 1.0,
    api_key: str = "EMPTY",
    num_decimals: int = 0,
    warm_state: Optional[dict] = None,
):
    """Run OPRO on one TSP instance (MetaRL ``solve(coords)`` backend).

    Parameters
    ----------
    num_steps : int
        OPRO meta-prompt iterations. Use 1--5 inside MetaRL; 500 matches the
        standalone ``optimize_tsp.py`` paper script.
    early_stop_patience : int
        Stop after this many consecutive steps with no improvement (0 = off).
    warm_state : dict, optional
        Reused across calls with the same ``n`` (pairs set + best tour).

    Returns
    -------
    best_length : float or None
    best_tour : np.ndarray or None
    """
    from openai import OpenAI as _OpenAIClient  # noqa: PLC0415

    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if n <= 1:
        tour = list(range(n))
        return evaluate_distance(*_coords_to_xy(coords, num_decimals), tour, num_decimals), np.array(tour, dtype=int)

    x, y = _coords_to_xy(coords, num_decimals=num_decimals)
    num_points = n

    old_value_pairs_set: set = set()
    if warm_state is not None and warm_state.get('n') == n:
        old_value_pairs_set = set(warm_state.get('pairs') or [])
        if warm_state.get('best_tour') is not None:
            _seed_tour = list(warm_state['best_tour'])
        else:
            _seed_tour = None
    else:
        _seed_tour = None

    best_length: Optional[float] = None
    best_tour: Optional[list[int]] = None

    def _consider(trace):
        nonlocal best_length, best_tour
        if trace is None:
            return
        if len(set(trace)) != num_points or len(trace) != num_points or trace[0] != 0:
            return
        dis = evaluate_distance(x, y, trace, num_decimals)
        if dis == -1:
            return
        trace_str = ",".join(str(i) for i in trace)
        old_value_pairs_set.add((trace_str, dis))
        if best_length is None or dis < best_length:
            best_length = dis
            best_tour = list(trace)

    if _seed_tour is not None:
        _consider(_seed_tour)
    nn_sol, _ = nearest_neighbor_seed(x, y, num_points, num_decimals)
    _consider(nn_sol)
    for sol in random_starting_tours(num_points, max(0, num_starting_points - 1)):
        _consider(sol)

    if verbose and best_length is not None:
        print(f"[OPRO-TSP] Initial best tour_length: {best_length}")

    if num_steps <= 0:
        if warm_state is not None:
            warm_state['n'] = n
            warm_state['pairs'] = old_value_pairs_set
            warm_state['best_tour'] = best_tour
        if best_tour is None:
            return None, None
        return float(best_length), np.array(best_tour, dtype=int)

    client = _OpenAIClient(base_url=vllm_base_url, api_key=api_key)
    steps_without_improve = 0

    for i_step in range(num_steps):
        prev_best = best_length
        meta_prompt = gen_meta_prompt(old_value_pairs_set, x, y, max_num_pairs=max_num_pairs)
        if verbose:
            print(f"\n[OPRO-TSP] Step {i_step + 1}/{num_steps}")
        raw_outputs = call_vllm_server_single_prompt(
            meta_prompt,
            base_url=vllm_base_url,
            model=model_name,
            max_decode_steps=max_decode_tokens,
            temperature=temperature,
            api_key=api_key,
            n=num_decode_per_step,
            client=client,
        )
        if isinstance(raw_outputs, str):
            raw_outputs = [raw_outputs]

        for raw_out in raw_outputs:
            parsed = extract_trace(raw_out)
            _consider(parsed)
            if verbose and best_length is not None:
                print(f"  [update] best tour_length: {best_length}")

        if early_stop_patience > 0:
            if prev_best is not None and best_length is not None and best_length >= prev_best - 1e-9:
                steps_without_improve += 1
                if steps_without_improve >= early_stop_patience:
                    if verbose:
                        print(f"[OPRO-TSP] Early stop after {steps_without_improve} stale steps")
                    break
            else:
                steps_without_improve = 0

    if verbose and best_length is not None:
        print(f"[OPRO-TSP] Final best tour_length: {best_length}")

    if warm_state is not None:
        warm_state['n'] = n
        warm_state['pairs'] = old_value_pairs_set
        warm_state['best_tour'] = best_tour

    if best_tour is None:
        return None, None
    return float(best_length), np.array(best_tour, dtype=int)
