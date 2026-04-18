from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DerivedConfig:
    n_users: int
    k_active: int
    power_budget: float
    bandwidth_low: float
    bandwidth_high: float
    channel_log_std: float


def derive_defaults(n_users: int) -> DerivedConfig:
    """
    Derive all internal simulation parameters from the user-specified number of users.

    Rules are intentionally simple and deterministic:
    - active-user budget grows sublinearly with N
    - total power budget grows linearly with active-user budget
    - channel and bandwidth generation ranges are fixed internally
    """
    if n_users < 2:
        raise ValueError("n_users must be at least 2")

    k_active = max(2, min(n_users, int(round(math.sqrt(n_users)))))
    power_budget = float(2 * k_active + 2)
    return DerivedConfig(
        n_users=n_users,
        k_active=k_active,
        power_budget=power_budget,
        bandwidth_low=0.2,
        bandwidth_high=3.0,
        channel_log_std=1.2,
    )


def _waterfill_on_support(
    B: np.ndarray,
    a: np.ndarray,
    support: Sequence[int],
    p_max: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[np.ndarray, float]:
    n = len(B)
    q = np.zeros(n, dtype=float)

    active = [i for i in support if 0 <= i < n]
    if len(active) == 0 or p_max <= 0:
        return q, 0.0

    def total_power(nu: float) -> float:
        return sum(max(B[i] / nu - 1.0 / a[i], 0.0) for i in active)

    hi = max(B[i] * a[i] for i in active) + 1.0
    while total_power(hi) > p_max:
        hi *= 2.0
    lo = 1e-15

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if total_power(mid) > p_max:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break

    nu = hi
    for i in active:
        q[i] = max(B[i] / nu - 1.0 / a[i], 0.0)

    total_q = float(q.sum())
    if total_q > p_max and total_q > 0:
        q *= p_max / total_q

    value = float(np.sum(B * np.log1p(a * q)))
    return q, value


def solve_greedy(
    n_users: int,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Solve one internally generated instance using the proposed greedy scheduler.

    User-facing inputs are intentionally minimal:
    - n_users: system size
    - seed: randomness control

    All other quantities (scheduled-user budget, power budget, channel model,
    user weights) are derived internally.
    """
    cfg = derive_defaults(n_users)
    rng = np.random.default_rng(seed)

    B = rng.uniform(cfg.bandwidth_low, cfg.bandwidth_high, size=cfg.n_users)
    a = np.exp(rng.normal(0.0, cfg.channel_log_std, size=cfg.n_users))

    selected: List[int] = []
    remaining = set(range(cfg.n_users))
    current_q = np.zeros(cfg.n_users, dtype=float)
    current_value = 0.0

    t0 = time.perf_counter()
    for _ in range(cfg.k_active):
        best_user = None
        best_q = None
        best_value = -np.inf

        for j in remaining:
            candidate_support = selected + [j]
            q, value = _waterfill_on_support(B, a, candidate_support, cfg.power_budget)
            if value > best_value + 1e-12:
                best_value = value
                best_user = j
                best_q = q

        if best_user is None:
            break
        selected.append(best_user)
        remaining.remove(best_user)
        current_q = best_q
        current_value = best_value
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "n_users": cfg.n_users,
        "k_active": cfg.k_active,
        "power_budget": cfg.power_budget,
        "seed": seed,
        "scheduled_users": tuple(sorted(selected)),
        "power_allocation": current_q.tolist(),
        "weighted_sum_rate": current_value,
        "runtime_ms": runtime_ms,
    }


def _oracle_small(B: np.ndarray, a: np.ndarray, k: int, p_max: float) -> Dict[str, object]:
    n = len(B)
    best = None
    for kk in range(0, min(k, n) + 1):
        for support in itertools.combinations(range(n), kk):
            q, value = _waterfill_on_support(B, a, support, p_max)
            if best is None or value > best["objective"] + 1e-12:
                best = {
                    "support": tuple(support),
                    "q": q,
                    "objective": value,
                }
    assert best is not None
    return best


def simulate(
    n_users: int,
    n_trials: int = 10,
    seed: int = 0,
    oracle_check: bool | None = None,
) -> Dict[str, object]:
    """
    Run a Monte Carlo study with internally derived parameters.

    Minimal user parameters:
    - n_users
    - n_trials
    - seed

    If oracle_check is None, it is enabled automatically for n_users <= 18.
    """
    cfg = derive_defaults(n_users)
    rng = np.random.default_rng(seed)
    if oracle_check is None:
        oracle_check = n_users <= 18

    rates: List[float] = []
    runtimes: List[float] = []
    oracle_rates: List[float] = []
    oracle_runtimes: List[float] = []
    exact_matches = 0

    for _ in range(n_trials):
        B = rng.uniform(cfg.bandwidth_low, cfg.bandwidth_high, size=cfg.n_users)
        a = np.exp(rng.normal(0.0, cfg.channel_log_std, size=cfg.n_users))

        selected: List[int] = []
        remaining = set(range(cfg.n_users))
        current_q = np.zeros(cfg.n_users, dtype=float)
        current_value = 0.0

        t0 = time.perf_counter()
        for _ in range(cfg.k_active):
            best_user = None
            best_q = None
            best_value = -np.inf
            for j in remaining:
                q, value = _waterfill_on_support(B, a, selected + [j], cfg.power_budget)
                if value > best_value + 1e-12:
                    best_value = value
                    best_user = j
                    best_q = q
            selected.append(best_user)
            remaining.remove(best_user)
            current_q = best_q
            current_value = best_value
        runtimes.append((time.perf_counter() - t0) * 1000.0)
        rates.append(current_value)

        if oracle_check:
            t0 = time.perf_counter()
            oracle = _oracle_small(B, a, cfg.k_active, cfg.power_budget)
            oracle_runtimes.append((time.perf_counter() - t0) * 1000.0)
            oracle_rates.append(oracle["objective"])
            exact_matches += int(tuple(sorted(selected)) == oracle["support"])

    out: Dict[str, object] = {
        "n_users": cfg.n_users,
        "k_active": cfg.k_active,
        "power_budget": cfg.power_budget,
        "n_trials": n_trials,
        "seed": seed,
        "greedy_avg_rate": float(np.mean(rates)),
        "greedy_std_rate": float(np.std(rates)),
        "greedy_avg_runtime_ms": float(np.mean(runtimes)),
        "oracle_checked": oracle_check,
    }
    if oracle_check:
        oracle_rates_np = np.array(oracle_rates, dtype=float)
        rates_np = np.array(rates, dtype=float)
        gap_pct = 100.0 * (oracle_rates_np - rates_np) / np.maximum(oracle_rates_np, 1e-12)
        out.update(
            {
                "oracle_avg_rate": float(np.mean(oracle_rates_np)),
                "oracle_avg_runtime_ms": float(np.mean(oracle_runtimes)),
                "exact_match_rate": float(exact_matches / n_trials),
                "avg_oracle_gap_pct": float(np.mean(gap_pct)),
            }
        )
    return out
