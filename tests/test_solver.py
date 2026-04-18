from user_sched import derive_defaults, simulate, solve_greedy


def test_defaults_are_valid():
    cfg = derive_defaults(25)
    assert cfg.k_active >= 2
    assert cfg.k_active <= cfg.n_users
    assert cfg.power_budget > 0


def test_single_instance_runs():
    out = solve_greedy(n_users=12, seed=1)
    assert len(out["scheduled_users"]) == out["k_active"]
    assert out["weighted_sum_rate"] >= 0.0


def test_small_oracle_simulation_runs():
    out = simulate(n_users=8, n_trials=3, seed=2)
    assert out["oracle_checked"] is True
    assert 0.0 <= out["exact_match_rate"] <= 1.0
