from __future__ import annotations

import argparse
import json

from .solver import simulate, solve_greedy


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy user scheduling and power allocation")
    parser.add_argument("n_users", type=int, help="Number of candidate users")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--trials", type=int, default=0, help="If >0, run Monte Carlo simulation")
    args = parser.parse_args()

    if args.trials > 0:
        result = simulate(n_users=args.n_users, n_trials=args.trials, seed=args.seed)
    else:
        result = solve_greedy(n_users=args.n_users, seed=args.seed)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
