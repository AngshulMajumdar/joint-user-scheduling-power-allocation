# joint-user-scheduling-power-allocation

Minimal Python repository for **joint user scheduling and power allocation** using a greedy scheduler with exact inner power allocation.

## Design goal
The user defines only the essentials:
- `n_users`
- optional random `seed`
- optional number of Monte Carlo `trials`

Everything else is derived internally:
- number of scheduled users
- total power budget
- channel-generation parameters
- bandwidth-weight ranges

## Model
The code solves instances of

\[
\max_{q \ge 0} \sum_{i=1}^N B_i \log(1 + a_i q_i)
\quad \text{s.t.} \quad
\|q\|_0 \le K,
\quad
\sum_{i=1}^N q_i \le P_{\max}.
\]

The greedy scheduler builds the scheduled-user set one user at a time using exact marginal weighted sum-rate evaluation.  
The inner power allocation on any fixed scheduled-user set is solved by the KKT / water-filling rule.

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Single instance
```bash
python -m user_sched.cli 20 --seed 7
```

## Monte Carlo simulation
```bash
python -m user_sched.cli 10 --seed 42 --trials 5
```

## Python API
```python
from user_sched import solve_greedy, simulate

result = solve_greedy(n_users=30, seed=0)
summary = simulate(n_users=12, n_trials=10, seed=0)
```

## Internal parameter derivation
For a given `n_users = N`, the repository internally sets:
- scheduled-user budget: `K = round(sqrt(N))`, clipped to a valid range
- total power budget: `P_max = 2K + 2`
- bandwidth weights: `B_i ~ Uniform[0.2, 3]`
- effective channel coefficients: `a_i ~ exp(Normal(0, 1.2^2))`

These rules keep the public interface minimal while still generating meaningful user-scheduling instances.

## Notes
This repository focuses on the **greedy scheduler** and its simulation harness. It does not include the convex surrogate baseline.
