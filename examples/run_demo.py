from user_sched import simulate, solve_greedy

print("Single instance demo")
print(solve_greedy(n_users=20, seed=7))

print("\nMonte Carlo demo")
print(simulate(n_users=10, n_trials=5, seed=42))
