import numpy as np
import time

def run_timing_cpu(adj, delays, mc_runs=1000, sigma=0.1):
    n = len(delays)
    results = []

    start = time.time()

    for _ in range(mc_runs):
        perturbed = delays * np.random.normal(1.0, sigma, size=n)
        path_delay = adj @ perturbed
        total_delay = np.max(path_delay + perturbed)
        results.append(total_delay)

    runtime = time.time() - start
    return np.array(results), runtime
