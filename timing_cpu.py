import numpy as np
import time
from collections import defaultdict, deque

def build_levels(edges, n):
    adj = defaultdict(list)
    indeg = [0] * n

    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    q = deque([i for i in range(n) if indeg[i] == 0])
    levels = []

    while q:
        level = list(q)
        levels.append(level)
        q.clear()

        for u in level:
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    return levels, adj

def run_timing_cpu(edges, delays, mc_runs):
    n = len(delays)
    levels, adj = build_levels(edges, n)

    start = time.time()
    results = np.zeros(mc_runs)

    for mc in range(mc_runs):
        perturbed = delays * np.random.normal(1.0, 0.05, n)
        arrival = np.zeros(n)

        for level in levels:
            for u in level:
                for v in adj[u]:
                    arrival[v] = max(arrival[v], arrival[u] + perturbed[v])

        results[mc] = arrival.max()

    return results, time.time() - start
