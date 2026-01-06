import torch
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

def run_timing_gpu(edges, delays, mc_runs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(delays)

    # Build graph once (CPU)
    levels, adj = build_levels(edges, n)

    # Move static data to GPU
    delays = torch.tensor(delays, device=device)

    start = time.time()

    # (mc_runs, n)
    perturbed = delays * torch.normal(
        1.0, 0.05, size=(mc_runs, n), device=device
    )
    arrival = torch.zeros((mc_runs, n), device=device)

    # Levelized propagation
    for level in levels:
        u = torch.tensor(level, device=device)
        u_val = arrival[:, u]                # (mc, |level|)
        for ui in level:
            for v in adj[ui]:
                arrival[:, v] = torch.maximum(
                    arrival[:, v],
                    arrival[:, ui] + perturbed[:, v]
                )

    results = arrival.max(dim=1).values
    return results.cpu().numpy(), time.time() - start
