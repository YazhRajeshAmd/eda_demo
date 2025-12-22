import torch
import time

def run_timing_gpu(adj, delays, mc_runs=1000, sigma=0.1):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU not available. Check PyTorch install.")

    device = torch.device("cuda")

    adj_t = torch.tensor(adj, device=device, dtype=torch.float32)
    delays_t = torch.tensor(delays, device=device, dtype=torch.float32)

    results = []

    start = time.time()

    for _ in range(mc_runs):
        noise = torch.normal(
            mean=1.0,
            std=sigma,
            size=delays_t.shape,
            device=device
        )
        perturbed = delays_t * noise
        path_delay = torch.matmul(adj_t, perturbed)
        total_delay = torch.max(path_delay + perturbed)
        results.append(total_delay)

    torch.cuda.synchronize()
    runtime = time.time() - start

    return torch.stack(results).cpu().numpy(), runtime
