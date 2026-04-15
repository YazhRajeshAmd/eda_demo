import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing as mp
import os
import time

from netlist_parser import load_netlist
from timing_cpu import run_timing_cpu
from timing_gpu import run_timing_gpu

# -------------------------------
# CONFIG
# -------------------------------
NUM_GPUS = 8

# -------------------------------
# MULTI-GPU WORKER
# -------------------------------
def gpu_worker(args):
    edges, delays, runs, gpu_id = args

    # Bind this process to ONE GPU
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    results, _ = run_timing_gpu(edges, delays, runs)
    runtime = time.time() - start

    return results, runtime


# -------------------------------
# MULTI-GPU DISPATCHER
# -------------------------------
def run_multi_gpu(edges, delays, total_runs):
    runs_per_gpu = total_runs // NUM_GPUS
    remainder = total_runs % NUM_GPUS

    tasks = []
    for i in range(NUM_GPUS):
        runs = runs_per_gpu + (1 if i < remainder else 0)
        tasks.append((edges, delays, runs, i))

    with mp.Pool(NUM_GPUS) as pool:
        outputs = pool.map(gpu_worker, tasks)

    # Combine all GPU results
    all_results = np.concatenate([o[0] for o in outputs])

    # Wall-clock time approximation
    max_runtime = max(o[1] for o in outputs)

    return all_results, max_runtime


# -------------------------------
# STREAMLIT UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="EDA Timing Analysis Demo",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    col1, col2 = st.columns([4,1])
    with col1:
        st.title("EDA Timing Analysis Accelerated by 8x AMD Instinct GPUs")
    with col2:
        amd_logo = Image.open("amd-logo.png")
        st.image(amd_logo, width=120)

    # Info
    st.markdown("""
    ### About this Demo
    Monte Carlo-based **EDA timing analysis** accelerated across **8 AMD GPUs**.

    - Parallel simulation across GPUs
    - Each GPU handles a chunk of runs
    - Results are merged for final statistics
    """)

    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")
    mc_runs = st.sidebar.slider("Monte Carlo Runs", 100, 20000, 4000)
    use_gpu = st.sidebar.checkbox("Use 8 GPUs (AMD Instinct)", True)

    # Load data
    edges, delays, nodes = load_netlist("data/ispd_demo_complex_10x.txt")
    st.write(f"**Loaded Netlist:** {len(nodes)} gates")

    # Run
    if st.button("Run Timing Analysis"):
        with st.spinner("Running analysis on 8 GPUs..."):

            if use_gpu:
                results, runtime = run_multi_gpu(edges, delays, mc_runs)
            else:
                results, runtime = run_timing_cpu(edges, delays, mc_runs)

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Runtime (seconds)", f"{runtime:.3f}")
        with col2:
            st.metric("Worst-Case Delay", f"{results.max():.2f}")

        # Histogram
        st.markdown("### Monte Carlo Critical Path Delay Distribution")
        fig, ax = plt.subplots()
        ax.hist(results, bins=50, alpha=0.8)
        ax.set_xlabel("Critical Path Delay")
        ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo Timing Distribution (8 GPUs)")
        st.pyplot(fig)

    st.markdown("---")

    st.markdown("""
    ### What Changed for 8 GPUs?

    1. Workload split into 8 chunks  
    2. Each chunk runs on a separate GPU  
    3. Parallel execution via multiprocessing  
    4. Results combined into one distribution  

    This mimics real-world **EDA acceleration scaling**.
    """)


# -------------------------------
# ENTRY POINT (IMPORTANT)
# -------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer for Streamlit + ROCm
    main()
