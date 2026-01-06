import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from netlist_parser import load_netlist
from timing_cpu import run_timing_cpu
from timing_gpu import run_timing_gpu

# --- Page Config ---
st.set_page_config(
    page_title="EDA Timing Analysis Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header with AMD Logo ---
col1, col2 = st.columns([4,1])
with col1:
    st.title("EDA Timing Analysis Accelerated by AMD Instinct GPUs")
with col2:
    # Load AMD logo (ensure you have a local AMD logo file in project folder)
    amd_logo = Image.open("amd-logo.png")  # Replace with your AMD logo path
    st.image(amd_logo, width=120)

# --- Dataset / Audience Explanation ---
st.markdown("""
### About this Demo

This demo simulates **core Electronic Design Automation (EDA) workloads** using a simplified **timing graph** dataset. 

- **Dataset**: ISPD-style netlist with gates (nodes) and nets (edges), where delays represent gate propagation time.
- **Purpose**: Perform Monte Carlo simulations to estimate critical path delays, mimicking **static timing analysis (STA)** in real IC design.
- **Target Audience**: 
  - EDA engineers and designers
  - Hardware acceleration specialists
  - GPU enthusiasts interested in high-performance circuit analysis
""")

st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
mc_runs = st.sidebar.slider("Monte Carlo Runs", 100, 5000, 1000)
use_gpu = st.sidebar.checkbox("Use GPU (AMD Instinct)", True)

# --- Load Data ---
edges, delays, nodes = load_netlist("data/ispd_demo_complex_10x.txt")
st.write(f"**Loaded Netlist:** {len(nodes)} gates")

# --- Run Analysis ---
if st.button("Run Timing Analysis"):
    with st.spinner("Running analysis..."):
        if use_gpu:
            results, runtime = run_timing_gpu(edges, delays, mc_runs)
        else:
            results, runtime = run_timing_cpu(edges, delays, mc_runs)

    # --- Display Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Runtime (seconds)", f"{runtime:.3f}")
    with col2:
        st.metric("Worst-Case Delay", f"{results.max():.2f}")

    # --- Histogram of Monte Carlo Results ---
    st.markdown("### Monte Carlo Critical Path Delay Distribution")
    st.markdown(
        "The histogram shows the distribution of critical path delays across "
        f"{mc_runs} Monte Carlo simulations. GPU acceleration (AMD Instinct) "
        "significantly reduces runtime compared to CPU-only execution."
    )
    fig, ax = plt.subplots()
    ax.hist(results, bins=50, color="#E53E3E", alpha=0.8)
    ax.set_xlabel("Critical Path Delay")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Timing Distribution")
    st.pyplot(fig)

# --- Explanation Box ---
st.markdown("---")
st.markdown("""
### What is Happening in this Demo?

1. **Dataset**: We load a simplified ISPD-style netlist (gates + connections).
2. **Monte Carlo Analysis**: We perturb gate delays randomly to simulate process, voltage, and temperature variations.
3. **Critical Path Computation**: For each simulation, the longest path through the circuit (critical path) is computed.
4. **Acceleration**: If GPU is enabled, the computations are performed in parallel on an AMD Instinct GPU, significantly speeding up large-scale simulations.
5. **Visualization**: The histogram shows the variability in critical path delays, helping EDA engineers understand worst-case timing behavior.
""")
