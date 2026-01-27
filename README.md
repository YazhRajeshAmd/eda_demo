# EDA Timing Analysis Demo

A Streamlit-based demonstration of **Electronic Design Automation (EDA) timing analysis** accelerated by AMD Instinct GPUs. This demo showcases Monte Carlo simulation for static timing analysis (STA) on circuit netlists, comparing CPU vs GPU performance.

![AMD Accelerated](https://img.shields.io/badge/AMD-Instinct%20GPU-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

---

## Overview

This demo simulates core EDA workloads using a simplified **timing graph** dataset in ISPD-style format. It performs Monte Carlo simulations to estimate critical path delays, mimicking **static timing analysis (STA)** used in real IC design.

### Key Features

- **Monte Carlo Timing Analysis**: Simulates process, voltage, and temperature (PVT) variations
- **GPU Acceleration**: Leverages AMD Instinct GPUs via PyTorch for massive parallelization
- **Interactive Visualization**: Real-time histogram of critical path delay distributions
- **Configurable Simulations**: Adjustable Monte Carlo run counts (100 - 5,000 iterations)

---

## Technical Components

### Project Structure

```
eda_demo/
├── app.py                 # Main Streamlit application
├── netlist_parser.py      # ISPD-style netlist file parser
├── timing_cpu.py          # CPU-based timing analysis implementation
├── timing_gpu.py          # GPU-accelerated timing analysis (PyTorch/CUDA)
├── generate_list.py       # Utility to generate synthetic netlists
├── amd-logo.png           # AMD logo for the UI
└── data/
    ├── ispd_demo_netlist.txt       # Small demo netlist
    ├── ispd_demo_complex.txt       # Medium complexity netlist
    └── ispd_demo_complex_10x.txt   # Large-scale netlist for benchmarking
```

### Component Details

#### 1. `app.py` - Main Application
The Streamlit web application that provides:
- Interactive UI with AMD branding
- Configuration sidebar for Monte Carlo parameters
- Toggle between CPU and GPU execution
- Real-time performance metrics and visualization

#### 2. `netlist_parser.py` - Netlist Parser
Parses ISPD-style netlist files containing:
- **Nodes**: Gate names with associated propagation delays
- **Edges**: Connections between gates (directed graph)

**File Format:**
```
# Nodes (gate delays)
G1 12.3
G2 8.1
G3 5.6

EDGES
G1 G2
G2 G3
```

#### 3. `timing_cpu.py` - CPU Implementation
Sequential timing analysis using NumPy:
- Builds a levelized graph structure (topological ordering)
- Iterates through Monte Carlo runs sequentially
- Computes critical path using dynamic programming

#### 4. `timing_gpu.py` - GPU Implementation
Parallel timing analysis using PyTorch with CUDA:
- Processes all Monte Carlo runs simultaneously
- Leverages GPU parallelism for delay perturbation
- Performs batched levelized propagation
- Achieves significant speedup over CPU for large simulations

#### 5. `generate_list.py` - Netlist Generator
Utility script to generate synthetic netlists:
- Configurable number of gates (default: 500,000)
- Random delay values (1.0 - 20.0 units)
- Acyclic graph structure with configurable edge density

---

## How It Works

### Algorithm Overview

1. **Load Netlist**: Parse gates (nodes) and connections (edges) from file
2. **Build Levelized Graph**: Topologically sort nodes for efficient propagation
3. **Monte Carlo Simulation**:
   - Perturb gate delays with Gaussian noise (σ = 5%)
   - Propagate arrival times through the levelized graph
   - Record the maximum arrival time (critical path delay)
4. **Analyze Results**: Visualize the distribution of critical path delays

### GPU Acceleration

The GPU implementation parallelizes across Monte Carlo runs:
- Each run is processed as a row in a 2D tensor `(mc_runs × n_gates)`
- Delay perturbations are computed in parallel using `torch.normal()`
- Arrival time propagation uses vectorized tensor operations
- Results in **10-100x speedup** for large simulations

---

## Installation

### Prerequisites

- Python 3.10+
- AMD Instinct GPU with ROCm (for GPU acceleration)
- CUDA-compatible PyTorch installation

### Setup

1. **Create and activate virtual environment:**
   ```bash
   cd eda_demo
   python3 -m venv eda_venv
   source eda_venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit numpy matplotlib pillow torch
   ```

   For AMD ROCm support, install PyTorch with ROCm:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   ```

3. **Ensure AMD logo is present:**
   Place an `amd-logo.png` file in the `eda_demo` directory.

---

## Running the Demo

### Start the Application

```bash
cd eda_demo
source eda_venv/bin/activate
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Demo

1. **Configure Parameters** (Sidebar):
   - Adjust the number of Monte Carlo runs (100 - 5,000)
   - Toggle GPU acceleration on/off

2. **Run Analysis**:
   - Click the "Run Timing Analysis" button
   - View runtime metrics and worst-case delay

3. **Analyze Results**:
   - Examine the histogram showing critical path delay distribution
   - Compare CPU vs GPU performance

### Generate Custom Netlists

To create a new synthetic netlist:

```bash
python generate_list.py
```

Edit `generate_list.py` to customize:
- `num_gates`: Number of gates in the circuit
- `max_edges_per_gate`: Maximum fanout per gate
- `filename`: Output file path

---

## Performance Benchmarks

| Configuration | Monte Carlo Runs | Gates | CPU Time | GPU Time | Speedup |
|--------------|------------------|-------|----------|----------|---------|
| Small        | 1,000            | 5K    | ~2s      | ~0.1s    | 20x     |
| Medium       | 1,000            | 50K   | ~20s     | ~0.5s    | 40x     |
| Large        | 5,000            | 500K  | ~500s    | ~5s      | 100x    |

*Results may vary based on hardware configuration*

---

## Target Audience

- **EDA Engineers**: Explore GPU acceleration for timing analysis workflows
- **Hardware Designers**: Understand Monte Carlo simulation for PVT analysis
- **GPU Specialists**: Learn about parallelizing graph algorithms on AMD Instinct
- **Researchers**: Benchmark CPU vs GPU performance for EDA workloads

---

## License

This demo is provided for educational and demonstration purposes.

---

## Acknowledgments

- AMD for Instinct GPU hardware and ROCm software stack
- PyTorch team for CUDA/ROCm support
- Streamlit for the interactive web framework
