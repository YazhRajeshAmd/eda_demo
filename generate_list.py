import random

num_gates = 500000        # Number of gates
max_edges_per_gate = 5  # Maximum outgoing connections
filename = "data/ispd_demo_complex_10x.txt"

# --- Generate nodes ---
nodes = []
for i in range(1, num_gates + 1):
    delay = round(random.uniform(1.0, 20.0), 2)
    nodes.append(f"G{i} {delay}")

# --- Generate edges (acyclic graph) ---
edges = []
for i in range(1, num_gates):
    num_edges = random.randint(1, max_edges_per_gate)
    targets = random.sample(range(i+1, num_gates+1), min(num_edges, num_gates-i))
    for t in targets:
        edges.append(f"G{i} G{t}")

# --- Write to file ---
with open(filename, "w") as f:
    f.write("# Nodes\n")
    f.write("\n".join(nodes))
    f.write("\nEDGES\n")
    f.write("\n".join(edges))

print(f"Complex netlist written to {filename}")
