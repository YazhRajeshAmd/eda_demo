import numpy as np

def load_netlist(path):
    nodes = {}
    delays = []
    edges = []

    with open(path) as f:
        mode = "nodes"
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line == "EDGES":
                mode = "edges"
                continue

            if mode == "nodes":
                name, delay = line.split()
                nodes[name] = len(nodes)
                delays.append(float(delay))
            else:
                src, dst = line.split()
                edges.append((nodes[src], nodes[dst]))

    return edges, np.array(delays, dtype=np.float32), list(nodes.keys())
