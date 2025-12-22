import numpy as np

def load_netlist(path):
    nodes = {}
    edges = []

    with open(path, "r") as f:
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
                nodes[name] = float(delay)
            else:
                src, dst = line.split()
                edges.append((src, dst))

    node_list = list(nodes.keys())
    node_index = {n: i for i, n in enumerate(node_list)}

    delays = np.array([nodes[n] for n in node_list])

    adj = np.zeros((len(node_list), len(node_list)))
    for s, d in edges:
        adj[node_index[d], node_index[s]] = 1.0

    return adj, delays, node_list
