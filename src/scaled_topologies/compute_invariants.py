# Computes graph invariants for the Phase 2 dataset.
# Reads the official "known" graph set (graphs.pkl) by default; see __main__
# for the commented-out alternative that computes invariants for the
# experimental/ held-out "test" batch instead.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import networkx.algorithms.approximation as approx
import numpy as np
import pickle
import pandas as pd
from utils import test_utils

def exact_maxcut_brute(G):
    """Brute force exact MaxCut — feasible at n<=20."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in G.edges():
            i, j = nodes.index(u), nodes.index(v)
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        best = max(best, cut)
    return best

def compute_invariants(G):
    n = G.number_of_nodes()
    #print(f"type n: {type(n)}")
    m = G.number_of_edges()
    #print(f"type m: {type(m)}")
    degrees = [d for _, d in G.degree()]
    #print(f"type degrees: {type(degrees[0])}")
    
    row = {
        "graph_id":      G.graph.get('graph_id', -1),
        "model":         G.graph.get('model', 'unknown'),
        "n":             n,
        "m":             m,
        "deg_mean":      round(np.mean(degrees), 4),
        "deg_std":       round(np.std(degrees), 4),
        "clust_mean":    round(nx.average_clustering(G), 4),
        "assortativity": round(nx.degree_assortativity_coefficient(G), 4),
        #"chromatic":     None,  # expensive — optional
        "mis":           round(
                            len(approx.maximum_independent_set(G)) / n, 4
                         ),
        "exact_maxcut":  exact_maxcut_brute(G),
    }
    
    # Chromatic number approximation (greedy — fast)
    #col = nx.coloring.greedy_color(G, strategy='largest_first')
    #row["chromatic"] = max(col.values()) + 1
    
    return row

if __name__ == "__main__":
    test_utils.set_dir(test_utils.get_path())
    folder = "../../data/scaled_topologies/graphs.pkl"
    # folder = "../../data/scaled_topologies/experimental/test_graphs.pkl"  # experimental/ held-out batch
    with open(folder, "rb") as f:
        graphs = pickle.load(f)

    rows = []
    for i, G in enumerate(graphs):
        if i % 50 == 0:
            print(f"Invariants: {i}/{len(graphs)}")
        rows.append(compute_invariants(G))

    df = pd.DataFrame(rows)
    df.to_csv("../../results/scaled_topologies/invariants.csv", index=False)
    # df.to_csv("../../results/scaled_topologies/experimental/test_invariants.csv", index=False)
    print("Saved invariants.csv")