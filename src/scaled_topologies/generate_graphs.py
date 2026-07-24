# Generates the 1000-graph Phase 2 dataset (four topology models, n in [6,20]).
#
# seed=42 (the function default) produces the "known" set used for every
# reported Phase 2 result in the paper (graphs.pkl). seed=420 produces a
# separate, distinct 1000-graph "test"/held-out batch (test_graphs.pkl) used
# only by the experimental/ loop-closing initialization experiment, which is
# not reported in the paper.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
import numpy as np
import pickle
from utils import test_utils

def generate_dataset(n_target=1000, size_range=(6, 20), seed=42):
    rng = np.random.default_rng(seed)
    graphs = []
    attempts = 0
    
    models = ['gnm', 'barabasi', 'watts', 'regular']
    
    while len(graphs) < n_target and attempts < n_target * 10:
        attempts += 1
        n = int(rng.integers(size_range[0], size_range[1] + 1))
        model = rng.choice(models)
        
        try:
            if model == 'gnm':
                max_edges = n * (n-1) // 2
                m = int(rng.integers(n-1, max_edges))
                G = nx.gnm_random_graph(n, m, seed=int(rng.integers(1e6)))
                
            elif model == 'barabasi':
                m = int(rng.integers(1, max(2, n//4)))
                G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1e6)))
                
            elif model == 'watts':
                k = int(rng.choice([2, 4]))
                p = float(rng.uniform(0.1, 0.5))
                G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(1e6)))
                
            elif model == 'regular':
                d = int(rng.choice([2, 3, 4]))
                if (n * d) % 2 != 0:
                    continue
                G = nx.random_regular_graph(d, n, seed=int(rng.integers(1e6)))
        
        except Exception:
            continue
        
        if not nx.is_connected(G):
            continue
            
        # Tag with metadata
        G.graph['model'] = model
        G.graph['graph_id'] = len(graphs)
        graphs.append(G)
        
        if len(graphs) % 100 == 0:
            print(f"Generated {len(graphs)}/{n_target}")
    
    return graphs

if __name__ == "__main__":
    test_utils.set_dir(test_utils.get_path())

    # Official "known" set — this is what every reported Phase 2 result uses.
    graphs = generate_dataset(n_target=1000, seed=42)
    with open("../../data/scaled_topologies/graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    print(f"Saved {len(graphs)} graphs to graphs.pkl")

    # Uncomment to regenerate the separate held-out "test" batch used only by
    # the experimental/ loop-closing initialization experiment (not reported
    # in the paper):
    # test_graphs = generate_dataset(n_target=1000, seed=420)
    # with open("../../data/scaled_topologies/experimental/test_graphs.pkl", "wb") as f:
    #     pickle.dump(test_graphs, f)
    # print(f"Saved {len(test_graphs)} graphs to experimental/test_graphs.pkl")