"""
Focused multi-restart check on the n=14 exception (g000888, g002942).

Reviewer question: is the reported two-basin behavior at n=14 a real
distinct-optima phenomenon, or an artifact of running only n_restarts=1 in
the originally reported results? This reruns both graphs with many restarts
at p=1 and p=2 using the existing run_multi_restart infrastructure, and
writes per-restart rows to a new CSV without touching any existing result
file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from qaoa_sim import qaoa_simulator

GRAPHS = {
    "g000888": "../../github/data/mqlib/test_instances/g000888.adj_list",
    "g002942": "../../github/data/mqlib/test_instances/g002942.adj_list",
}
N_RESTARTS = 40
DEPTHS = [1, 2]

rows = []
for name, path in GRAPHS.items():
    for p in DEPTHS:
        print(f"\n=== {name}  p={p} ===")
        sim = qaoa_simulator(instance_name=name)
        sim.read_graph_adjlist(path)
        sim.set_layer_count(p)
        sim.run_multi_restart(n_restarts=N_RESTARTS)

        for run in sim.all_runs:
            rows.append({
                "graphname": name,
                "p": p,
                "restart_idx": run["restart_idx"],
                "gamma1": run["final_gamma"][0],
                "beta1": run["final_beta"][0],
                "final_expectation": run["final_expectation"],
                "n_obj_calls": run["n_obj_calls"],
            })

        exps = [r["final_expectation"] for r in sim.all_runs]
        g1s  = [r["final_gamma"][0] for r in sim.all_runs]
        b1s  = [r["final_beta"][0] for r in sim.all_runs]
        print(f"  best expectation: {min(exps):.6f}  (worst: {max(exps):.6f})")
        print(f"  gamma1: mean={np.mean(g1s):.4f} std={np.std(g1s):.4f} "
              f"range=[{min(g1s):.4f},{max(g1s):.4f}]")
        print(f"  beta1:  mean={np.mean(b1s):.4f} std={np.std(b1s):.4f} "
              f"range=[{min(b1s):.4f},{max(b1s):.4f}]")

df = pd.DataFrame(rows)
out_path = "../results/n14_multirestart_experiment.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows -> {out_path}")
