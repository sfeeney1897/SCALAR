# Merges a depth-specific knowledge_table_p{N}.csv (from qaoa_sim.py) with
# invariants.csv into the final run_known_random_p{N}_merged.csv.
# Set LAYER_COUNT to match whichever depth you just ran qaoa_sim.py for.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

LAYER_COUNT = 2

sim_df = pd.read_csv(f"../../results/scaled_topologies/knowledge_table_p{LAYER_COUNT}.csv")
inv_df = pd.read_csv("../../results/scaled_topologies/invariants.csv")

df = sim_df.merge(inv_df, on="graph_id", how="inner")

if "model_x" in df.columns:
    df = df.rename(columns={"model_x": "model"}).drop(columns=["model_y"])

df["approx_ratio"] = df["optimal_expectation"].abs() / df["exact_maxcut"]
df["approx_ratio"] = df["approx_ratio"].clip(upper=1.0)
df['assortativity'] = df['assortativity'].fillna(0.0)

print(f"Merged rows: {len(df)}")
print(f"approx_ratio mean: {df['approx_ratio'].mean():.4f}")
print(df[["graph_id", "model", "n", "optimal_gamma",
          "optimal_beta", "approx_ratio"]].head())

out_path = f"../../results/scaled_topologies/run_known_random_p{LAYER_COUNT}_merged.csv"
df.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
