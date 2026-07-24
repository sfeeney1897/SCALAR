# Builds full_knowledge_table.csv (whole-1000-graph, unstratified) — this was
# originally the input to conjecture_scaling.py. NOT required to reproduce
# the paper's results: conjecture_scaling.py now reads
# run_known_random_p5_merged.csv directly, which already contains everything
# this merge would produce. Kept for provenance/documentation of how that
# merge was originally done. Uses the p=5 knowledge table, confirmed by
# matching the paper's reported obj-calls bound exactly (min slack ~ -0.65 at
# p=5 vs wildly violated at p=1/p=2).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

dynamic_df  = pd.read_csv("../../results/scaled_topologies/knowledge_table_p5.csv")
instance_df = pd.read_csv("../../results/scaled_topologies/invariants.csv")

df = dynamic_df.merge(instance_df, on="graph_id", how="inner")

df["approx_ratio"] = df["optimal_expectation"] / df["exact_maxcut"]

if "model_x" in df.columns:
    df = df.rename(columns={"model_x": "model"}).drop(columns=["model_y"])
df["approx_ratio"] = df["approx_ratio"].clip(upper=1.0)
df['assortativity'] = df['assortativity'].fillna(0.0)
df.to_csv("../../results/scaled_topologies/full_knowledge_table.csv", index=False)
print(df["approx_ratio"].describe())
print(f"{len(df)} rows merged")
