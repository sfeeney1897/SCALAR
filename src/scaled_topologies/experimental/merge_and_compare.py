# analysis/merge_and_compare.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

# ------------------------------------------------------------------
# Step 1: Load known random baseline — already fully merged
# ------------------------------------------------------------------
exp_random = pd.read_csv("../../../results/scaled_topologies/run_known_random_p5_merged.csv")
exp_random['assortativity'] = exp_random['assortativity'].fillna(0.0)
print(f"Known random baseline: {len(exp_random)} graphs")

# ------------------------------------------------------------------
# Step 2: Combine Experiment A files — stopped halfway, two CSVs
# ------------------------------------------------------------------
conj_a1 = pd.read_csv("../../../results/scaled_topologies/experimental/run_known_conjecture.csv")
conj_a2 = pd.read_csv("../../../results/scaled_topologies/experimental/run_known_conjecture_1.csv")
exp_a   = pd.concat([conj_a1, conj_a2], ignore_index=True)
exp_a   = exp_a.drop_duplicates(subset='graph_id', keep='last')
print(f"Experiment A (known + conjecture): {len(exp_a)} graphs")

# ------------------------------------------------------------------
# Step 3: Load test invariants for B and C approx_ratio calculation
# ------------------------------------------------------------------
test_inv = pd.read_csv("../../../results/scaled_topologies/experimental/test_invariants.csv")
test_inv['assortativity'] = test_inv['assortativity'].fillna(0.0)

# ------------------------------------------------------------------
# Step 4: Load and enrich Experiments B and C
# ------------------------------------------------------------------
exp_b = pd.read_csv("../../../results/scaled_topologies/experimental/run_test_random_B.csv")
exp_c = pd.read_csv("../../../results/scaled_topologies/experimental/run_test_static_conjecture_C.csv")
print(f"Experiment B (test + random):     {len(exp_b)} graphs")
print(f"Experiment C (test + conjecture): {len(exp_c)} graphs")

def add_approx_ratio(df, inv):
    merged = df.merge(
        inv[['graph_id', 'exact_maxcut', 'n', 'm',
             'deg_mean', 'clust_mean', 'mis', 'assortativity']],
        on='graph_id', how='inner'
    )
    if 'model_x' in merged.columns:
        merged = merged.rename(
            columns={'model_x': 'model'}
        ).drop(columns=['model_y'])
    merged['approx_ratio'] = (
        merged['optimal_expectation'].abs() / merged['exact_maxcut']
    ).clip(upper=1.0)
    return merged

# Add approx_ratio to Experiment A (needs inv from known graphs)
inv_known = exp_random[['graph_id', 'exact_maxcut', 'n', 'm',
                         'deg_mean', 'clust_mean', 'mis', 'assortativity']]
exp_a = add_approx_ratio(exp_a, inv_known)

# Add approx_ratio to B and C
exp_b = add_approx_ratio(exp_b, test_inv)
exp_c = add_approx_ratio(exp_c, test_inv)

# ------------------------------------------------------------------
# Step 5: Comparison 1 — Known graphs: random vs conjecture (A)
# ------------------------------------------------------------------
comp_known = exp_random.merge(
    exp_a[['graph_id', 'mean_n_obj_calls', 'approx_ratio',
            'optimal_gamma', 'init_strategy']],
    on='graph_id',
    suffixes=('_random', '_conj')
)

comp_known['call_reduction']     = (
    comp_known['mean_n_obj_calls_random'] -
    comp_known['mean_n_obj_calls_conj']
)
comp_known['call_reduction_pct'] = (
    comp_known['call_reduction'] /
    comp_known['mean_n_obj_calls_random'] * 100
)
comp_known['quality_delta'] = (
    comp_known['approx_ratio_conj'] -
    comp_known['approx_ratio_random']
)

print("\n=== COMPARISON 1: Known Graphs — Random vs Conjecture Init ===")
print(f"Graphs compared: {len(comp_known)}")
print(comp_known[['call_reduction_pct', 'quality_delta']].describe().round(4))

print("\nCall reduction by model:")
print(
    comp_known.groupby('model')['call_reduction_pct']
    .agg(['mean', 'std', 'min', 'max'])
    .round(2)
)

print("\nQuality delta by model:")
print(
    comp_known.groupby('model')['quality_delta']
    .agg(['mean', 'std', 'min', 'max'])
    .round(4)
)

# ------------------------------------------------------------------
# Step 6: Comparison 2 — Test graphs: random (B) vs conjecture (C)
# ------------------------------------------------------------------
comp_test = exp_b.merge(
    exp_c[['graph_id', 'mean_n_obj_calls', 'approx_ratio',
            'optimal_gamma', 'init_strategy']],
    on='graph_id',
    suffixes=('_random', '_conj')
)

comp_test['call_reduction']     = (
    comp_test['mean_n_obj_calls_random'] -
    comp_test['mean_n_obj_calls_conj']
)
comp_test['call_reduction_pct'] = (
    comp_test['call_reduction'] /
    comp_test['mean_n_obj_calls_random'] * 100
)
comp_test['quality_delta'] = (
    comp_test['approx_ratio_conj'] -
    comp_test['approx_ratio_random']
)

print("\n=== COMPARISON 2: Test Graphs — Random (B) vs Conjecture (C) ===")
print(f"Graphs compared: {len(comp_test)}")
print(comp_test[['call_reduction_pct', 'quality_delta']].describe().round(4))

print("\nCall reduction by model:")
print(
    comp_test.groupby('model')['call_reduction_pct']
    .agg(['mean', 'std', 'min', 'max'])
    .round(2)
)

print("\nQuality delta by model:")
print(
    comp_test.groupby('model')['quality_delta']
    .agg(['mean', 'std', 'min', 'max'])
    .round(4)
)

# ------------------------------------------------------------------
# Step 7: Save everything
# ------------------------------------------------------------------
exp_a.to_csv("../../../results/scaled_topologies/experimental/exp_a_known_conjecture_merged.csv", index=False)
exp_b.to_csv("../../../results/scaled_topologies/experimental/exp_b_test_random_merged.csv", index=False)
exp_c.to_csv("../../../results/scaled_topologies/experimental/exp_c_test_conjecture_merged.csv", index=False)
comp_known.to_csv("../../../results/scaled_topologies/experimental/comparison_known.csv", index=False)
comp_test.to_csv("../../../results/scaled_topologies/experimental/comparison_test.csv", index=False)

print("\nAll files saved.")

# ------------------------------------------------------------------
# Step 8: Summary
# ------------------------------------------------------------------
print("\n=== HIGH LEVEL SUMMARY ===")
print(f"Known graphs compared:        {len(comp_known)}")
print(f"  Call reduction:             "
      f"{comp_known['call_reduction_pct'].mean():.1f}% avg")
print(f"  Quality delta:              "
      f"{comp_known['quality_delta'].mean():.4f} avg")
print(f"\nTest graphs compared:         {len(comp_test)}")
print(f"  Call reduction:             "
      f"{comp_test['call_reduction_pct'].mean():.1f}% avg")
print(f"  Quality delta:              "
      f"{comp_test['quality_delta'].mean():.4f} avg")
print(f"\nGeneralization gap:           "
      f"{comp_known['call_reduction_pct'].mean() - comp_test['call_reduction_pct'].mean():.1f}%")
print(f"  (positive = known graphs benefit more than unseen)")