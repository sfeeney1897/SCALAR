import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, Stage
import sys
import io

# At the top of your conjecture script, redirect stdout to both terminal and file
class TeeOutput:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file     = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()



STAGES = [
    Stage.RATIO, Stage.LP1, Stage.LP2, Stage.LP3,
    Stage.POLY_SINGLE, Stage.MIXED, Stage.SQRT,
]

SOPHIE_CFG = dict(
    eq_tol=1e-2,
    min_target_support=3,
    min_h_support=2,
    max_violations=2,
    min_new_coverage=1,
)

FEATURES = {
    'optimal_gamma':  ['optimal_gamma', 'deg_mean', 'clust_mean', 'mis', 'n'],
    'approx_ratio':   ['approx_ratio',  'deg_mean', 'clust_mean', 'mis', 'n', 'assortativity'],
    'mean_n_obj_calls': ['mean_n_obj_calls', 'n', 'm', 'deg_mean', 'clust_mean','num_parameters'],
}

def run_graffiti_on_subset(subset, label, min_rows=15):
    if len(subset) < min_rows:
        print(f"  Skipping {label} — only {len(subset)} rows")
        return {}

    results = {}
    for target, cols in FEATURES.items():
        available = [c for c in cols if c in subset.columns]
        df_sub = subset[available].dropna()

        if len(df_sub) < min_rows:
            continue

        print(f"\n--- {label} | target={target} | n={len(df_sub)} ---")
        try:
            g3 = Graffiti3(
                df_sub,
                max_boolean_arity=2,
                morgan_filter=morgan_filter,
                dalmatian_filter=dalmatian_filter,
                sophie_cfg=SOPHIE_CFG,
            )
            result = g3.conjecture(
                targets=[target],
                stages=STAGES,
                include_invariant_products=False,
                include_abs=False,
                include_min_max=False,
                include_log=False,
                enable_sophie=True,
                sophie_stages=STAGES,
                quick=True,
                show=True,
                show_k_conjectures=5,
            )
            results[target] = result
        except Exception as e:
            print(f"  Graffiti failed: {e}")

    return results


if __name__ == '__main__':
    from utils import test_utils
    test_utils.set_dir(test_utils.get_path())
    sys.stdout = TeeOutput("../../results/scaled_topologies/conjecture_output.txt")
    # NOTE: this reads the p=5 merged file — confirmed by reproducing the
    # exact committed conjecture_output.txt formula-for-formula. (The file
    # this script pointed to before migration, exp_b_test_random_merged.csv,
    # did not actually match the committed output — a stale reference left
    # over from later work on the experimental/ loop-closing branch.)
    df = pd.read_csv("../../results/scaled_topologies/run_known_random_p5_merged.csv")
    df['assortativity'] = df['assortativity'].fillna(0.0)

    all_conjectures = {}

    # --- Level 1: By graph model ---
    print("\n" + "="*60)
    print("LEVEL 1: STRATIFIED BY GRAPH MODEL")
    print("="*60)

    for model in df['model'].unique():
        subset = df[df['model'] == model].copy()
        label  = f"model={model}"
        all_conjectures[label] = run_graffiti_on_subset(subset, label)

    # --- Level 2: By n (size buckets) ---
    print("\n" + "="*60)
    print("LEVEL 2: STRATIFIED BY SIZE")
    print("="*60)

    bins   = [0, 10, 15, 20, 25]
    labels = ['small(n≤10)', 'medium(n≤15)', 'large(n≤20)', 'xl(n>20)']
    df['size_bucket'] = pd.cut(df['n'], bins=bins, labels=labels)

    for bucket in df['size_bucket'].cat.categories:
        subset = df[df['size_bucket'] == bucket].copy()
        label  = f"size={bucket}"
        all_conjectures[label] = run_graffiti_on_subset(subset, label)

    # --- Level 3: Model x Size interaction ---
    print("\n" + "="*60)
    print("LEVEL 3: MODEL x SIZE")
    print("="*60)

    for (model, bucket), subset in df.groupby(['model', 'size_bucket']):
        label = f"model={model}_size={bucket}"
        all_conjectures[label] = run_graffiti_on_subset(
            subset.copy(), label, min_rows=10
        )