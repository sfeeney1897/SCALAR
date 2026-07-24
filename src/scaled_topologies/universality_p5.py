# src/universality_p5.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import ast
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

df = pd.read_csv("../../results/scaled_topologies/run_known_random_p5_merged.csv")

print(f"Total instances: {len(df)}")
print(f"n range: {df['n'].min()} to {df['n'].max()}")
print(f"Models: {df['model'].unique()}")
print(f"Mean approx_ratio: {df['approx_ratio'].mean():.4f}")

# ------------------------------------------------------------------
# Parse full parameter vector [g1,g2,g3,g4,g5, b1,b2,b3,b4,b5]
# ------------------------------------------------------------------
def parse_params(s):
    try:
        return np.array(ast.literal_eval(str(s).replace('\n', ' ')))
    except:
        return np.array([np.nan]*10)

df['params'] = df['optimal_parameters'].apply(parse_params)

p = 5
for i in range(p):
    df[f'gamma{i+1}'] = df['params'].apply(
        lambda x: x[i]   if len(x) >= 2*p else np.nan)
    df[f'beta{i+1}']  = df['params'].apply(
        lambda x: x[p+i] if len(x) >= 2*p else np.nan)

# Verify
gamma1_match = np.allclose(df['gamma1'], df['optimal_gamma'], atol=1e-6)
beta1_match  = np.allclose(df['beta1'],  df['optimal_beta'],  atol=1e-6)
print(f"\nParsing check:")
print(f"  gamma1 matches optimal_gamma: {gamma1_match}")
print(f"  beta1  matches optimal_beta:  {beta1_match}")
for i in range(p):
    print(f"  gamma{i+1} mean: {df[f'gamma{i+1}'].mean():.4f}  "
          f"beta{i+1} mean: {df[f'beta{i+1}'].mean():.4f}")

# ------------------------------------------------------------------
# Fingerprint columns
# ------------------------------------------------------------------
df['deg_mean_r']   = df['deg_mean'].round(3)
df['clust_mean_r'] = df['clust_mean'].round(3)
df['mis_r']        = df['mis'].round(3)
df['deg_std_r']    = df['deg_std'].round(2)
df['assort_r']     = df['assortativity'].round(2)

threshold = 0.01

gamma_cols = [f'gamma{i+1}' for i in range(p)]
beta_cols  = [f'beta{i+1}'  for i in range(p)]
all_cols   = gamma_cols + beta_cols

# ------------------------------------------------------------------
# Extended fingerprint analysis
# ------------------------------------------------------------------
fingerprints = {
    '4-invariant (baseline)': ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r'],
    '5-invariant (+deg_std)': ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r', 'deg_std_r'],
    '5-invariant (+assort)':  ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r', 'assort_r'],
    '6-invariant (+both)':    ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r',
                                'deg_std_r', 'assort_r'],
}

print(f"\n=== EXTENDED FINGERPRINT ANALYSIS (p=5, all 10 parameters) ===")

for name, fp in fingerprints.items():
    agg_dict = {'graph_id': 'count', 'model': lambda x: list(x.unique())}
    for col in all_cols:
        agg_dict[col] = 'std'

    g = df.groupby(fp).agg(
        count  = ('graph_id', 'count'),
        models = ('model', lambda x: list(x.unique())),
        **{f'{c}_std': (c, 'std') for c in all_cols}
    ).reset_index()

    for c in all_cols:
        g[f'{c}_std'] = g[f'{c}_std'].fillna(0)

    rep = g[g['count'] >= 2]

    # All 10 parameters tight
    all_mask = np.ones(len(rep), dtype=bool)
    for c in all_cols:
        all_mask &= (rep[f'{c}_std'] < threshold).values
    uni_all = rep[all_mask]

    # Gammas only tight
    gamma_mask = np.ones(len(rep), dtype=bool)
    for c in gamma_cols:
        gamma_mask &= (rep[f'{c}_std'] < threshold).values
    uni_gamma = rep[gamma_mask]

    # Betas only tight
    beta_mask = np.ones(len(rep), dtype=bool)
    for c in beta_cols:
        beta_mask &= (rep[f'{c}_std'] < threshold).values
    uni_beta = rep[beta_mask]

    # Layer 1 only (gamma1 + beta1)
    uni_layer1 = rep[
        (rep['gamma1_std'] < threshold) &
        (rep['beta1_std']  < threshold)
    ]

    # Same vs cross model
    same  = rep[rep['models'].apply(lambda x: len(x) == 1)]
    cross = rep[rep['models'].apply(lambda x: len(x) > 1)]
    same_uni  = same[np.array([all_mask[rep.index.get_loc(i)]
                                for i in same.index])] \
        if len(same) > 0 else same
    cross_uni = cross[np.array([all_mask[rep.index.get_loc(i)]
                                 for i in cross.index])] \
        if len(cross) > 0 else cross

    rate_all    = len(uni_all)    / len(rep) * 100 if len(rep) > 0 else 0
    rate_gamma  = len(uni_gamma)  / len(rep) * 100 if len(rep) > 0 else 0
    rate_beta   = len(uni_beta)   / len(rep) * 100 if len(rep) > 0 else 0
    rate_layer1 = len(uni_layer1) / len(rep) * 100 if len(rep) > 0 else 0

    print(f"\n{name}:")
    print(f"  Repeated groups:          {len(rep)}")
    print(f"  Universal (all 10):       {len(uni_all):3d} = {rate_all:.1f}%")
    print(f"  Universal (gammas only):  {len(uni_gamma):3d} = {rate_gamma:.1f}%")
    print(f"  Universal (betas only):   {len(uni_beta):3d} = {rate_beta:.1f}%")
    print(f"  Universal (layer 1 only): {len(uni_layer1):3d} = {rate_layer1:.1f}%")
    if len(same) > 0:
        same_all_mask = np.ones(len(same), dtype=bool)
        for c in all_cols:
            same_all_mask &= (same[f'{c}_std'] < threshold).values
        print(f"  Same-model universal:     "
              f"{same_all_mask.sum()}/{len(same)} = "
              f"{same_all_mask.sum()/len(same)*100:.1f}%")
    if len(cross) > 0:
        cross_all_mask = np.ones(len(cross), dtype=bool)
        for c in all_cols:
            cross_all_mask &= (cross[f'{c}_std'] < threshold).values
        print(f"  Cross-model universal:    "
              f"{cross_all_mask.sum()}/{len(cross)} = "
              f"{cross_all_mask.sum()/len(cross)*100:.1f}%")

# ------------------------------------------------------------------
# Summary comparison across all depths
# ------------------------------------------------------------------
print(f"\n=== FULL DEPTH COMPARISON (same-model, all parameters) ===")
print(f"{'Fingerprint':<30} {'p=1':>8} {'p=2':>8} {'p=5':>8}")
print(f"{'─'*54}")
print(f"{'4-invariant (baseline)':<30} {'60.9%':>8} {'16.3%':>8} {'?':>8}")
print(f"{'5-invariant (+deg_std)':<30} {'96.6%':>8} {'27.6%':>8} {'?':>8}")
print(f"{'6-invariant (+both)':<30} {'98.6%':>8} {'39.4%':>8} {'?':>8}")
print(f"(p=5 values filled in after running this script)")