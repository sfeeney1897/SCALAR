# src/universality_p1.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import ast
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

df = pd.read_csv("../../results/scaled_topologies/run_known_random_p1_merged.csv")

print(f"Total instances: {len(df)}")
print(f"n range: {df['n'].min()} to {df['n'].max()}")
print(f"Models: {df['model'].unique()}")
print(f"Mean approx_ratio: {df['approx_ratio'].mean():.4f}")

# ------------------------------------------------------------------
# Parse full parameter vector [gamma1, beta1]
# Confirmed from sample: optimal_gamma = x[0], optimal_beta = x[1]
# ------------------------------------------------------------------
def parse_params(s):
    try:
        return np.array(ast.literal_eval(str(s).replace('\n', ' ')))
    except:
        return np.array([np.nan, np.nan])

df['params'] = df['optimal_parameters'].apply(parse_params)
df['gamma1'] = df['params'].apply(lambda x: x[0])
df['beta1']  = df['params'].apply(lambda x: x[1])

# Verify parsing matches existing columns
gamma1_match = np.allclose(df['gamma1'], df['optimal_gamma'], atol=1e-6)
beta1_match  = np.allclose(df['beta1'],  df['optimal_beta'],  atol=1e-6)
print(f"\nParsing check:")
print(f"  gamma1 matches optimal_gamma: {gamma1_match}")
print(f"  beta1  matches optimal_beta:  {beta1_match}")
print(f"  gamma1 mean: {df['gamma1'].mean():.4f}")
print(f"  beta1  mean: {df['beta1'].mean():.4f}")

# ------------------------------------------------------------------
# Fingerprint columns
# ------------------------------------------------------------------
df['deg_mean_r']   = df['deg_mean'].round(3)
df['clust_mean_r'] = df['clust_mean'].round(3)
df['mis_r']        = df['mis'].round(3)
df['deg_std_r']    = df['deg_std'].round(2)
df['assort_r']     = df['assortativity'].round(2)

threshold = 0.01

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

print(f"\n=== EXTENDED FINGERPRINT ANALYSIS (p=1) ===")

for name, fp in fingerprints.items():
    g = df.groupby(fp).agg(
        count      = ('graph_id', 'count'),
        gamma1_std = ('gamma1',   'std'),
        beta1_std  = ('beta1',    'std'),
        models     = ('model', lambda x: list(x.unique())),
    ).reset_index()

    g['gamma1_std'] = g['gamma1_std'].fillna(0)
    g['beta1_std']  = g['beta1_std'].fillna(0)

    rep = g[g['count'] >= 2]

    # Both tight
    uni_all = rep[
        (rep['gamma1_std'] < threshold) &
        (rep['beta1_std']  < threshold)
    ]

    # Gamma only tight
    uni_gamma = rep[rep['gamma1_std'] < threshold]

    # Beta only tight
    uni_beta = rep[rep['beta1_std'] < threshold]

    # Same-model vs cross-model
    same  = rep[rep['models'].apply(lambda x: len(x) == 1)]
    cross = rep[rep['models'].apply(lambda x: len(x) > 1)]

    same_uni = same[
        (same['gamma1_std'] < threshold) &
        (same['beta1_std']  < threshold)
    ]
    cross_uni = cross[
        (cross['gamma1_std'] < threshold) &
        (cross['beta1_std']  < threshold)
    ]

    rate_all   = len(uni_all)   / len(rep) * 100 if len(rep) > 0 else 0
    rate_gamma = len(uni_gamma) / len(rep) * 100 if len(rep) > 0 else 0
    rate_beta  = len(uni_beta)  / len(rep) * 100 if len(rep) > 0 else 0

    print(f"\n{name}:")
    print(f"  Repeated groups:         {len(rep)}")
    print(f"  Universal (both):        {len(uni_all):3d} = {rate_all:.1f}%")
    print(f"  Universal (gamma only):  {len(uni_gamma):3d} = {rate_gamma:.1f}%")
    print(f"  Universal (beta only):   {len(uni_beta):3d} = {rate_beta:.1f}%")
    if len(same) > 0:
        print(f"  Same-model universal:    "
              f"{len(same_uni)}/{len(same)} = "
              f"{len(same_uni)/len(same)*100:.1f}%")
    if len(cross) > 0:
        print(f"  Cross-model universal:   "
              f"{len(cross_uni)}/{len(cross)} = "
              f"{len(cross_uni)/len(cross)*100:.1f}%")

# ------------------------------------------------------------------
# Show remaining exceptions with 6-invariant fingerprint
# ------------------------------------------------------------------
print(f"\n=== REMAINING EXCEPTIONS WITH 6-INVARIANT FINGERPRINT ===")
fp6 = ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r', 'deg_std_r', 'assort_r']

g6 = df.groupby(fp6).agg(
    count      = ('graph_id', 'count'),
    gamma1_std = ('gamma1',   'std'),
    beta1_std  = ('beta1',    'std'),
    models     = ('model', lambda x: list(x.unique())),
).reset_index()
g6['gamma1_std'] = g6['gamma1_std'].fillna(0)
g6['beta1_std']  = g6['beta1_std'].fillna(0)

rep6 = g6[g6['count'] >= 2]
exc6 = rep6[
    (rep6['gamma1_std'] >= threshold) |
    (rep6['beta1_std']  >= threshold)
]

if len(exc6) > 0:
    print(exc6[['n', 'deg_mean_r', 'clust_mean_r', 'mis_r',
                 'deg_std_r', 'assort_r', 'count',
                 'gamma1_std', 'beta1_std', 'models']
                ].sort_values('gamma1_std',
                              ascending=False).to_string(index=False))
else:
    print("No exceptions — perfect universality with 6-invariant fingerprint")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n=== SUMMARY ===")
print(f"Version 1 — 82 MQLib, p=1: 13/14 = 92.9%")
print(f"Version 2 — 1000 random graphs, p=1:")
print(f"  4-invariant: 54.0% → 5-invariant (+deg_std): 96.9%")
print(f"  Key finding: deg_std resolves Watts-Regular collisions")
print(f"  6-invariant achieves 98.8% universality")