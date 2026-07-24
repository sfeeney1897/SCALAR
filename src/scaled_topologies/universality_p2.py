# src/universality_p2.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import ast
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

df = pd.read_csv("../../results/scaled_topologies/run_known_random_p2_merged.csv")

print(f"Total instances: {len(df)}")
print(f"n range: {df['n'].min()} to {df['n'].max()}")
print(f"Models: {df['model'].unique()}")
print(f"Mean approx_ratio: {df['approx_ratio'].mean():.4f}")

# ------------------------------------------------------------------
# Parse full parameter vector [gamma1, gamma2, beta1, beta2]
# Confirmed from sample: optimal_gamma = x[0], optimal_beta = x[2]
# ------------------------------------------------------------------
def parse_params(s):
    try:
        return np.array(ast.literal_eval(str(s).replace('\n', ' ')))
    except:
        return np.array([np.nan]*4)

df['params'] = df['optimal_parameters'].apply(parse_params)

df['gamma1'] = df['params'].apply(lambda x: x[0])
df['gamma2'] = df['params'].apply(lambda x: x[1])
df['beta1']  = df['params'].apply(lambda x: x[2])
df['beta2']  = df['params'].apply(lambda x: x[3])

# Verify parsing matches existing columns
gamma1_match = np.allclose(df['gamma1'], df['optimal_gamma'], atol=1e-6)
beta1_match  = np.allclose(df['beta1'],  df['optimal_beta'],  atol=1e-6)
print(f"\nParsing check:")
print(f"  gamma1 matches optimal_gamma: {gamma1_match}")
print(f"  beta1  matches optimal_beta:  {beta1_match}")
print(f"  gamma1 mean: {df['gamma1'].mean():.4f}")
print(f"  gamma2 mean: {df['gamma2'].mean():.4f}")
print(f"  beta1  mean: {df['beta1'].mean():.4f}")
print(f"  beta2  mean: {df['beta2'].mean():.4f}")

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
# Extended fingerprint analysis — all four parameters
# ------------------------------------------------------------------
fingerprints = {
    '4-invariant (baseline)': ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r'],
    '5-invariant (+deg_std)': ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r', 'deg_std_r'],
    '5-invariant (+assort)':  ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r', 'assort_r'],
    '6-invariant (+both)':    ['n', 'deg_mean_r', 'clust_mean_r', 'mis_r',
                                'deg_std_r', 'assort_r'],
}

print(f"\n=== EXTENDED FINGERPRINT ANALYSIS (p=2, all 4 parameters) ===")

for name, fp in fingerprints.items():
    g = df.groupby(fp).agg(
        count      = ('graph_id', 'count'),
        gamma1_std = ('gamma1',   'std'),
        gamma2_std = ('gamma2',   'std'),
        beta1_std  = ('beta1',    'std'),
        beta2_std  = ('beta2',    'std'),
        models     = ('model', lambda x: list(x.unique())),
    ).reset_index()

    for col in ['gamma1_std', 'gamma2_std', 'beta1_std', 'beta2_std']:
        g[col] = g[col].fillna(0)

    rep = g[g['count'] >= 2]

    # All 4 parameters tight
    uni_all = rep[
        (rep['gamma1_std'] < threshold) &
        (rep['gamma2_std'] < threshold) &
        (rep['beta1_std']  < threshold) &
        (rep['beta2_std']  < threshold)
    ]

    # Gammas only tight
    uni_gamma = rep[
        (rep['gamma1_std'] < threshold) &
        (rep['gamma2_std'] < threshold)
    ]

    # Betas only tight
    uni_beta = rep[
        (rep['beta1_std'] < threshold) &
        (rep['beta2_std'] < threshold)
    ]

    rate_all   = len(uni_all)   / len(rep) * 100 if len(rep) > 0 else 0
    rate_gamma = len(uni_gamma) / len(rep) * 100 if len(rep) > 0 else 0
    rate_beta  = len(uni_beta)  / len(rep) * 100 if len(rep) > 0 else 0

    # Same-model vs cross-model
    same  = rep[rep['models'].apply(lambda x: len(x) == 1)]
    cross = rep[rep['models'].apply(lambda x: len(x) > 1)]
    same_uni = same[
        (same['gamma1_std'] < threshold) &
        (same['gamma2_std'] < threshold) &
        (same['beta1_std']  < threshold) &
        (same['beta2_std']  < threshold)
    ]
    cross_uni = cross[
        (cross['gamma1_std'] < threshold) &
        (cross['gamma2_std'] < threshold) &
        (cross['beta1_std']  < threshold) &
        (cross['beta2_std']  < threshold)
    ]

    print(f"\n{name}:")
    print(f"  Repeated groups:         {len(rep)}")
    print(f"  Universal (all 4):       {len(uni_all):3d} = {rate_all:.1f}%")
    print(f"  Universal (gammas only): {len(uni_gamma):3d} = {rate_gamma:.1f}%")
    print(f"  Universal (betas only):  {len(uni_beta):3d} = {rate_beta:.1f}%")
    print(f"  Same-model universal:    {len(same_uni)}/{len(same)}"
          f" = {len(same_uni)/len(same)*100:.1f}%" if len(same) > 0 else "")
    print(f"  Cross-model universal:   {len(cross_uni)}/{len(cross)}"
          f" = {len(cross_uni)/len(cross)*100:.1f}%" if len(cross) > 0 else "")

# ------------------------------------------------------------------
# Summary comparison
# ------------------------------------------------------------------
print(f"\n=== SUMMARY ===")
print(f"p=1 extended fingerprint results (from previous analysis):")
print(f"  4-invariant: 54.0%")
print(f"  5-invariant (+deg_std): 96.9%")
print(f"  6-invariant (+both): 98.8%")
print(f"\np=2 extended fingerprint results (all 4 parameters):")
print(f"  See above")
print(f"\nConclusion: universality at p=2 requires checking all 4 parameters")
print(f"simultaneously — significantly stricter than p=1")