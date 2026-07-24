# src/compare_conjectures.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from utils import test_utils
test_utils.set_dir(test_utils.get_path())

# ------------------------------------------------------------------
# Load both conjecture files
# ------------------------------------------------------------------
known_conj = pd.read_csv("../../../results/scaled_topologies/conjectures_parsed.csv")
test_conj  = pd.read_csv("../../../results/scaled_topologies/experimental/conjectures_parsed_test.csv")  
# adjust filename if different

known_conj['source'] = 'known'
test_conj['source']  = 'test'

print(f"Known conjectures: {len(known_conj)} rows")
print(f"Test conjectures:  {len(test_conj)} rows")

# ------------------------------------------------------------------
# Comparison 1: What strata exist in both vs only one?
# ------------------------------------------------------------------
known_strata = set(known_conj['stratum'].unique())
test_strata  = set(test_conj['stratum'].unique())

print("\n=== STRATA COVERAGE ===")
print(f"Strata in both:       {len(known_strata & test_strata)}")
print(f"Only in known:        {len(known_strata - test_strata)}")
print(f"Only in test:         {len(test_strata - known_strata)}")

if known_strata - test_strata:
    print(f"\nStrata only in known: {sorted(known_strata - test_strata)}")
if test_strata - known_strata:
    print(f"Strata only in test:  {sorted(test_strata - known_strata)}")

# ------------------------------------------------------------------
# Comparison 2: For shared strata — do the formulas match?
# ------------------------------------------------------------------
shared_strata = known_strata & test_strata

print("\n=== FORMULA COMPARISON BY STRATUM ===")

match_rows = []
for stratum in sorted(shared_strata):
    for target in ['optimal_gamma', 'approx_ratio', 'mean_n_obj_calls']:
        k = known_conj[
            (known_conj['stratum'] == stratum) &
            (known_conj['target']  == target)
        ]['formula'].tolist()

        t = test_conj[
            (test_conj['stratum'] == stratum) &
            (test_conj['target']  == target)
        ]['formula'].tolist()

        # Exact matches
        k_set = set(k)
        t_set = set(t)
        exact_matches = k_set & t_set
        only_known    = k_set - t_set
        only_test     = t_set - k_set

        match_rows.append({
            'stratum':       stratum,
            'target':        target,
            'known_count':   len(k),
            'test_count':    len(t),
            'exact_matches': len(exact_matches),
            'only_known':    len(only_known),
            'only_test':     len(only_test),
            'match_pct':     len(exact_matches) / max(len(k_set | t_set), 1) * 100,
        })

match_df = pd.DataFrame(match_rows)
print(match_df.to_string(index=False))

# ------------------------------------------------------------------
# Comparison 3: Touch count comparison — which conjectures got tighter?
# ------------------------------------------------------------------
print("\n=== TOUCH COUNT COMPARISON ===")
print("Conjectures with touches > 0 in known:")
print(known_conj[known_conj['touches'] > 0][
    ['stratum','target','formula','touches','support']
].to_string(index=False))

print("\nConjectures with touches > 0 in test:")
print(test_conj[test_conj['touches'] > 0][
    ['stratum','target','formula','touches','support']
].to_string(index=False))

# ------------------------------------------------------------------
# Comparison 4: Structural signal — do key patterns persist?
# ------------------------------------------------------------------
print("\n=== KEY PATTERN CHECKS ===")

# Check if Watts negative deg_mean persists in test
watts_known = known_conj[
    (known_conj['stratum'].str.contains('watts')) &
    (known_conj['target'] == 'optimal_gamma')
]['formula'].tolist()

watts_test = test_conj[
    (test_conj['stratum'].str.contains('watts')) &
    (test_conj['target'] == 'optimal_gamma')
]['formula'].tolist()

watts_neg_known = [f for f in watts_known if '(-1/8) · deg_mean' in f
                   or '(-2/15) · deg_mean' in f]
watts_neg_test  = [f for f in watts_test  if '(-1/8) · deg_mean' in f
                   or '(-2/15) · deg_mean' in f]

print(f"\nWatts negative deg_mean gamma bounds:")
print(f"  Known dataset: {len(watts_neg_known)} instances")
print(f"  Test dataset:  {len(watts_neg_test)} instances")

# Check if 4n+4 universal term persists
obj_known = known_conj[known_conj['target'] == 'mean_n_obj_calls']['formula'].tolist()
obj_test  = test_conj[test_conj['target']  == 'mean_n_obj_calls']['formula'].tolist()

four_n_known = [f for f in obj_known if '4 · n' in f or '4 · m' in f]
four_n_test  = [f for f in obj_test  if '4 · n' in f or '4 · m' in f]

print(f"\nUniversal 4n+4 / 4m+4 complexity bounds:")
print(f"  Known dataset: {len(four_n_known)} instances")
print(f"  Test dataset:  {len(four_n_test)} instances")

# Check regular graph (1/2)*deg_mean touches
reg_known = known_conj[
    (known_conj['stratum'].str.contains('regular')) &
    (known_conj['target'] == 'approx_ratio') &
    (known_conj['touches'] > 0)
]
reg_test = test_conj[
    (test_conj['stratum'].str.contains('regular')) &
    (test_conj['target'] == 'approx_ratio') &
    (test_conj['touches'] > 0)
]

print(f"\nRegular graph tight approx_ratio bounds (touches > 0):")
print(f"  Known: {len(reg_known)} tight conjectures")
for _, r in reg_known.iterrows():
    print(f"    [{r['touches']} touches] {r['formula']}")
print(f"  Test:  {len(reg_test)} tight conjectures")
for _, r in reg_test.iterrows():
    print(f"    [{r['touches']} touches] {r['formula']}")

# ------------------------------------------------------------------
# Save comparison
# ------------------------------------------------------------------
match_df.to_csv("../../../results/scaled_topologies/experimental/conjecture_comparison.csv", index=False)
print("\nSaved → analysis/conjecture_comparison.csv")