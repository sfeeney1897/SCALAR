"""
Experiment B: LLM-vs-deterministic conjecture ranking check.

Mirrors the exact working invocation already validated in
test2/sym_conj copy.py (same feature pairs, same methods, same 82-instance
Phase 1 data) rather than exploring new feature combinations — txGraffiti's
search space explodes with more invariants at once, per prior experience.

For each of the two feature sets used in the original Phase 1 exploration
(['optimal_beta_abs','n'] and ['deg_mean','m'], target='optimal_gamma'), this:
  1. Regenerates the candidate conjectures fresh (read-only on
     test2/knowledge_table_small.csv, no existing files touched).
  2. Computes slack for each conjecture deterministically against the data.
  3. Reports the deterministic order already produced by
     sort_by_touch_count (txGraffiti's own post-processor).
Ranking commentary/LLM-style judgment is done separately, informed by this
script's output, not baked into the script itself.
"""
import re
import pandas as pd
import numpy as np

from txgraffiti.playground import ConjecturePlayground
from txgraffiti.generators  import convex_hull, ratios
from txgraffiti.heuristics  import morgan_accept, dalmatian_accept
from txgraffiti.processing  import remove_duplicates, sort_by_touch_count

df = pd.read_csv("../../test2/knowledge_table_small.csv")
print(f"Loaded {len(df)} instances")

FEATURE_SETS = [
    ['optimal_beta_abs', 'n'],
    ['deg_mean', 'm'],
]

def formula_to_expr(formula: str) -> str:
    """Turn the printed txGraffiti formula into a Python-evaluable expression
    string, splitting off the comparison operator."""
    f = formula.replace('·', '*')
    for op in ['<=', '>=']:
        if op in f:
            lhs, rhs = f.split(op)
            return lhs.strip(), op, rhs.strip()
    raise ValueError(f"no comparator found in: {formula}")

results = []
for features in FEATURE_SETS:
    print(f"\n=== target=optimal_gamma  features={features} ===")
    ai = ConjecturePlayground(df, object_symbol='G.MaxCut')
    ai.discover(
        methods         = [convex_hull, ratios],
        features        = features,
        target          = 'optimal_gamma',
        hypothesis      = None,
        heuristics      = [morgan_accept, dalmatian_accept],
        post_processors = [remove_duplicates, sort_by_touch_count],
    )

    for order, conj in enumerate(ai.conjectures[:10], start=1):
        formula = str(ai.forall(conj))
        acc = conj.accuracy(df)
        is_true = conj.is_true(df)
        n_counterex = len(conj.counterexamples(df))

        # strip the "forall G.MaxCut: (True) -> (...)" wrapper to get the bare inequality
        m = re.search(r'\(True\)\s*(?:->|→)\s*\((.*)\)\s*$', formula)
        bare = m.group(1) if m else formula
        try:
            lhs_str, op, rhs_str = formula_to_expr(bare)
            local = {c: df[c].values for c in features + ['optimal_gamma']}
            lhs_val = eval(lhs_str, {"__builtins__": {}}, local)
            rhs_val = eval(rhs_str, {"__builtins__": {}}, local)
            lhs_val = np.broadcast_to(np.asarray(lhs_val, dtype=float), (len(df),))
            rhs_val = np.broadcast_to(np.asarray(rhs_val, dtype=float), (len(df),))
            slack = (rhs_val - lhs_val) if op == '<=' else (lhs_val - rhs_val)
            mean_slack, min_slack = float(np.mean(slack)), float(np.min(slack))
        except Exception as e:
            mean_slack, min_slack = float('nan'), float('nan')
            print(f"  [slack computation failed for C{order}: {e}]")

        n_terms = bare.count('*') + bare.count('+') + 1  # rough complexity proxy

        results.append({
            "features": "+".join(features),
            "det_rank": order,
            "formula": bare,
            "is_true": is_true,
            "accuracy": acc,
            "n_counterexamples": n_counterex,
            "mean_slack": mean_slack,
            "min_slack": min_slack,
            "n_terms": n_terms,
        })
        print(f"  C{order} (det_rank={order}): acc={acc:.4f} counterex={n_counterex} "
              f"mean_slack={mean_slack:.4f} min_slack={min_slack:.4f} terms~{n_terms}")
        print(f"       {bare}")

out = pd.DataFrame(results)
out_path = "../analysis/experiment_b_llm_ranking.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out)} rows -> {out_path}")
