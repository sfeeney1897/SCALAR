# Generates candidate conjectures for optimal_gamma via txGraffiti's
# ConjecturePlayground (convex_hull + ratios methods) over the 82-instance
# Phase 1 knowledge table.
#
# Toggle `features` below to reproduce the two conjecture families reported
# in the paper's Table I:
#   features = ['deg_mean', 'm']          -> C5, C6 (density/edges bounds)
#   features = ['optimal_beta_abs', 'n']  -> C7 (linear upper bound)
from __future__ import annotations

import pandas as pd

from txgraffiti.playground import ConjecturePlayground
from txgraffiti.generators import convex_hull, ratios
from txgraffiti.heuristics import morgan_accept, dalmatian_accept
from txgraffiti.processing import remove_duplicates, sort_by_touch_count

if __name__ == '__main__':

    df = pd.read_csv("../../results/mqlib/knowledge_table_small.csv")

    ai = ConjecturePlayground(
        df,
        object_symbol='G.MaxCut'
    )

    ai.discover(
        methods         = [convex_hull, ratios],
        features        = ['deg_mean', 'm'],  # or ['optimal_beta_abs', 'n'] for C7
        target          = 'optimal_gamma',
        hypothesis      = None,
        heuristics      = [morgan_accept, dalmatian_accept],
        post_processors = [remove_duplicates, sort_by_touch_count],
    )

    for idx, conj in enumerate(ai.conjectures[:10], start=1):
        formula = ai.forall(conj)
        print(f"Conjecture {idx}. {formula}\n")

    for idx, conj in enumerate(ai.conjectures[:10], start=1):
        formula = ai.forall(conj)
        acc = conj.accuracy(df)
        is_true = conj.is_true(df)
        n_counterex = len(conj.counterexamples(df))
        print(f"C{idx}: is_true={is_true}, accuracy={acc:.4f}, counterexamples={n_counterex}")
        print(f"       {formula}\n")
