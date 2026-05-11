# class object that initilaizes the symbolic conjecture
#input: KNowledge table with properties or invariants
# Output COnjectures in Symbolic form
from __future__ import annotations

import pandas as pd

from txgraffiti.playground    import ConjecturePlayground
from txgraffiti.generators    import convex_hull, ratios
from txgraffiti.heuristics    import morgan_accept, dalmatian_accept
from txgraffiti.processing    import remove_duplicates, sort_by_touch_count

if __name__ == '__main__':

    df = pd.read_csv("/home/sfeen/projects/symbolic_ai/test2/knowledge_table_small.csv")



    ai = ConjecturePlayground(
        df,
        object_symbol='G.MaxCut'
    )

    ai.discover(
        methods         = [convex_hull, ratios],
        features        = ['deg_mean', 'm'],#['optimal_beta_abs', 'n'],
        target          = 'optimal_gamma',
        hypothesis      = None,  # no boolean conditions for now
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
"""    for idx, conj in enumerate(ai.conjectures[:10], start=1):
        # evaluate conjecture as boolean series, but we need the slack
        # call the conjecture to get the boolean mask, then extract conclusion values
        actual = df['optimal_gamma']
        
        # re-evaluate RHS by calling conclusion on df
        conclusion_vals = conj.conclusion(df)  # try this
        print(f"C{idx}: {type(conclusion_vals)}")
        conj_result = conj(df)  # callable from Predicate
        print(type(conj_result), conj_result[:3])
        break    """
    


