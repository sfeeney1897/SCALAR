# class object that initilaizes the symbolic conjecture
#input: KNowledge table with properties or invariants
# Output COnjectures in Symbolic form
from __future__ import annotations

import pandas as pd

from txgraffiti.graffiti3.heuristics.morgan import morgan_filter#, dalmatian_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, print_g3_result, Stage

if __name__ == '__main__':

    df = pd.read_csv("/home/sfeen/projects/symbolic_ai/test2/knowledge_table_small.csv")
    df2 = df[['optimal_beta_abs','optimal_gamma','chromatic']]
    


    g3 = Graffiti3(
        df2,
        max_boolean_arity=2,
        morgan_filter=morgan_filter,
        dalmatian_filter=dalmatian_filter,
        sophie_cfg=dict(
            eq_tol=1e-2,
            min_target_support=5,
            min_h_support=3,
            max_violations=0,
            min_new_coverage=1,
        ),
    )

    STAGES = [

        # Stage.CONSTANT,
        Stage.RATIO,
        Stage.LP1,
        Stage.LP2,
        Stage.LP3,
        Stage.LP4,
        Stage.POLY_SINGLE,
        Stage.MIXED,
        Stage.SQRT,
        Stage.LOG,
        Stage.SQRT_LOG,
        Stage.GEOM_MEAN,
        Stage.LOG_SUM,
        Stage.SQRT_PAIR,
        Stage.SQRT_SUM,
        Stage.EXP_EXPONENT,

    ]

    TARGETS = [
    #"approx_ratio",
    "optimal_gamma",
    #"optimal_beta",
    ]


    result = g3.conjecture(
        targets=TARGETS,
        stages=STAGES,
        include_invariant_products=False,
        include_abs=False,
        include_min_max=False,
        include_log=False,
        enable_sophie=True,
        sophie_stages=STAGES,
        quick=True,
        show=True,
        show_k_conjectures=10,
    )
    