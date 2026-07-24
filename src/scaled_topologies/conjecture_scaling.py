from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.graffiti3 import Graffiti3, print_g3_result, Stage

# Unstratified, whole-1000-graph conjecture generation — feeds the paper's
# "Additional Patterns at Scale" section (obj-calls scaling, approx-ratio
# bounds). Reads run_known_random_p5_merged.csv directly since it already
# contains everything build_full_table.py's merge would produce.

STAGES = [
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

SOPHIE_CFG = dict(
    eq_tol=1e-2,
    min_target_support=5,
    min_h_support=3,
    max_violations=3,   # allow small violations — noisy quantum data
    min_new_coverage=1,
)

if __name__ == '__main__':
    from utils import test_utils
    test_utils.set_dir(test_utils.get_path())
    df = pd.read_csv("../../results/scaled_topologies/run_known_random_p5_merged.csv")

    # --- Run 1: approx_ratio (most impactful) ---
    print("\n=== TARGET: approx_ratio ===")
    df_t1 = df[['approx_ratio', 'deg_mean', 'clust_mean', 'mis', 'n', 'assortativity']].dropna()
    g3 = Graffiti3(df_t1, max_boolean_arity=2,
                   morgan_filter=morgan_filter,
                   dalmatian_filter=dalmatian_filter,
                   sophie_cfg=SOPHIE_CFG)
    g3.conjecture(
        targets=["approx_ratio"],
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

    # --- Run 2: optimal_gamma at scale ---
    print("\n=== TARGET: optimal_gamma ===")
    df_t2 = df[['optimal_gamma', 'optimal_beta', 'n',
                'deg_mean', 'clust_mean', 'mis']].dropna()
    g3 = Graffiti3(df_t2, max_boolean_arity=2,
                   morgan_filter=morgan_filter,
                   dalmatian_filter=dalmatian_filter,
                   sophie_cfg=SOPHIE_CFG)
    g3.conjecture(
        targets=["optimal_gamma"],
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

    # --- Run 3: optimizer complexity (novel) ---
    print("\n=== TARGET: mean_n_obj_calls ===")
    df_t3 = df[['mean_n_obj_calls', 'n', 'm',
                'deg_mean', 'clust_mean', 'mis', 'deg_std']].dropna()
    g3 = Graffiti3(df_t3, max_boolean_arity=2,
                   morgan_filter=morgan_filter,
                   dalmatian_filter=dalmatian_filter,
                   sophie_cfg=SOPHIE_CFG)
    g3.conjecture(
        targets=["mean_n_obj_calls"],
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
    df['C1_slack'] = df['mean_n_obj_calls'] - (4*df['m'] + 4)
    df['C2_slack'] = df['mean_n_obj_calls'] - ((30/17)*df['deg_mean']**2 + 4*df['deg_mean'] + 4)
    df['C3_slack'] = df['mean_n_obj_calls'] - ((22/17)*df['n']**2 + 4*df['n'] + 4)

    print(df[['C1_slack','C2_slack','C3_slack']].describe())
