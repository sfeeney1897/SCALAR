# conjecture_init.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.approximation as approx
import re
import os
from utils import test_utils
test_utils.set_dir(test_utils.get_path())
CONJECTURES_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../../../results/scaled_topologies/conjectures_parsed.csv"
)

# ------------------------------------------------------------------
# Safe invariant helpers
# ------------------------------------------------------------------

def _safe_assortativity(G):
    try:
        val = nx.degree_assortativity_coefficient(G)
        return 0.0 if (val is None or np.isnan(val)) else float(val)
    except:
        return 0.0

def get_invariants(G):
    n        = G.number_of_nodes()
    deg_mean = float(np.mean([d for _, d in G.degree()]))
    clust    = float(nx.average_clustering(G))
    mis      = float(len(approx.maximum_independent_set(G)) / n)
    assort   = _safe_assortativity(G)
    m        = G.number_of_edges()
    return {
        'n':        n,
        'deg_mean': deg_mean,
        'clust_mean': clust,
        'mis':      mis,
        'assortativity': assort,
        'm':        m,
    }

# ------------------------------------------------------------------
# Formula parser — converts Graffiti string to evaluatable Python
# ------------------------------------------------------------------

def _clean_formula(formula_str, bound_direction):
    """
    Takes raw formula string from conjectures_parsed.csv.
    Strips the target variable and inequality.
    Returns clean expression string ready for eval().

    Examples:
      '((((56/15) · (mis)²) + (-4 · mis)) + (11/14)) ≤ optimal_gamma'
      → '((((56/15) * (mis)**2) + (-4 * mis)) + (11/14))'

      'optimal_gamma ≤ (22/7)'
      → '(22/7)'
    """
    s = formula_str

    # Strip target variable and inequality from both sides
    s = re.sub(r'optimal_gamma\s*≤\s*', '', s)
    s = re.sub(r'≤\s*optimal_gamma', '', s)
    s = re.sub(r'approx_ratio\s*≤\s*', '', s)
    s = re.sub(r'≤\s*approx_ratio', '', s)
    s = re.sub(r'mean_n_obj_calls\s*≤\s*', '', s)
    s = re.sub(r'≤\s*mean_n_obj_calls', '', s)

    # Replace Graffiti math notation with Python
    s = s.replace('·', '*')        # multiplication dot
    s = s.replace('²', '**2')      # squared
    s = s.replace('³', '**3')      # cubed
    s = s.replace('√', 'sqrt_')    # sqrt — handle below
    s = s.replace('⌊', 'floor_(')  # floor open
    s = s.replace('⌋', ')')        # floor close

    # Handle sqrt(x) → np.sqrt(x)
    s = re.sub(r'sqrt_\(([^)]+)\)', r'np.sqrt(\1)', s)

    # Handle floor_(x) → np.floor(x)
    s = re.sub(r'floor_\(([^)]+)\)', r'np.floor(\1)', s)

    # Clean up variable names to match invariants dict
    s = s.replace('clust_mean', 'inv["clust_mean"]')
    s = s.replace('deg_mean',   'inv["deg_mean"]')
    s = s.replace('mis',        'inv["mis"]')
    s = s.replace('assortativity', 'inv["assortativity"]')
    s = s.replace('sqrt(n)',    'np.sqrt(inv["n"])')
    s = re.sub(r'\bn\b',        'inv["n"]', s)
    s = re.sub(r'\bm\b',        'inv["m"]', s)

    return s.strip()


def _evaluate_formula(formula_str, inv):
    """
    Evaluates a cleaned formula string against invariant dict.
    Returns float or None if evaluation fails.
    """
    try:
        result = eval(formula_str, {"np": np, "inv": inv})
        return float(result)
    except Exception as e:
        return None


def _parse_bound_direction(formula_str, target):
    """
    Returns 'lower' if formula ≤ target (formula is lower bound on target)
    Returns 'upper' if target ≤ formula (formula is upper bound on target)
    """
    # 'X ≤ target' means X is lower bound
    if re.search(rf'≤\s*{target}', formula_str):
        return 'lower'
    # 'target ≤ X' means X is upper bound
    elif re.search(rf'{target}\s*≤', formula_str):
        return 'upper'
    return None

# ------------------------------------------------------------------
# Bound lookup — reads conjectures_parsed.csv dynamically
# ------------------------------------------------------------------

_conjecture_df = None  # module-level cache

def _load_conjectures():
    global _conjecture_df
    if _conjecture_df is None:
        _conjecture_df = pd.read_csv(CONJECTURES_PATH)
    return _conjecture_df


def get_conjecture_bounds(G, model, target='optimal_gamma'):
    """
    Looks up conjecture bounds for this graph's stratum from CSV.
    Returns (lower, upper) for the target variable.
    Falls back to (0, pi) if no bounds found.
    """
    inv = get_invariants(G)
    n   = inv['n']

    # Determine size bucket
    if n <= 10:
        size = 'small(n≤10)'
    elif n <= 15:
        size = 'medium(n≤15)'
    else:
        size = 'large(n≤20)'

    df = _load_conjectures()

    # Most specific first
    stratum_candidates = [
        f"model={model}_size={size}",
        f"model={model}",
        f"size={size}",
    ]

    lower_vals = []
    upper_vals = []

    for stratum in stratum_candidates:
        subset = df[
            (df['stratum'] == stratum) &
            (df['target']  == target)
        ]

        if subset.empty:
            continue

        for _, row in subset.iterrows():
            formula_str = row['formula']
            direction   = _parse_bound_direction(formula_str, target)

            if direction is None:
                continue

            cleaned = _clean_formula(formula_str, direction)
            value   = _evaluate_formula(cleaned, inv)

            if value is None or np.isnan(value) or np.isinf(value):
                continue

            if direction == 'lower':
                lower_vals.append(value)
            elif direction == 'upper':
                upper_vals.append(value)

        # Only stop if we found BOTH bounds at this stratum level
        if lower_vals and upper_vals:
            break

    # Take tightest valid bounds
    lower = max(lower_vals) if lower_vals else 0.0
    upper = min(upper_vals) if upper_vals else float(np.pi)

    # Sanity check
    if upper <= lower:
        upper = lower + 0.5

    lower = float(np.clip(lower, 0.0, np.pi))
    upper = float(np.clip(upper, lower + 0.05, np.pi))

    return lower, upper

# ------------------------------------------------------------------
# Strategy 1: Pure random baseline
# ------------------------------------------------------------------

def init_random(p):
    gamma = np.random.uniform(0, np.pi, p)
    beta  = np.random.uniform(0, np.pi/2, p)
    return gamma, beta


# ------------------------------------------------------------------
# Strategy 2: Static conjecture — single midpoint init
# ------------------------------------------------------------------

def init_static_conjecture(G, p, model, target='optimal_gamma'):
    lower, upper = get_conjecture_bounds(G, model, target)
    mid   = (lower + upper) / 2.0
    gamma = np.full(p, np.clip(mid, 0, np.pi))
    beta  = np.random.uniform(0, np.pi/2, p)
    return gamma, beta


# ------------------------------------------------------------------
# Strategy 3: Constrained sampling — n_samples within bounds
# ------------------------------------------------------------------

def init_constrained_sample(G, p, model, n_samples=5, target='optimal_gamma'):
    lower, upper = get_conjecture_bounds(G, model, target)
    results = []
    for _ in range(n_samples):
        gamma = np.random.uniform(lower, upper, p)
        beta  = np.random.uniform(0, np.pi/2, p)
        results.append((gamma, beta))
    return results


# ------------------------------------------------------------------
# Dispatcher — single entry point for run_pipeline
# ------------------------------------------------------------------

def get_init(G, p, model, strategy='random', n_samples=5):
    """
    Returns list of (gamma_array, beta_array) tuples.
    Always a list so pipeline loop is identical regardless of strategy.

    strategy: 'random' | 'static_conjecture' | 'constrained_sample'
    """
    if strategy == 'random':
        return [init_random(p)]

    elif strategy == 'static_conjecture':
        return [init_static_conjecture(G, p, model)]

    elif strategy == 'constrained_sample':
        return init_constrained_sample(G, p, model, n_samples=n_samples)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose from: random, static_conjecture, "
                         f"constrained_sample")


# ------------------------------------------------------------------
# Diagnostic — useful for debugging init ranges
# ------------------------------------------------------------------

def describe_bounds(G, model, target='optimal_gamma'):
    inv           = get_invariants(G)
    lower, upper  = get_conjecture_bounds(G, model, target)
    mid           = (lower + upper) / 2.0
    print(f"Model: {model} | n={inv['n']} | "
          f"deg_mean={inv['deg_mean']:.3f} | "
          f"mis={inv['mis']:.3f}")
    print(f"Conjecture bounds for {target}: "
          f"[{lower:.4f}, {upper:.4f}] → mid={mid:.4f}")
    print(f"vs random range: [0, {np.pi:.4f}]")
    print(f"Range reduction: "
          f"{((upper-lower)/np.pi*100):.1f}% of full range")
    
