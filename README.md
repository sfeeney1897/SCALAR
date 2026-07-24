# SCALAR

SCALAR (Symbolic Conjecture and LLM-Assisted Reasoning) is a neurosymbolic framework for automated conjecture generation and reasoning in quantum circuit analysis.

This repository accompanies the paper:

> *SCALAR: A Neurosymbolic Framework for Automated Conjecture and Reasoning in Quantum Circuit Analysis*

The framework integrates:
- CUDA-Q quantum simulation
- symbolic conjecture generation via txGraffiti
- graph-theoretic invariant analysis
- LLM-assisted interpretation and reasoning

## Repository Structure

The paper's evaluation is in two phases: **`mqlib`** (82 filtered MQLib
benchmark instances) and **`scaled_topologies`** (1000 synthetically
generated graphs across four topology models). Both follow the same layout:

```text
data/{mqlib,scaled_topologies}/     Graph instances and generated datasets
src/{mqlib,scaled_topologies}/      Pipeline scripts (see below)
results/{mqlib,scaled_topologies}/  Committed outputs of every pipeline stage
src/utils/, src/kernels/            Shared CUDA-Q kernels and helpers
```

### `scaled_topologies` pipeline order

Each script's docstring notes what it reads and produces; run in this order
to reproduce everything from scratch:

1. `generate_graphs.py` — generates the 1000-graph dataset (`graphs.pkl`)
2. `compute_invariants.py` — computes graph invariants (`invariants.csv`)
3. `qaoa_sim.py` — runs QAOA (set `LAYER_COUNT` to 1, 2, or 5; run once per
   depth) → `knowledge_table_p{N}.csv`
4. `merge.py` — merges sim output with invariants (set `LAYER_COUNT` to
   match) → `run_known_random_p{N}_merged.csv`
5. `universality_p1.py` / `universality_p2.py` / `universality_p5.py` —
   the parameter-consistency analysis behind the paper's Table IV
6. `conjecture_stratified.py` + `conjecture_parser.py` — per-topology/size
   stratified conjecture generation → `conjecture_output.txt`,
   `conjectures_parsed.csv`
7. `conjecture_scaling.py` — whole-dataset (unstratified) conjecture
   generation, reading `run_known_random_p5_merged.csv` directly — behind
   the paper's "Additional Patterns at Scale" section
8. `n14_multirestart_experiment.py`, `experiment_b_llm_ranking.py` —
   two targeted checks run in response to peer review (see the paper's
   revised sections on the n=14 case and the LLM ranking ablation)

`build_full_table.py` is kept for provenance (it documents how an earlier,
now-superseded intermediate merge was done) but isn't required for the above.

One honest gap: `universality_p2_groups.csv` / `universality_p2_repeated.csv`
were produced by an earlier version of `universality_p2.py` that computed a
couple of additional columns (per-group parameter means) the current script
doesn't save. The printed analysis in the current script reproduces the same
underlying rates reported in the paper; these two files are kept as
historical artifacts rather than being silently regenerated with different
columns.

### `mqlib` pipeline

`qaoa_sim.py` runs QAOA on the 82 filtered instances (see the repository
root's `get_unweighted.py` for how they were filtered from MQLib).
`conjecture_playground.py` (toggle its `features` parameter) and
`conjecture_graffiti3.py` generate the candidate conjectures behind the
paper's Table I (C1–C7).

### `experimental/`

Both `src/scaled_topologies/experimental/` and
`data/`/`results/scaled_topologies/experimental/` hold a dropped side
experiment (conjecture-derived optimizer initialization) that is **not
reported in the current paper**. See its own README for what it is and why
it's kept. One file there, `qaoa_sim_tnmps.py`, also underlies the paper's
reported 77-qubit tensor-network demo — see that README for the distinction.

## Reproducibility note

Every script in `src/` was verified against its committed output in
`results/` during the last cleanup pass — in particular, regenerating the
full `scaled_topologies` pipeline from `graphs.pkl` through `qaoa_sim.py`
and `merge.py` reproduces `run_known_random_p1_merged.csv` exactly
(1000/1000 rows matching). Where a script's original committed output
couldn't be reproduced from the script as it stood (a couple of intermediate
scripts had drifted to point at the wrong input file after later work), the
script was corrected to match its actual committed output rather than the
output being silently kept without an explanation.

## Citation

If you use this repository, please cite:

```bibtex
@misc{feeney2026scalarneurosymbolicframeworkautomated,
      title={SCALAR: A Neurosymbolic Framework for Automated Conjecture and Reasoning in Quantum Circuit Analysis}, 
      author={Sean Feeney and Pooja Rao and Andreas Klappenecker and Reuben Tate and Yuri Alexeev and Stefano Mensa and Elica Kyoseva and Stephan Eidenbenz},
      year={2026},
      eprint={2605.10327},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.10327}, 
}
```
