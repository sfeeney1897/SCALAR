# Experimental: loop-closing via conjecture-derived initialization

This directory holds a branch of work that is **not reported in the current
paper**. It's kept here — rather than deleted — because it's a real,
completed experiment with a genuine (if inconclusive/negative) result, and
because part of it overlaps with a result that *is* reported (see the note
on `qaoa_sim_tnmps.py` below).

## What this experiment was

The main pipeline (`../`) shows that QAOA parameters are often numerically
consistent across graphs sharing a structural fingerprint. The natural next
question: can a conjecture-derived estimate of the optimal parameters be used
to *initialize* the optimizer on a new instance, instead of a uniform random
guess — closing the loop from "conjecture" back into "simulation"?

This was tested with three initialization strategies (`init_strategy` in
`qaoa_sim_1.py` / `qaoa_sim_tnmps.py`): `random` (baseline), `static_conjecture`
(seed from a conjecture-derived formula), and `constrained_sample`.

## The two datasets

- **`known`** — the official 1000-graph set (`../../../data/scaled_topologies/graphs.pkl`),
  same as the main pipeline.
- **`test`** — a *separate*, distinct 1000-graph batch
  (`../../../data/scaled_topologies/experimental/test_graphs.pkl`, different
  seed) generated specifically as a held-out set for this experiment: derive
  a conjecture-based initialization from the `known` set, then test whether
  it transfers to speed up or improve optimization on the `test` set.

These are two genuinely different graph batches, not a train/test split of
the same 1000 — confirmed by their differing topology-model distributions.

## Files

- `qaoa_sim_1.py` — adds `init_strategy` support on top of the main
  pipeline's `qaoa_sim.py`.
- `qaoa_sim_tnmps.py` — the same, but targeting CUDA-Q's tensor-network
  (`tensornet-mps`) backend instead of statevector. **Note:** this file's
  underlying simulator class/kernel is also what the paper's reported
  77-qubit tensor-network demo used — that part is a real, reported result.
  Its `__main__` block here, however (bulk-running `init_strategy` variants
  over the 1000-graph `test` batch), is specific to this dropped experiment;
  the actual 77-qubit demo was a separate, one-off invocation of the
  tensor-network backend on a single large instance, not captured as a
  standalone committed script.
- `conjecture_init.py` — derives a static initialization from
  `results/scaled_topologies/conjectures_parsed.csv` (the stratified
  conjecture output from the `known` set).
- `conjecture_parser.py`, `compare_conjectures.py`, `merge_and_compare.py` —
  parse and compare the `known`-vs-`test` conjecture/result sets.

## Result

Inconclusive / negative: this is the "honest negative result from static
midpoint initialization" mentioned (then removed) from an earlier draft of
the paper's contributions list. It's not written up in the current
manuscript. The result files here (`../../../results/scaled_topologies/experimental/`)
are preserved as-is for anyone who wants to pick this back up.
