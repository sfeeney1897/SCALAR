# SCALAR

SCALAR (Symbolic Conjecture and LLM-Assisted Reasoning) is a neurosymbolic framework for automated conjecture generation and reasoning in quantum circuit analysis.

This repository accompanies the paper:

> *SCALAR: A Neurosymbolic Framework for Automated Conjecture and Reasoning in Quantum Circuit Analysis*

The framework integrates:
- CUDA-Q quantum simulation
- symbolic conjecture generation via txGraffiti
- graph-theoretic invariant analysis
- LLM-assisted interpretation and reasoning

## Repository Status

⚠️ The repository is currently undergoing active refactoring and cleanup following the initial arXiv release.  
Code structure, APIs, and experiment pipelines may change substantially over the coming updates.

Additional documentation, reproducibility scripts, and environment setup instructions will be added incrementally.

## Repository Structure

```text
src/        Core framework and simulation code
analysis/   Conjecture analysis and invariant studies
data/       Benchmark and generated graph datasets
```

## Citation

If you use this repository, please cite:

```bibtex
@article{feeney2026scalar,
  title={SCALAR: A Neurosymbolic Framework for Automated Conjecture and Reasoning in Quantum Circuit Analysis},
  author={Feeney, Sean and Rao, Pooja and Klappenecker, Andreas and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

