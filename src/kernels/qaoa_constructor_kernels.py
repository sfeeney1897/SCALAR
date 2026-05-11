"""
The Quantum Alternating Operator Ansatz (QAOA) CUDA-Q Constructor Module
-----------------------------------------------------------------------

Author: Sean Feeney
Date: January 2026

This module provides utilities for the construction of fixed-parameter
QAOA circuits using the CUDA-Q kernel builder interface. Unlike the
@cudaq.kernel decorator, which JIT-compiles parameterized quantum kernels
at runtime for execution on quantum coprocessors, this module constructs
fully initialized, static CUDA-Q kernels that are not compiled via the
decorator mechanism.

The primary motivation for this design is interoperability: parameterized
CUDA-Q kernels defined with @cudaq.kernel cannot currently be exported to
OpenQASM 2.0. By explicitly constructing kernels with fixed parameters,
this module enables reliable conversion to OpenQASM representations,
facilitating downstream classical simulation, tensor-network backends,
and cross-platform benchmarking.

In particular, this module is designed to:
  - Construct QAOA circuits with fixed variational parameters
  - Preserve kernel attributes without invoking runtime compilation
  - Enable export to OpenQASM2-compatible circuit descriptions

For comparison, the @cudaq.kernel decorator marks a Python function as a
CUDA-Q kernel to be compiled and executed dynamically on an available
quantum coprocessor. While suitable for hardware execution and hybrid
workflows, such kernels are not compatible with OpenQASM export when
parameterized.

Typical usage example:
"""



import numpy as np
import cudaq
import networkx as nx
from networkx import random_regular_graph
from networkx import random_regular_graph, write_adjlist, read_adjlist
import matplotlib.pyplot as plt
from typing import List
import os
import sys
from utils import test_utils



def build_qaoa_maxcut_kernel(
    qubit_count: int,
    layer_count: int,
    edges_src: list[int],
    edges_tgt: list[int],
    thetas: list[float]
) -> cudaq.Kernel:
    """Build a fixed-parameter QAOA MaxCut kernel suitable for OpenQASM2 export."""

    assert len(thetas) == 2 * layer_count, \
        "thetas must have length 2 * layer_count"
    print("before make_kernel")
    k = cudaq.make_kernel()
    print("before qalloc")
    q = k.qalloc(qubit_count)

    # |+> initialization
    print("before hadamard")
    k.h(q)

    for i in range(layer_count):
        gamma = float(thetas[i])
        beta  = float(thetas[i + layer_count])

        # ----- Problem unitary -----
        print("before phase separator")
        for edge in range(len(edges_src)):
            qubitu = int(edges_src[edge])
            qubitv = int(edges_tgt[edge])
            k.cx(q[qubitu], q[qubitv])
            k.rz(2.0 * gamma, q[qubitv])
            k.cx(q[qubitu], q[qubitv])

        # ----- Mixer -----
        print("before mixer")
        for j in range(qubit_count):
            k.rx(2.0 * beta, q[j])

    # Optional measurements (uncomment if needed by downstream tool)
    # k.mz(q)

    return k
