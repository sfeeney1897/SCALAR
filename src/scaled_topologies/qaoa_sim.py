import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cudaq
from cudaq import spin
import networkx as nx
from networkx import read_adjlist, read_multiline_adjlist
import pickle
import pandas as pd
import time
from typing import List
from kernels import qaoa_kernels
from utils import test_utils


def hamiltonian_max_cut(edges_src, edges_tgt):
    hamiltonian = 0
    for edge in range(len(edges_src)):
        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                              spin.i(qubitu) * spin.i(qubitv))
    return hamiltonian


class qaoa_simulator:

    def __init__(self, instance_name: str, test=False):
        print("Initialize qaoa_simulator")
        self.instance_name = instance_name
        if test:
            self.test_problem()
            self.simulate_maxcut_qaoa_test()

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def read_graph_adjlist(self, file):
        self.g = read_adjlist(file)
        self.edges = list(self.g.edges)
        self.qubit_count = self.g.number_of_nodes()
        self._get_edges()

    def read_graph_multiline_adjlist(self, file):
        self.g = read_multiline_adjlist(file)
        self.edges = list(self.g.edges)
        self.qubit_count = self.g.number_of_nodes()
        self._get_edges()

    def read_graph_from_networkx(self, G: nx.Graph):
        """
        Load a NetworkX graph directly — used when iterating over
        the pickle dataset so no file I/O is needed per graph.
        Nodes are re-mapped to 0-indexed integers to keep CUDA-Q happy.
        """
        self.g = nx.convert_node_labels_to_integers(G, first_label=0)
        self.edges = list(self.g.edges)
        self.qubit_count = self.g.number_of_nodes()
        self._get_edges_zero_indexed()

    def _get_edges(self):
        """Original adjlist loader — nodes are 1-indexed strings."""
        self.edges_src: List[int] = [int(self.edges[i][0]) - 1 for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [int(self.edges[i][1]) - 1 for i in range(len(self.edges))]

    def _get_edges_zero_indexed(self):
        """For NetworkX graphs already mapped to 0-indexed integers."""
        self.edges_src: List[int] = [int(self.edges[i][0]) for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [int(self.edges[i][1]) for i in range(len(self.edges))]

    def test_problem(self):
        self.nodes: List[int] = [0, 1, 2, 3, 4]
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
        self.edges_src: List[int] = [self.edges[i][0] for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [self.edges[i][1] for i in range(len(self.edges))]
        self.qubit_count: int = 5
        self.layer_count: int = 2
        self.parameter_count: int = 2 * self.layer_count

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_cuda_seed(self, r_seed=13):
        cudaq.set_random_seed(r_seed)

    def set_optimizer(self, opt=None):
        self.optimizer = opt if opt is not None else cudaq.optimizers.NelderMead()

    def set_numpy_seed(self, r_seed=13):
        np.random.seed(r_seed)

    def set_opt_init_params(self, init_gamma=None, init_beta=None):
        """
        Allow explicit warm-start initialization.
        If not provided, samples uniformly at random.
        """
        p = self.parameter_count // 2
        self.initial_gammas = init_gamma if init_gamma is not None \
            else np.random.uniform(0, np.pi, p)
        self.initial_betas  = init_beta  if init_beta  is not None \
            else np.random.uniform(0, np.pi / 2, p)
        self.initial_parameters = np.append(self.initial_gammas, self.initial_betas)
        self.optimizer.initial_parameters = self.initial_parameters

    def set_target(self, target='qpp-cpu'):
        self.target = target

    def set_layer_count(self, layer_count: int = 2):
        assert isinstance(layer_count, int)
        self.layer_count = layer_count
        self._set_param_count()

    def _set_param_count(self):
        self.parameter_count: int = 2 * self.layer_count

    # ------------------------------------------------------------------
    # Instrumented objective — this is how we get metadata out of CUDA-Q
    # ------------------------------------------------------------------

    def _build_instrumented_objective(self):
        """
        Wraps objective_test with a call counter and loss history logger.
        CUDA-Q's optimizer.optimize() gives us no callback hook, so we
        instrument the objective itself instead.

        If you want richer metadata (gradients, step sizes) swap to scipy:
            from scipy.optimize import minimize
            result = minimize(self._raw_objective, x0, method='COBYLA',
                              callback=lambda xk: self._loss_history.append(...))
        """
        self._call_count  = 0
        self._loss_history = []
        self._param_history = []

        def instrumented(parameters):
            val = cudaq.observe(
                self.kernel, self.hamiltonian,
                self.qubit_count, self.layer_count,
                self.edges_src, self.edges_tgt,
                parameters
            ).expectation()
            self._call_count += 1
            self._loss_history.append(val)
            self._param_history.append(parameters.copy())
            return val

        return instrumented

    # ------------------------------------------------------------------
    # Simulation — single run
    # ------------------------------------------------------------------

    def simulate_maxcut_qaoa(self, init_gamma=None, init_beta=None):
        self.set_cuda_seed()
        self.set_optimizer()
        self.set_numpy_seed()
        self.set_opt_init_params(init_gamma=init_gamma, init_beta=init_beta)
        self.set_target()
        self.kernel     = qaoa_kernels.kernel_qaoa
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)

        objective = self._build_instrumented_objective()

        t1 = time.time()
        self.optimal_expectation, self.optimal_parameters = self.optimizer.optimize(
            dimensions=self.parameter_count, function=objective)
        t2 = time.time()
        self.sim_time = t2 - t1

        print('optimal_expectation =', self.optimal_expectation)
        print('optimal_parameters  =', self.optimal_parameters)
        print(f'n_objective_calls   = {self._call_count}')

    def simulate_maxcut_qaoa_test(self):
        self.simulate_maxcut_qaoa()

    # ------------------------------------------------------------------
    # Multi-restart runner — the main entry point for pipeline use
    # ------------------------------------------------------------------

    def run_multi_restart(self, n_restarts: int = 5):
        """
        Run QAOA n_restarts times with different random initializations.
        Stores all run metadata and sets self.optimal_* to the best run.
        """
        self.kernel      = qaoa_kernels.kernel_qaoa
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)

        self.all_runs = []
        best_expectation = np.inf

        for i in range(n_restarts):
            print(f"  Restart {i+1}/{n_restarts}")
            self.set_cuda_seed(r_seed=i)
            self.set_numpy_seed(r_seed=i)
            self.set_optimizer()
            self.set_opt_init_params()   # fresh random init each restart
            self.set_target()

            objective = self._build_instrumented_objective()

            t1 = time.time()
            exp, params = self.optimizer.optimize(
                dimensions=self.parameter_count, function=objective)
            t2 = time.time()
            self.sim_time = t2 - t1

            run = {
                "restart_idx":       i,
                "init_gamma":        self.initial_gammas,
                "init_beta":         self.initial_betas,
                "final_gamma":       params[:self.layer_count],
                "final_beta":        params[self.layer_count:],
                "final_expectation": exp,
                "n_obj_calls":       self._call_count,      # metadata
                "loss_history":      self._loss_history,     # metadata
                "sim_time":          t2 - t1,
            }
            self.all_runs.append(run)

            if exp < best_expectation:
                best_expectation       = exp
                self.optimal_expectation  = exp
                self.optimal_parameters   = params
                self.initial_gammas_best  = self.initial_gammas.copy()
                self.initial_betas_best   = self.initial_betas.copy()

        # Convergence spread — key universality diagnostic
        final_gammas = [r["final_gamma"][0] for r in self.all_runs]
        final_betas  = [r["final_beta"][0]  for r in self.all_runs]
        self.gamma_std_across_restarts = float(np.std(final_gammas))
        self.beta_std_across_restarts  = float(np.std(final_betas))
        self.mean_n_obj_calls          = float(np.mean([r["n_obj_calls"] for r in self.all_runs]))

    # ------------------------------------------------------------------
    # Knowledge table row builder
    # ------------------------------------------------------------------

    def build_dynamic_knowledge_table_row(self) -> dict:
        p = self.layer_count
        return {
            "graphname":                  [self.instance_name],
            "optimal_expectation":        [-1 * self.optimal_expectation],
            #"approx_ratio":               [(-1 * self.optimal_expectation) / self.best],
            "optimal_parameters":         [self.optimal_parameters],
            "optimal_gamma":              [self.optimal_parameters[0]],
            "optimal_beta":               [self.optimal_parameters[p]],
            "initial_parameters":         [self.initial_parameters],
            "initial_gamma":              [self.initial_gammas[0]],
            "initial_beta":               [self.initial_betas[0]],
            "num_layers":                 [self.layer_count],
            "num_qubits":                 [self.qubit_count],
            "num_interactions":           [len(self.edges)],
            "sim_time":                   [self.sim_time],
            # --- new metadata columns ---
            "n_restarts":                 [len(self.all_runs)],
            "mean_n_obj_calls":           [self.mean_n_obj_calls],
            "gamma_std_across_restarts":  [self.gamma_std_across_restarts],
            "beta_std_across_restarts":   [self.beta_std_across_restarts],
        }

    def exact_maxcut(self):
        n = self.g.number_of_nodes()
        nodes = list(self.g.nodes())
        self.best = 0
        for bits in range(1 << (n - 1)):
            cut = 0
            for u, v, d in self.g.edges(data=True):
                i, j = nodes.index(u), nodes.index(v)
                si = (bits >> i) & 1
                sj = (bits >> j) & 1
                if si != sj:
                    cut += d.get('weight', 1)
            self.best = max(self.best, cut)
        return self.best


# ----------------------------------------------------------------------
# Pipeline runner — reads pickle, runs all graphs, builds knowledge table
# ----------------------------------------------------------------------

def run_pipeline(
    pickle_path: str,
    output_csv:  str  = "knowledge_table.csv",
    layer_count: int  = 1,
    n_restarts:  int  = 5,
    checkpoint_every: int = 25,
):
    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs from {pickle_path}")

    all_rows = []

    for idx, G in enumerate(graphs):
        graph_id   = G.graph.get("graph_id", idx)
        model      = G.graph.get("model", "unknown")
        instance_name = f"{model}_{graph_id}"

        print(f"\n[{idx+1}/{len(graphs)}] {instance_name} | n={G.number_of_nodes()} m={G.number_of_edges()}")

        sim = qaoa_simulator(instance_name=instance_name)
        sim.read_graph_from_networkx(G)
        sim.set_layer_count(layer_count)
        #sim.exact_maxcut()

        try:
            sim.run_multi_restart(n_restarts=n_restarts)
            row = sim.build_dynamic_knowledge_table_row()
            row["graph_id"] = [graph_id]
            row["model"]    = [model]
            all_rows.append(row)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            _flush(all_rows, output_csv)
            print(f"  Checkpoint saved ({idx+1} graphs)")

    _flush(all_rows, output_csv)
    print(f"\nDone. Knowledge table saved to {output_csv}")


def _flush(all_rows, path):
    rows_flat = {}
    for row in all_rows:
        for k, v in row.items():
            rows_flat.setdefault(k, []).extend(v)
    pd.DataFrame(rows_flat).to_csv(path, index=False)


if __name__ == "__main__":
    test_utils.set_dir(test_utils.get_path())

    # Run once per depth (p=1, p=2, p=5) to reproduce all three merged result
    # files via merge.py. Each run writes a depth-specific intermediate
    # knowledge_table_p{N}.csv that merge.py then combines with invariants.csv.
    LAYER_COUNT = 2   # set to 1, 2, or 5 to reproduce that depth

    run_pipeline(
        pickle_path      = "../../data/scaled_topologies/graphs.pkl",
        output_csv       = f"../../results/scaled_topologies/knowledge_table_p{LAYER_COUNT}.csv",
        layer_count      = LAYER_COUNT,
        n_restarts       = 1,
        checkpoint_every = 25,
    )