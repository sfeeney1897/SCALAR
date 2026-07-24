import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import cudaq
from cudaq import spin
import networkx as nx
from networkx import read_adjlist, read_multiline_adjlist
import pickle
import pandas as pd
import time
import os
from typing import List
from kernels import qaoa_kernels
from utils import test_utils
from conjecture_init import get_init


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
        self.g = nx.convert_node_labels_to_integers(G, first_label=0)
        self.edges = list(self.g.edges)
        self.qubit_count = self.g.number_of_nodes()
        self._get_edges_zero_indexed()

    def _get_edges(self):
        self.edges_src: List[int] = [int(self.edges[i][0]) - 1
                                     for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [int(self.edges[i][1]) - 1
                                     for i in range(len(self.edges))]

    def _get_edges_zero_indexed(self):
        self.edges_src: List[int] = [int(self.edges[i][0])
                                     for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [int(self.edges[i][1])
                                     for i in range(len(self.edges))]

    def test_problem(self):
        self.nodes: List[int] = [0, 1, 2, 3, 4]
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
        self.edges_src: List[int] = [self.edges[i][0]
                                     for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [self.edges[i][1]
                                     for i in range(len(self.edges))]
        self.qubit_count: int = 5
        self.layer_count: int = 2
        self.parameter_count: int = 2 * self.layer_count

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_cuda_seed(self, r_seed=13):
        cudaq.set_random_seed(r_seed)

    def set_optimizer(self, opt=None):
        self.optimizer = (opt if opt is not None
                          else cudaq.optimizers.NelderMead())

    def set_numpy_seed(self, r_seed=13):
        np.random.seed(r_seed)

    def set_opt_init_params(self, init_gamma=None, init_beta=None):
        p = self.parameter_count // 2
        self.initial_gammas = (init_gamma if init_gamma is not None
                               else np.random.uniform(0, np.pi, p))
        self.initial_betas  = (init_beta if init_beta is not None
                               else np.random.uniform(0, np.pi / 2, p))
        self.initial_parameters = np.append(self.initial_gammas,
                                            self.initial_betas)
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
    # Instrumented objective
    # ------------------------------------------------------------------

    def _build_instrumented_objective(self):
        self._call_count    = 0
        self._loss_history  = []
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
        self.kernel      = qaoa_kernels.kernel_qaoa
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)

        objective = self._build_instrumented_objective()

        t1 = time.time()
        self.optimal_expectation, self.optimal_parameters = \
            self.optimizer.optimize(
                dimensions=self.parameter_count, function=objective)
        t2 = time.time()
        self.sim_time = t2 - t1

        print('optimal_expectation =', self.optimal_expectation)
        print('optimal_parameters  =', self.optimal_parameters)
        print(f'n_objective_calls   = {self._call_count}')

    def simulate_maxcut_qaoa_test(self):
        self.simulate_maxcut_qaoa()

    # ------------------------------------------------------------------
    # Multi-restart runner — supports all init strategies
    # ------------------------------------------------------------------

    def run_multi_restart(self, n_restarts: int = 1,
                          init_strategy: str = 'random',
                          graph_model: str = 'unknown'):
        """
        init_strategy: 'random' | 'static_conjecture' | 'constrained_sample'
        graph_model:   passed to get_init for conjecture lookup
        """
        self.kernel      = qaoa_kernels.kernel_qaoa
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)
        self.init_strategy = init_strategy

        # Get list of (gamma, beta) init points
        inits = get_init(
            G        = self.g,
            p        = self.layer_count,
            model    = graph_model,
            strategy = init_strategy,
            n_samples= n_restarts,
        )

        # Pad to n_restarts if needed
        while len(inits) < n_restarts:
            inits.extend(get_init(
                G        = self.g,
                p        = self.layer_count,
                model    = graph_model,
                strategy = init_strategy,
                n_samples= 1,
            ))

        self.all_runs    = []
        best_expectation = np.inf

        for i, (gamma_init, beta_init) in enumerate(inits[:n_restarts]):
            print(f"  [{init_strategy}] Restart {i+1}/{n_restarts}")
            self.set_cuda_seed(r_seed=i)
            self.set_numpy_seed(r_seed=i)
            self.set_optimizer()
            self.set_opt_init_params(
                init_gamma=gamma_init,
                init_beta=beta_init,
            )
            self.set_target()

            objective = self._build_instrumented_objective()

            t1 = time.time()
            exp, params = self.optimizer.optimize(
                dimensions=self.parameter_count, function=objective)
            t2 = time.time()

            self.all_runs.append({
                "restart_idx":       i,
                "init_gamma":        gamma_init,
                "init_beta":         beta_init,
                "final_gamma":       params[:self.layer_count],
                "final_beta":        params[self.layer_count:],
                "final_expectation": exp,
                "n_obj_calls":       self._call_count,
                "sim_time":          t2 - t1,
            })

            if exp < best_expectation:
                best_expectation         = exp
                self.optimal_expectation = exp
                self.optimal_parameters  = params
                self.initial_gammas      = gamma_init
                self.initial_betas       = beta_init
                self.initial_parameters  = np.append(gamma_init, beta_init)
                self.sim_time            = t2 - t1

        final_gammas = [r["final_gamma"][0] for r in self.all_runs]
        final_betas  = [r["final_beta"][0]  for r in self.all_runs]
        self.gamma_std_across_restarts = float(np.std(final_gammas))
        self.beta_std_across_restarts  = float(np.std(final_betas))
        self.mean_n_obj_calls = float(
            np.mean([r["n_obj_calls"] for r in self.all_runs])
        )

    # ------------------------------------------------------------------
    # Knowledge table row builder
    # ------------------------------------------------------------------

    def build_dynamic_knowledge_table_row(self,
                                          graph_source: str = 'known',
                                          init_strategy: str = 'random') -> dict:
        p = self.layer_count
        return {
            "graph_source":               [graph_source],
            "init_strategy":              [init_strategy],
            "graphname":                  [self.instance_name],
            "optimal_expectation":        [-1 * self.optimal_expectation],
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
            "n_restarts":                 [len(self.all_runs)],
            "mean_n_obj_calls":           [self.mean_n_obj_calls],
            "gamma_std_across_restarts":  [self.gamma_std_across_restarts],
            "beta_std_across_restarts":   [self.beta_std_across_restarts],
        }


# ----------------------------------------------------------------------
# Output path helper — never overwrites existing files
# ----------------------------------------------------------------------

def _make_output_path(base_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    candidate = os.path.join(results_dir, f"{base_name}.csv")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(results_dir, f"{base_name}_{i}.csv")
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ----------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------

def run_pipeline(
    pickle_path:   str,
    run_name:      str = "run",
    graph_source:  str = "known",
    init_strategy: str = "random",
    layer_count:   int = 5,
    n_restarts:    int = 1,
    results_dir:   str = "../../../results/scaled_topologies/experimental",
):
    """
    run_name:      output filename e.g. 'run_known_random'
    graph_source:  'known' or 'test'
    init_strategy: 'random' | 'static_conjecture' | 'constrained_sample'
    results_dir:   output directory — never overwrites existing files
    approx_ratio:  NOT computed here — join on graph_id with invariants.csv
    """
    output_csv = _make_output_path(run_name, results_dir)
    print(f"Output will be saved to: {output_csv}")

    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"Loaded {len(graphs)} graphs from {pickle_path}")
    print(f"Strategy: {init_strategy} | "
          f"Source: {graph_source} | "
          f"p={layer_count}")

    header_written = False

    for idx, G in enumerate(graphs):

        graph_id      = G.graph.get("graph_id", idx)
        model         = G.graph.get("model", "unknown")
        instance_name = f"{model}_{graph_id}"

        print(f"\n[{idx+1}/{len(graphs)}] {instance_name} "
              f"| n={G.number_of_nodes()} "
              f"m={G.number_of_edges()} "
              f"| strategy={init_strategy}")

        sim = qaoa_simulator(instance_name=instance_name)
        sim.read_graph_from_networkx(G)
        sim.set_layer_count(layer_count)

        try:
            sim.run_multi_restart(
                n_restarts    = n_restarts,
                init_strategy = init_strategy,
                graph_model   = model,
            )
            row = sim.build_dynamic_knowledge_table_row(
                graph_source  = graph_source,
                init_strategy = init_strategy,
            )
            row["graph_id"] = [graph_id]
            row["model"]    = [model]

            # Write immediately — append mode, crash safe
            pd.DataFrame({k: v for k, v in row.items()}).to_csv(
                output_csv,
                mode   = 'a',
                header = not header_written,
                index  = False,
            )
            header_written = True
            print(f"  Saved → {output_csv}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone → {output_csv}")
    return output_csv


if __name__ == "__main__":
    test_utils.set_dir(test_utils.get_path())

    # --- Experiment A: known graphs + static conjecture init ---
    run_pipeline(
        pickle_path    = "../../../data/scaled_topologies/experimental/test_graphs.pkl",
        run_name      = "run_known_2",
        graph_source  = "test",
        init_strategy = "static_conjecture",
        layer_count   = 1,
        n_restarts    = 1,
        results_dir   = "../../../results/scaled_topologies/experimental",
    )