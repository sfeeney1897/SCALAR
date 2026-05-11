import numpy as np
import os
#os.environ["CUDAQ_LOG_LEVEL"] = "debug"
import cudaq
from cudaq import spin
import networkx as nx
from networkx import random_regular_graph
from networkx import random_regular_graph, write_adjlist, read_adjlist, read_multiline_adjlist
import matplotlib.pyplot as plt
from typing import List
import os
from kernels import qaoa_kernels
from utils import test_utils
import pandas as pd
import time


def hamiltonian_max_cut(edges_src, edges_tgt):
        """Hamiltonian for finding the max cut for the graph with given edges and nodes

        Parameters
        ----------
        edges_src: List[int]
            List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
        edges_tgt: List[int]
            List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes

        Returns
        -------
        cudaq.SpinOperator
            Hamiltonian for finding the max cut of the graph with given edges
        """

        hamiltonian = 0

        for edge in range(len(edges_src)):

            qubitu = edges_src[edge]
            qubitv = edges_tgt[edge]
            # Add a term to the Hamiltonian for the edge (u,v)
            hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                                spin.i(qubitu) * spin.i(qubitv))
        #Maybe don't need to return
        return hamiltonian

class qaoa_simulator:
    """
    First off what is the purpose of this code: th epurpose of this code is to handle the simulation of the qaoa maxcut cuda q cirucits
    Based on this what does it need to do:
    - read in the graph (done)
    -produce the interactions (edges) (done)
    - produce the cudaq kernel
    - produce the cuda hamiltonian
    - produce angles
    - run the optimization
    - does this need to spit out the qasm? no
        - This code purely handles the code for dynamic simulation of qaoa (maxcut circuits)
    """
    def __init__(self, instance_name:str, test=False):
        print("Initialize qaoa_simulator")
        self.instance_name = instance_name
        if test:
            print("Initialize test_problem")
            self.test_problem()
            self.simulate_maxcut_qaoa_test()
            #self.set_layer_count()
            #self.set_param_count()
        else:
            print("qaoa_simulator initialzed TRUE")

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
    def test_graph(self):
        pass
    def _get_edges(self):
        print("Getting edges")
        #0 indexing causes failure
        self.edges_src: List[int] = [int(self.edges[i][0]) -1 for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [int(self.edges[i][1]) -1 for i in range(len(self.edges))]
    


    def test_problem(self):
        self.nodes: List[int] = [0, 1, 2, 3, 4]
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
        self.edges_src: List[int] = [self.edges[i][0] for i in range(len(self.edges))]
        self.edges_tgt: List[int] = [self.edges[i][1] for i in range(len(self.edges))]
        # Problem parameters
        # The number of qubits we'll need is the same as the number of vertices in our graph
        self.qubit_count: int = len(self.nodes)

        print("Setting layer count to 2")
        # We can set the layer count to be any positive integer.  Larger values will create deeper circuits
        self.layer_count: int = 2 # this would be read in from a config file

        print("Setting parameter count to 2*layer count")
        # Each layer of the QAOA kernel contains 2 parameters
        self.parameter_count: int = 2 * self.layer_count

    def set_cuda_seed(self, r_seed=13):
        cudaq.set_random_seed(r_seed)

    def set_optimizer(self, opt=cudaq.optimizers.NelderMead()):
        self.optimizer = opt

    def set_numpy_seed(self, r_seed=13):
        np.random.seed(r_seed)

    def set_opt_init_params(self):
        self.initial_gammas = np.random.uniform(0, np.pi, int(self.parameter_count/2))
        self.initial_betas = np.random.uniform(0, np.pi, int(self.parameter_count/2))
        self.initial_parameters = np.append(self.initial_gammas,self.initial_betas)
        self.optimizer.initial_parameters = self.initial_parameters
        print("Initial parameters = ", self.optimizer.initial_parameters)

    def set_target(self, target='qpp-cpu'):
        self.target = target
        #cudaq.set_target('nvidia')
        #cudaq.set_target('qpp-cpu')

    def set_layer_count(self, layer_count: int = 2):
        print(f"Setting layer count: {layer_count}")
        assert type(layer_count) == int
        self.layer_count = layer_count
        self._set_param_count()

    def _set_param_count(self):
        # Each layer of the QAOA kernel contains 2 parameters
        print(f"Setting parameter count: {2 * self.layer_count}")
        self.parameter_count: int = 2 * self.layer_count
    

    def objective_test(self, parameters):
        return cudaq.observe(self.kernel, self.hamiltonian, self.qubit_count, self.layer_count,
                         self.edges_src, self.edges_tgt, parameters).expectation()
    
    def simulate_maxcut_qaoa(self):
        # Optimize!
        self.set_cuda_seed()
        self.set_optimizer()
        self.set_numpy_seed()
        self.set_opt_init_params()
        print("set_target")
        self.set_target()
        print("build kernel")
        self.kernel = qaoa_kernels.kernel_qaoa
        print("build hamiltonian")
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)
        print("running simulation")
        t1 = time.time()
        #ValueError: Invalid NLOpt arguments (e.g. lower bounds are bigger than upper bounds): bounds 0 fail -3.14159 <= 4.88645 <= 3.14159
        self.optimal_expectation, self.optimal_parameters = self.optimizer.optimize(
            dimensions=self.parameter_count, function=self.objective_test)
        t2 = time.time()
        self.sim_time = t2 - t1
        print('optimal_expectation =', self.optimal_expectation)
        print('Therefore, the max cut value is at least ', -1 * self.optimal_expectation)
        print('optimal_parameters =', self.optimal_parameters)



    def simulate_maxcut_qaoa_test(self):
        # Optimize!
        self.set_cuda_seed()
        self.set_optimizer()
        self.set_numpy_seed()
        self.set_opt_init_params()
        self.set_target()
        self.kernel = qaoa_kernels.kernel_qaoa
        self.hamiltonian = hamiltonian_max_cut(self.edges_src, self.edges_tgt)
        t1 = time.time()
        self.optimal_expectation, self.optimal_parameters = self.optimizer.optimize(
            dimensions=self.parameter_count, function=self.objective_test)
        t2 = time.time()
        self.sim_time = t2 - t1
        print('optimal_expectation =', self.optimal_expectation)
        print('Therefore, the max cut value is at least ', -1 * self.optimal_expectation)
        print('optimal_parameters =', self.optimal_parameters)

    def build_dynamic_knowledge_table_row(self) -> dict:
        temp = {"graphname":[self.instance_name],
                "optimal_expectation":[ -1 *self.optimal_expectation],
                "approx_ratio":[ (-1 *self.optimal_expectation)/self.best], 
                "optimal_parameters":[self.optimal_parameters],
                "optimal_gamma":[self.optimal_parameters[0]],
                "optimal_beta":[self.optimal_parameters[1]],
                "initial_parameters":[self.initial_parameters],
                "initial_gamma":[self.initial_gammas[0]],
                "initial_beta":[self.initial_betas[0]],
                "num_layers":[self.layer_count],
                "num_qubits":[self.qubit_count],
                "num_interactions":[len(self.edges)],
                "sim_time":[self.sim_time]}
        return temp
    
    def exact_maxcut(self):
        n = self.g.number_of_nodes()
        nodes = list(self.g.nodes())
        self.best = 0
        for bits in range(1 << (n-1)):  # fix one node to avoid symmetry
            cut = 0
            for u, v, d in self.g.edges(data=True):
                i, j = nodes.index(u), nodes.index(v)
                si = (bits >> i) & 1
                sj = (bits >> j) & 1
                if si != sj:
                    cut += d.get('weight', 1)
            self.best = max(self.best, cut)
        return self.best


if __name__ == '__main__':
    test_utils.set_dir(test_utils.get_path())

    folder = "data/maxcut_test_instances"   # change to your folder
    #We will need to consider additional rounds of the QAOA for this in th near future
    #For preliminary results we only consider single round qaoa
    df = pd.DataFrame( columns=["graphname",
                "optimal_expectation",
                "approx_ratio", 
                "optimal_parameters",
                "optimal_gamma",
                "optimal_beta",
                "initial_parameters",
                "initial_gamma",
                "initial_beta",
                "num_layers",
                "num_qubits",
                "num_interactions",
                "sim_time"])#compile circuit depth etc
    for name in os.listdir(folder):
        if name.endswith(".adj_list"):
            path = os.path.join(folder, name)
            print(f"Testing run for instance {name}")
            #construct the qaoa object
            instance = name.removesuffix(".adj_list")
            qs = qaoa_simulator(instance_name=instance)
            #read in the graph
            qs.read_graph_adjlist(path)
            #num qaoa rounds
            qs.set_layer_count(layer_count=2)
            #simulate
            
            qs.simulate_maxcut_qaoa()
            qs.exact_maxcut()
            #get dynamic features
            #new_row = pd.DataFrame(data=qs.build_dynamic_knowledge_table_row())
            df = pd.concat([df,pd.DataFrame(data=qs.build_dynamic_knowledge_table_row())],ignore_index=True)
    df.to_csv("test_l2.csv", index=False)