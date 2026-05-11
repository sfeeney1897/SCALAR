from networkx import random_regular_graph, write_adjlist, read_adjlist
import os
import sys
import shutil

def generate_k_reg_graph(num_nodes: int, k: int = 3, seed: int = 1):
    """Generates a k random-regular networkx graph

    Parameters
    ----------
    num_nodes : int
        The number of nodes n of the graph
    k : int
        The degree of each node n in the graph
        (default is 3-regular)
    seed : int
        The random seed used to generate the graph

    Returns
    -------
    g
        A networkx k-random-regular graph
    """

    g = random_regular_graph(k, num_nodes, seed=seed)
    return g

def write_adj_lists(num_nodes: int, dir: str, num_seeds: int = 10) -> None:
    """Writes the k-random-regular graph adjacency list
    to a specified directory

    Parameters
    ----------
    num_nodes : int
        The number of nodes n of the graph
    dir : str
        a string for the directory to write the graph to
    num_seeds : int
        The random seed used to generate the graph
        (default 10)
    """

    # Loop through the number of specified graphs to generate and use a new seed for each graph
    for seed in range(0,num_seeds):
        write_adjlist(generate_k_reg_graph(num_nodes, seed=seed),f"{dir}/g_n={num_nodes}_seed={seed}.adjlist")

def read_adj_lists(file: str):
    """Reads the k-random-regular graph adjacency list
    from a specified file

    Parameters
    ----------
    file : str
        The file of the adjacency list to be read in

    Returns
    -------
    g
        A networkx k-random-regular graph
    """

    #TODO Maybe we want more functionality than just reading the graph, but this can be modularized
    # i.e. g.edges()
    g = read_adjlist(file)
    return g
    

def get_path(dir=True) -> str:
    """Gets and returns the path of the directory or script
    
    If the argument `dir` isn't passed in, the default path
    directory is returned.

    Parameters
    ----------
    dir : bool
        Flag to return either directory or script path 
        (directory path by default)

    Returns
    -------
    directory path
        a string for the directory of the script path

    script path
        a string for the script path
    """

    if dir:
        # Get the directory name from the script path
        return os.path.dirname(os.path.realpath(sys.argv[0]))
    else:
        # sys.argv[0] gives the path used to execute the script
        # os.path.realpath() handles symbolic links
        return os.path.realpath(sys.argv[0])
    
def set_dir(script_dir: str) -> None:
    """Sets the directory to the current script directory
    Returns nothing
    """
    try:
        # Change the current working directory to the script's directory
        os.chdir(script_dir)
        print(f"Current working directory changed to: {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def make_output_dir(directory_path: str) -> None:
    """Make the output directory for data
    Returns nothing
    """
    if os.path.exists(directory_path):
        print(f"Directory '{directory_path}' already exists.")
        pass
    else:
        try:
            os.mkdir(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

def remove_output_dir(directory_path: str) -> None:
    #TODO document
    # Check if the directory exists before attempting to delete
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and all its contents removed successfully.")
        except OSError as e:
            print(f"Error: {directory_path} : {e.strerror}")
    else:
        print(f"Directory '{directory_path}' not found.")