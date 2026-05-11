from src.utils import utils as ut
import networkx as nx
import os




if __name__ == "__main__":

    working_directory = ut.get_path()+"/data/mqlib/"
    ut.set_dir(working_directory)

    graph_dir = 'mqlib_instances_unzip_small'
    instance_dir = working_directory+'mqlib_instances_unzip_small/'
    write_dir = working_directory+'/test_instances/'
    iter = 0
    for name in os.listdir(graph_dir):
        #print(name)
        test_instance = instance_dir+name
        with open(test_instance, 'r') as file:
            g = nx.Graph()
            tab = 0
            wg = True
            l = file.readlines()
            for line in l:
                #skip comment lines
                if line.startswith("#"):
                    continue
                #first line not a comment gives us n(nodes) and m(edges)
                if tab == 0:
                    print(line.split())
                    temp = line.split()
                    num_nodes = int(temp[0])
                    num_edges = int(temp[1])
                    print(f'num_nodes: {num_nodes}, num_edges: {num_edges}')
                    tab+=1
                #get edgelist
                else:
                    temp = line.split()
                    if len(temp)>3:
                        print(f"TEMP WAS LONGER THAN 3 ON FILE: {test_instance}....TEMP WAS:{temp}")
                        wg = False
                        break
                    v1 = int(temp[0])
                    #print(f"v1: {v1}")
                    v2 = int(temp[1])
                    #print(f"v2: {v2}")
                    if float(temp[2]) > 1 or float(temp[2]) < 1:
                        wg = False
                        break
                    weight = int(temp[2])
                    

                    #print(f"weight {weight}")
                    g.add_edge(v1, v2, weight=weight)
            if wg:
                iter+=1
                nam_suf = name.removesuffix(".txt")
                #out_dir = write_dir+nam_suf + ".multi_adjlist"
                out_dir = write_dir+nam_suf + ".adj_list"
                nx.write_adjlist(g,out_dir)
                print(f"iter {iter}")
