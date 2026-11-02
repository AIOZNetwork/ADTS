import os
import cvxpy as cp
import networkx as nx
import numpy as np
from collections import defaultdict
from app.common.logger import logger

def get_optimal_mixing_matrix(adjacency_matrix, method="FDLA"):
    """

    :param adjacency_matrix: np.array()
    :param method:method to construct the mixing matrix weights;
     possible are:
      FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf
      FDLA (Fast Distributed Linear Averaging), https://web.stanford.edu/~boyd/papers/pdf/fastavg.pdf
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    if method == "FDLA":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N)
                       ]
    elif method == "FMMC":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
                       np.zeros((N, N)) <= W
                       ]
    else:
        raise NotImplementedError

    prob = cp.Problem(objective, constraints)
    prob.solve()

    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix

    if method == "FMMC":
        mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)

    # Force symmetry
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :(i)])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix

def get_network(root, network_name, architecture, experiment="", i=""):
    """
    load network  and generate mixing matrix,
    :param network_name: (str) should present in "graph_utils/data"
    :param architecture: possible are: "ring", "complete", "mst", "centralized" and "no_communication"
    :return: nx.DiGraph if architecture is "ring" and nx.DiGraph otherwise
    """
    if (architecture == "no_communication") or (architecture == "matcha") or (architecture == "matcha+"):
        path = os.path.join(root, network_name, experiment, "original.gml")
    else:
        if len(i) == 0:
            path = os.path.join(root, network_name, experiment, "{}.gml".format(architecture))
        else:
            path = os.path.join(root, network_name, experiment, architecture, "{}_{}.gml".format(architecture, i))
    logger.info(f"- Get network from: {path}")
    if architecture == "ring":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(mixing_matrix.shape[0])
        mixing_matrix *= 0.5

        idx_2_node = list(network.nodes)
        if len(i) != 0:
            out_network = nx.from_numpy_array(mixing_matrix, create_using=nx.DiGraph())
            for u, v in out_network.edges:
                if u == v:
                    out_network.add_edge(u, v, edge=1)
                else:
                    out_network.add_edge(u, v, edge=network.edges[idx_2_node[u],idx_2_node[v]]['edge'])
            return out_network
        else:
            return nx.from_numpy_array(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "mct_plus":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        n = network.number_of_nodes()
        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(n)
        mixing_matrix = mixing_matrix / mixing_matrix.sum(axis=0)

        return nx.from_numpy_array(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "complete" or architecture == "matcha":
        network = nx.read_gml(path).to_undirected()

        n = network.number_of_nodes()
        mixing_matrix = np.ones((n, n)) / n

        return nx.from_numpy_array(mixing_matrix)

    elif architecture == "matcha+":
        network = nx.read_gml(path).to_undirected()

        return network

    elif architecture == "matcha+mst":
        path = os.path.join(root, network_name, "mst.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+ring":
        path = os.path.join(root, network_name, "ring.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+delta_mbst":
        path = os.path.join(root, network_name, "mct_congest.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "no_communication":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = np.eye(network.number_of_nodes())

        return nx.from_numpy_array(mixing_matrix)
    else:
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        adjacency_matrix = nx.adjacency_matrix(network, weight=None).todense()
        adjacency_matrix += np.eye(mixing_matrix.shape[0], dtype=np.int64)
        mixing_matrix = get_optimal_mixing_matrix(adjacency_matrix, method="FDLA")

        return nx.from_numpy_array(mixing_matrix)

def get_multi_network(root, network_name, experiment, architecture):
    list_multi_network = []
    list_file = os.listdir(os.path.join(root, network_name, experiment, architecture))
    list_file = [i for i in list_file if "gml" in i]
    for i in range(len(list_file)):
        network = get_network(root, network_name, architecture, experiment, str(i))
        list_multi_network.append(network)
    return list_multi_network

def convert_multigraph_to_dfl_format(list_multi_network, num_rounds):
    results = {}
    list_topology = []
    for idx in range(len(list_multi_network)):
        network = list_multi_network[idx]
        topology = convert_to_topology(network)
        list_topology.append(topology)
    num_silos = len(list_topology[0].keys()) if len(list_topology) >= 1 else None
    if num_silos is not None:
        for idx_round in range(num_rounds):
            idx_topology = idx_round % len(list_topology)
            for idx_silo in range(num_silos):
                if idx_silo not in results.keys():
                    results[idx_silo] = {}
                # DFL framework starts at round idx 1
                results[idx_silo][idx_round + 1] = list_topology[idx_topology][idx_silo]
        return results
    else:
        return None

def convert_to_topology(network):
    """Convert a networkx graph to a topology dictionary.
    
    Args:
        network: A networkx graph object.
    
    Returns:
        A dictionary representing the topology.
        dict[silo_id] = {'send_ids': [...], 'receive_ids': [...]}
    """
    topology = {}
    if isinstance(network, nx.DiGraph):
        for i in network.nodes():
            # Get send_ids and receive_ids
            send_ids = list(network.successors(i))
            receive_ids = list(network.predecessors(i))

            # delete element = i, filter edge == 1  and convert int to string
            send_ids = [str(x) for x in send_ids if (x != i and network.get_edge_data(i, x)['edge'] == 1)]
            receive_ids = [str(x) for x in receive_ids if (x != i and network.get_edge_data(x, i)['edge'] == 1)]

            topology[i] =  {
                'send_ids': send_ids,
                'receive_ids': receive_ids
            }
        return topology
    else:
        raise NotImplementedError("Convert Topology function only support DiGraph!")

def topology_to_yaml(topology):
    yaml_data = defaultdict(dict)

    for node_id, rounds in topology.items():
        node_name = f"node-{node_id}"

        for round_id, comm in rounds.items():
            round_key = str(round_id)  # YAML uses string keys like '1', '2'

            yaml_data[round_key][node_name] = {
                "send_ids": [f"node-{i}" for i in comm.get("send_ids", [])],
                "recv_ids": [f"node-{i}" for i in comm.get("receive_ids", [])],
            }

    return dict(yaml_data)

