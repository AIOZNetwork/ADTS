import yaml

def get_topology_round(num_topo: int, cur_round: int) -> int:
    """Map a training round to a topology round.

    Reuse topology rounds cyclically when the training round exceeds
    the total number of available topologies.

    Args:
        num_topo: Total number of available topology rounds.
        cur_round: Current training round.

    Returns:
        The corresponding topology round.
    """
    if cur_round <= num_topo:
        return int(cur_round)
    else:
        return int((cur_round - 1) % num_topo + 1)

def load_topology(round_topology_path: str, node_path: str) -> tuple[dict, dict]:
    """Load node configurations and topology definitions, then combine them.

    Args:
        round_topology_path: Path to graph.yaml, which defines the topology
            for each round. Each graph ID = a training round
            and specifies each node's send_ids and recv_ids.
        node_path: Path to nodes.yaml, which contains the configuration
            of all nodes, including each node's weight and address.

    Returns:
        A tuple containing:
            - result: Topology configurations indexed by graph ID and node ID.
              Each node entry contains weight, address, send_ids, and recv_ids.
            - nodes: The original configuration of all nodes loaded from
              nodes.yaml.
    """
    # Load graph.yaml
    with open(round_topology_path, "r") as f:
        graph = yaml.safe_load(f)

    # Load nodes.yaml
    with open(node_path, "r") as f:
        nodes = yaml.safe_load(f)

    result = {}

    for graph_id, graph_data in graph.items():
        result[graph_id] = {}

        for node_id, topo in graph_data.items():
            node_cfg = nodes[node_id]

            result[graph_id][node_id] = {
                "weight": node_cfg["weight"],
                "address": node_cfg["address"],
                "send_ids": topo["send_ids"],
                "recv_ids": topo["recv_ids"],
            }

    return result, nodes