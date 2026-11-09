import os
import uuid
import yaml
import json
import networkx as nx
from app.common.logger import logger
from app.topology.custom_topology.convert_gml_to_dfl_multigraph_format import get_multi_network, convert_multigraph_to_dfl_format, topology_to_yaml
from app.topology.custom_topology.utils.utils import get_connectivity_graph, get_ring_overlay, generate_multigraph, convert_id_to_name
from app.topology.custom_topology.constants import TOPOLOGY_NAME, EXPERIMENT, LOCAL_STEPS, ARCH, T_MAX

def generate_network_multigraph(
        root: str = "./results", 
        json_path: str = 'topology_generation_config.json',
        active_nodes: set[str] | None = None,
    ):
    arch = ARCH # "ring"
    t_max = T_MAX # maximum number of edges, ex: t_max = 2
    name = TOPOLOGY_NAME # "amazon_us"
    experiment = EXPERIMENT # "femnist"
    local_steps = LOCAL_STEPS # number of local steps, ex: local_steps = 1
    
    uuid_str = str(uuid.uuid4())
    root = os.path.join(root, uuid_str)
    with open(json_path, 'r') as file:
        topology_generation_config = json.load(file)

    # Size of the model that will be transmitted on the network in bit, ex: model_size = 1e8
    model_size = topology_generation_config["model_size"]
    # Total number of training rounds, ex: num_rounds = 6400
    num_rounds = topology_generation_config["num_rounds"]
    path_to_underlay = topology_generation_config["path_to_underlay"]

    result_dir = os.path.join(root, name, experiment)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    logger.info(f"===== Multigraph: {experiment} dataset - {name} network =====")

    underlay = nx.read_gml(path_to_underlay)
    node_names = [data["name"] for node, data in underlay.nodes(data=True)]
    node_labels = {
        data["name"]: node
        for node, data in underlay.nodes(data=True)
    }

    if active_nodes is not None:
        active_nodes.intersection_update(node_names)
        logger.info(f"Active nodes: {active_nodes}")

        if not active_nodes:
            raise ValueError("No active nodes found in the source GML")

        active_node_names = {
            node_labels[n]
            for n in active_nodes
        }
        logger.info(f"Active node names: {active_node_names}")
        underlay = underlay.subgraph(active_node_names).copy()

    logger.info("Active workers: %d", underlay.number_of_nodes())
    logger.info("Active links: %d", underlay.number_of_edges())

    upload_delay = {}
    download_delay = {}
    computation_time_dict = {}
    for node, data in underlay.nodes(data=True):
        upload_capacity = data["upload"] # upload bandwidth (node) in bits per second
        download_capacity = data["download"] # download bandwidth (node) in bits per second
        computation_time = data["computation"] # Time taken by the model to perform one local update / one training step 
        upload_delay[node] = (model_size / upload_capacity) * 1e3
        download_delay[node] = (model_size / download_capacity) * 1e3
        computation_time_dict[node] = local_steps * computation_time
    nx.set_node_attributes(underlay, upload_delay, 'uploadDelay')
    nx.set_node_attributes(underlay, download_delay, "downloadDelay")
    nx.set_node_attributes(underlay, computation_time_dict, "computationTime")

    connectivity_graph = get_connectivity_graph(underlay)

    if arch == "ring":
        ring = get_ring_overlay(connectivity_graph.copy(), model_size)

        ring_multigraph = generate_multigraph(ring.copy(), model_size, t_max=t_max)

        result_dir_ring = os.path.join(result_dir, "ring")
        if not os.path.exists(result_dir_ring):
            os.makedirs(result_dir_ring)
        
        for i in range(len(ring_multigraph)):
            nx.write_gml(ring_multigraph[i], os.path.join(result_dir_ring, "ring_{}.gml".format(i)))
    
    list_multi_network = get_multi_network(root, name, experiment, arch)
    topology = convert_multigraph_to_dfl_format(list_multi_network, num_rounds)

    yaml_path = os.path.join(root, name, experiment, arch, f'{arch}_num_rounds_{num_rounds}.yaml')
    logger.info(f"yaml_path: {yaml_path}")
    yaml_data = topology_to_yaml(topology)
    with open(yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            sort_keys=True,
            default_flow_style=False
        )

    convert_id_to_name(os.path.join(result_dir_ring, "ring_0.gml"), yaml_path, yaml_path)

    return yaml_path

if __name__ == "__main__":
    yaml_path = generate_network_multigraph()
    logger.info(f"YAML topology path: {yaml_path}")