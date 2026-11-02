import grpc
from concurrent import futures
from typing import NoReturn
import argparse

from app.topology.server import TopologyServer
from app.topology.proto import topology_pb2_grpc
from app.common.logger import logger
from app.common.constants import MAX_TOPOLOGY_WORKER, TOPOLOGY_ADDRESS
from app.topology.custom_topology.generate_network_multigraph import generate_network_multigraph

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the topology server."""
    parser = argparse.ArgumentParser(description="Decentralized FL Node")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "generate"],
        default="",
        help=(
            "Graph mode: "
            "default - uses the input graph directly; "
            "generate - generates a graph from the specified topology (e.g., Gaia).",
        ),
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="",
        help=(
            "Path to the cluster configuration YAML containing"
            "the IP address, port, weight, and other settings for all nodes.",
        ),
    )

    parser.add_argument(
        "--graph-config",
        type=str,
        default="",
        help="Path to the communication graph YAML defining which nodes communicate in each round.",
    )
    parser.add_argument(
        "--graph-generation-config",
        type=str,
        help=(
            "Path to the graph generation configuration YAML containing "
            "parameters such as model size, node capacities, and topology "
            "information (required in generate mode)."
        ),
    )
    parser.add_argument(
        "--etcd-host",
        type=str,
        default="127.0.0.1",
        help="etcd host; defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--etcd-port",
        type=int,
        default="2379",
        help="etcd port; defaults to 2379.",
    )
    args = parser.parse_args()
    return args

def serve() -> NoReturn:
    """Initialize and run the gRPC topology server."""
    args = parse_args()

    if args.mode == "default":
        if not args.graph_config:
            logger.info("--graph_config is required when mode=default")
            return
    elif args.mode == "generate":
        if not args.graph_generation_config:
            logger.info("--graph_generation_config is required when mode=generate")
            return

        # Generate the communication graph from the supplied configuration.
        args.graph_config = generate_network_multigraph(json_path=args.graph_generation_config)

    logger.info(f"args.graph_config: {args.graph_config}")

    # Initialize the topology service with graph, cluster, and membership settings.
    topology_servicer = TopologyServer(
        args.graph_config,
        args.cluster_config,
        etcd_host=args.etcd_host,
        etcd_port=args.etcd_port,
        graph_generation_config=args.graph_generation_config,
    )

    # Create the gRPC server and register the topology service implementation.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_TOPOLOGY_WORKER))
    topology_pb2_grpc.add_TopologyServiceServicer_to_server(topology_servicer, server)

    # Bind the server to the configured address and start accepting requests.
    server.add_insecure_port(TOPOLOGY_ADDRESS)
    server.start()

    try:
        # Start watching etcd membership events to track node joins and leaves.
        topology_servicer.start_membership_watcher()
        logger.info("[Topology] Server started at 0.0.0.0:7000")

        # Keep the process running until the gRPC server is terminated.
        server.wait_for_termination()
    finally:
        # Stop the membership watcher before shutting down the gRPC server.
        topology_servicer.stop_membership_watcher()
        server.stop(grace=5).wait()
        logger.info("[Topology] Server stopped")

if __name__ == "__main__":
    # Start the topology server when this module is executed directly.
    serve()
