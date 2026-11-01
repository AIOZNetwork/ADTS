# src/app/main.py

import argparse
import yaml
from pathlib import Path

from app.node.node import Node
from app.node.local_node import LocalNode
from app.common.logger import logger
from app.topology.custom_topology.generate_network_multigraph import generate_network_multigraph

def parse_args() -> argparse.Namespace:
    """Parse command-line options for training and inference modes."""
    parser = argparse.ArgumentParser(
        description="Decentralized FL Node"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Run mode: train or inference."
    )

    parser.add_argument(
        "--weight-path",
        type=str,
        default="",
        help="Path to weight folder or weight file (inference mode).",
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="Path to input folder or input file (inference mode).",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Path to output folder (inference mode).",
    )

    parser.add_argument(
        "--graph-mode",
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
        "--local-node-config",
        type=str,
        help="Path to the local node configuration YAML (e.g., node ID, IP address, port).",
    )

    parser.add_argument(
        "--cluster-config",
        type=str,
        default="",
        help="Path to the cluster configuration YAML containing the IP address, port, weight, and other settings for all nodes.",
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
            "information (required in generate graph-mode)."
        ),
    )
    args = parser.parse_args()
    return args

def main() -> None:
    """Load node configuration and start the selected application mode."""
    args = parse_args()
    # The local configuration provides identity and runtime settings for this node.
    with open(args.local_node_config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "inference":
        local_node = LocalNode(cfg)
        local_node.inference(Path(args.weight_path), Path(args.input_path), Path(args.output_path))
    elif args.mode == "train":
        # Select an existing graph or generate one from the supplied configuration.
        if args.graph_mode == "default":
            if not args.graph_config:
                logger.info("--graph_config is required when graph-mode=default")
                return
        elif args.graph_mode == "generate":
            if not args.graph_generation_config:
                logger.info("--graph_generation_config is required when graph-mode=generate")
                return
            args.graph_config = generate_network_multigraph(json_path=args.graph_generation_config)

        logger.info(f"args.graph_config: {args.graph_config}")

        node = Node(cfg, args.graph_config, args.cluster_config)
        # Start the event-driven node; this call runs for the node's lifetime.
        node.start()


if __name__ == "__main__":
    main()
