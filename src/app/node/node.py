import time
import threading
from pathlib import Path

from app.strategy.graphfed import GraphFedStrategy
from app.peer.server import start_peer_server
from app.config.schema import NodeConfig
from app.topology.client import TopologyClient
from app.common.logger import logger
from app.config.loader import load_user_client
from app.client.runner import ClientAdapter
from app.model.store import ModelStore
from app.node.store import node_store, NodeStatus
from app.training.engine import TrainingEngine
from app.membership.manager import MembershipManager
from utils.utils import load_topology


class Node:
    """Manage node lifecycle, membership, peer communication, and training."""

    def __init__(self, cfg: NodeConfig, graph_cfg: str, node_cfg: str):
        self.shutdown_event = threading.Event()
        self.node_id = cfg["node_id"]
        self.peer_address = f'{cfg["peer"]["host"]}:{cfg["peer"]["port"]}'
        self.peer_rounds = cfg["peer"]["rounds"]
        self.delay_time = cfg["flower"]["delay"]
        self.has_topology = cfg["has_topology"]
        user_client = load_user_client(cfg)
        self.ckpt_root = cfg["flower"]["ckpt_root"]
        # Persisted marker: indicates training was active before a process restart.
        self.training_marker = (Path(self.ckpt_root) / f".{self.node_id}.training_active")

        self.flower_client = ClientAdapter(user_client, self.node_id, self.ckpt_root)
        self.model_store = ModelStore()
        logger.info(f"[{self.node_id}] Peer server address: {self.peer_address}")
        logger.info(f"[{self.node_id}] Peer rounds (Flower server round): {self.peer_rounds}")
        logger.info(f"[{self.node_id}] Node config: {cfg}")

        # Register this node with the etcd membership service.
        self.membership = MembershipManager(
            node_id=self.node_id,
            endpoint=self.peer_address,
            etcd_host=cfg["membership"]["etcd_host"],
            etcd_port=cfg["membership"]["etcd_port"],
            namespace=cfg["membership"]["namespace"],
            lease_ttl=cfg["membership"]["lease_ttl"],
        )

        self.topology = None
        self._training_lock = threading.Lock()

        if self.has_topology:
            # Get topology from the topology server.
            self.topology_client = TopologyClient(
                node_id=self.node_id,
                peer_address=self.peer_address,
                server_address=cfg["topology_url"],
                on_topology=self._on_topology,
                on_trigger=self._on_trigger,
            )
            self.num_topology = self.topology_client.get_num_topology()
            self.topology = self.topology_client.list_topologies()
        else:
            # Load topology definitions locally from YAML files.
            self.topology_client = None
            self.topology, _ = load_topology(graph_cfg, node_cfg)
            self.num_topology = len(self.topology)

        logger.info(f"[{self.node_id}] Num topo: {self.num_topology}")

        # Initialize the federated learning strategy and training engine.
        self.strategy = GraphFedStrategy(self.model_store, self.topology, self.num_topology, 
                                        self.node_id, self.peer_rounds, self.membership)
        self.training_engine = TrainingEngine(self.strategy, self.flower_client)
        self.session_start_ts = None
        self.session_end_ts = None

    def _training_was_interrupted(self) -> bool:
        """Check whether a previous training session was interrupted."""
        return self.training_marker.is_file()

    def _mark_training_active(self) -> None:
        """Create a marker before starting a training session."""
        self.training_marker.parent.mkdir(parents=True, exist_ok=True)
        self.training_marker.write_text("active\n", encoding="utf-8")
        logger.info(f"[{self.node_id}] Training marker created: {self.training_marker}")

    def _clear_training_marker(self) -> None:
        """Remove the marker after successful training completion."""
        try:
            self.training_marker.unlink(missing_ok=True)
            logger.info(f"[{self.node_id}] Training marker cleared")
        except OSError:
            logger.exception(f"[{self.node_id}] Failed to clear training marker")

    def start(self) -> None:
        """Start the node and wait for shutdown."""

        logger.info(f"[{self.node_id}] Starting node")
        node_store.set_status(status = NodeStatus.STARTING)

        if self.has_topology:
            self.topology_client.start()
            logger.info(f"[{self.node_id}] Connected to topology server")

            # A previous training session was interrupted. Rejoin automatically without waiting for a new trigger.
            if self._training_was_interrupted():
                logger.warning(f"[{self.node_id}] Previous training was interrupted; rejoining from round 1")
                self._on_trigger()
        else:
            # Start training directly when no topology server is configured.
            self._on_trigger()
            logger.info(f"[{self.node_id}] Do not have the topology server")

        self.shutdown_event.wait()

    def _on_training_finished(self) -> None:
        """Clean up session state after successful training."""
        self.session_end_ts = time.time()
        logger.info(f"Training time: {self.session_end_ts - self.session_start_ts} s")

        try:
            self.membership.stop()
        except Exception:
            logger.exception(f"[{self.node_id}] Failed to stop membership")

        try:
            self.strategy.cleanup()
        except Exception:
            logger.exception(f"[{self.node_id}] Failed to cleanup strategy")

        # Only clear after normal training completion.
        self._clear_training_marker()
        logger.info(f"✅✅✅✅✅[{self.node_id}] Training finished")

    def _on_topology(self, instance_id: str, version: int, topology: dict) -> None:
        """Stage a new snapshot; strategy applies it at an aggregation boundary."""
        self.strategy.stage_topology_update(instance_id, version, topology)
        logger.info(
            f"[{self.node_id}] Received topology "
            f"instance={instance_id}, version={version}"
        )

    def _on_trigger(self, round_id: int = 1) -> None:
        """Start training when a trigger arrives, unless a session is active."""
        if not self._training_lock.acquire(blocking=False):
            logger.warning(f"[{self.node_id}] Training already running")
            return

        threading.Thread(
            target=self._run_training,
            args=(round_id,),
            daemon=True,
        ).start()

    def _start_peer(self) -> None:
        """Start gRPC peer server"""
        self.peer_server = start_peer_server(
            self.strategy,
            self.peer_address,
            self.flower_client, 
            self.model_store
        )
        logger.info(f"[{self.node_id}] Peer server started at {self.peer_address}")

    def _start_training(self) -> None:
        """Run training and release the session lock when finished."""
        try:
            logger.info(f"[{self.node_id}] Start training")
            node_store.set_status(status = NodeStatus.RUNNING)
            self.training_engine.run(self.peer_rounds)

            # Training finished, clear the marker and call the callback.
            self._on_training_finished()
        except Exception:
            # Keep the marker if training fails, allowing a later restart to attempt rejoining from scratch.
            logger.exception(f"[{self.node_id}] Training failed")
        finally:
            # Keep the lock held for the entire training session so a concurrent topology trigger cannot start a second session.
            self._training_lock.release()

    def _start_node(self) -> None:
        """Start the peer server and training in separate threads."""
        threading.Thread(target=self._start_peer, daemon=True).start()
        threading.Thread(target=self._start_training, daemon=True).start()

    def _run_training(self, round_id: int) -> None:
        """Prepare services and launch a training session."""
        try:
            # Persist state before starting any training-related services.
            self._mark_training_active()

            # Start membership before training/topology.
            self.membership.start()

            self.session_start_ts = time.time()
            self._start_node()

            # self._training_lock.release()
        except Exception:
            logger.exception(f"[{self.node_id}] Failed to start training")
            # Keep the marker so a subsequent process restart can retry.
            self._training_lock.release()
