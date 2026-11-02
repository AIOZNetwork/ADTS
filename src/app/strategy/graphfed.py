# app/strategy.py
import time
import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.strategy.aggregate import aggregate

from threading import Event, Lock
from typing import Dict, Tuple, Optional

from app.peer.client import push_model
from app.common.logger import logger
from app.model.store import ModelStore
from app.node.store import node_store
from utils.utils import get_topology_round
from app.membership.manager import MembershipManager
from app.common.constants import PUSH_MODEL_TIMEOUT, RECEIVE_MODEL_TIMEOUT


class GraphFedStrategy:
    """Manage topology-based model exchange and aggregation between DFL nodes."""

    def __init__(self, model_store: ModelStore, topology: dict, num_topology: int, 
                node_id: str, num_rounds: int, membership: MembershipManager,
                alpha: float=0.6, receive_model_timeout: float = RECEIVE_MODEL_TIMEOUT, num_peers: int=10, 
                push_model_timeout: float = PUSH_MODEL_TIMEOUT, **kwargs):
        self.node_id = node_id
        self.alpha = alpha
        self.num_topology = num_topology
        self.topology = topology
        self.topology_cache = {}
        self.num_peers = num_peers
        self.push_model_timeout = push_model_timeout
        self.membership = membership
        self.training_done_cb = None
        self.num_rounds = num_rounds

        # Store models received from neighboring nodes.
        self.peer_models = {}
        self.send_neighbors = {}
        self.recv_neighbors = []
        self.current_round = 0
        self.receive_model_timeout = receive_model_timeout
        self.lock = Lock()
        self.ready_event = Event()
        self.peer_rnds = []
        self.model_store = model_store

        # Track which peers each round is waiting for and has received.
        self._round_waiting_peers: Dict[int, set[str]] = {}
        self._round_received_peers: Dict[int, set[str]] = {}

        # Protect topology updates separately from model aggregation.
        self._topology_update_lock = Lock()
        self.topology_instance_id: Optional[str] = None
        self.topology_version = 0
        self.pending_topology: Optional[Tuple[str, int, dict]] = None

        # Register callback so membership changes can wake active rounds.
        if self.membership is not None:
            self.membership.add_membership_listener(
                self._on_membership_change
            )

    def stage_topology_update(
        self,
        instance_id: str,
        version: int,
        topology: dict,
    ) -> None:
        """Store the newest snapshot; apply it at the next aggregation boundary."""
        with self._topology_update_lock:
            pending_instance_id, pending_version = (
                (self.pending_topology[0], self.pending_topology[1])
                if self.pending_topology is not None
                else (self.topology_instance_id, self.topology_version)
            )

            # Versions are ordered only within one server instance.
            if instance_id == pending_instance_id and version <= pending_version:
                logger.info(f"[{self.node_id}] Ignore stale topology version={version}; current/pending={pending_version}")
                return

            self.pending_topology = (instance_id, version, topology)

        logger.info(
            f"[{self.node_id}] Staged topology instance={instance_id}, "
            f"version={version}; will apply at next aggregation boundary"
        )

    def _apply_pending_topology_update(self) -> None:
        """Apply a pending topology snapshot and invalidate the topology cache."""
        with self._topology_update_lock:
            if self.pending_topology is None:
                return

            instance_id, version, topology = self.pending_topology
            self.topology = topology
            self.num_topology = len(topology)
            self.topology_cache.clear()
            self.topology_instance_id = instance_id
            self.topology_version = version
            self.pending_topology = None

        logger.info(
            f"[{self.node_id}] Applied topology instance={instance_id}, "
            f"version={version}; num_topology={self.num_topology}"
        )

    def _on_membership_change(
        self,
        event: str,
        node_id: str,
        present_nodes: set[str],
    ) -> None:
        """Update active round wait conditions when a peer leaves."""
        logger.info(f"[{self.node_id}] Membership change event: {event} for node {node_id}")
        logger.info(f"round_waiting_peers: {self._round_waiting_peers}")

        # Only a peer leaving can reduce the set of expected model updates.
        if event != "LEAVE":
            return

        with self.lock:
            for rnd, waiting_peers in self._round_waiting_peers.items():
                if node_id not in waiting_peers:
                    continue

                waiting_peers.discard(node_id)
                received_peers = self._round_received_peers.get(rnd, set())

                logger.info(
                    f"[{self.node_id}] Peer left during round {rnd}: "
                    f"{node_id}; remaining_waiting={sorted(waiting_peers)}"
                )

                # Wake aggregation if all remaining expected peers have responded.
                if waiting_peers.issubset(received_peers):
                    self.ready_event.set()

    def _convert_topology(self, topology: dict) -> None:
        """Build active send/receive neighbors from topology and membership."""
        self.send_neighbors = {}
        self.recv_neighbors = {}

        configured_send_ids = set(topology.get(self.node_id, {}).get("send_ids", []))
        configured_recv_ids = set(topology.get(self.node_id, {}).get("recv_ids", []))

        # Currently, only keep topology neighbors that are currently present
        # according to the Membership Layer.
        if self.membership is not None:
            present_nodes = self.membership.get_present_nodes()
            logger.info(f"[{self.node_id}] Present nodes from membership: {present_nodes}")
            active_send_ids = configured_send_ids & present_nodes
            active_recv_ids = configured_recv_ids & present_nodes
        else:
            # Backward compatibility: if membership is not provided,
            # preserve the original topology behavior.
            active_send_ids = configured_send_ids
            active_recv_ids = configured_recv_ids

        # Store destination addresses and aggregation weights for send neighbors.
        self.send_neighbors = {
            peer_id: {
                "address": topology[peer_id]["address"],
                "weight": topology[peer_id]["weight"],
            }
            for peer_id in active_send_ids
        }

        # Store IDs of peers whose models this node receives.
        self.recv_neighbors = [peer_id for peer_id in active_recv_ids]

        logger.info(
            f"[{self.node_id}] Active neighbors after membership filter: "
            f"send={sorted(self.send_neighbors.keys())}, "
            f"recv={self.recv_neighbors}"
        )

    def _get_topology(self, topology_round: int) -> Dict:
        """Get a topology for a round, using the cache when available."""
        if topology_round in self.topology_cache:
            logger.info(f"[{self.node_id}] Get topology of round {topology_round} from cache")
            return self.topology_cache[topology_round]
        else:
            logger.info(f"[{self.node_id}] Get topology of round {topology_round} from topology server")
            self.topology_cache[topology_round] = self.topology[str(topology_round)]
            return self.topology_cache[topology_round]

    def configure_fit(self, server_round: int) -> None:
        """Select the topology and update neighbors for the current training round."""
        self._apply_pending_topology_update()

        # pull topology for this Flower round
        topology_round = get_topology_round(self.num_topology, server_round)
        current_topology = self._get_topology(topology_round)

        self._convert_topology(current_topology)
        self.current_round = server_round

        logger.info(f"[{self.node_id}] Update topology for current round {self.current_round}, topology round {topology_round}")
        logger.info(f"[{self.node_id}] Current topology: {current_topology}")
    
    def _mix(self, rnd: int, parameters: Parameters, num_samples: int, peer_params: dict) -> Parameters:
        """Aggregate the local model with models received from neighbors."""
        local_weights = fl.common.parameters_to_ndarrays(parameters)
        local_models = (local_weights, num_samples)
        mixed = [local_models]

        logger.info(f"[{self.node_id}] Mixing with peer models: {peer_params.keys()}")
        for peer_id in self.recv_neighbors:
            result = False
            if peer_id in peer_params:
                mixed.append(peer_params[peer_id])
                result = True
            node_store.update_receive_status(rnd, peer_id, result)
        
        mixed = aggregate(mixed)
        mixed = fl.common.ndarrays_to_parameters(mixed)
        return mixed

    def aggregate_fit(
            self, 
            rnd: int, 
            parameters: Parameters,
            num_samples: int,
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Exchange model parameters, wait for peers, and aggregate the results."""

        # Select the topology and determine active neighbors for this round.
        self.configure_fit(rnd)
        logger.info(f"🧩🧩🧩🧩🧩[{self.node_id}] Start aggregate_fit round {rnd}")
        logger.info(f"[{self.node_id}] Aggregated from client for round {rnd} to server")
        logger.info(f"[{self.node_id}] send_neighbors: {len(self.send_neighbors)}, recv_neighbors: {self.recv_neighbors}")

        # Initialize the expected and already received peer sets for this round.
        with self.lock:
            waiting_peers = set(self.recv_neighbors)
            received_peers = set(self.peer_models.get(rnd, {}).keys())
            self._round_waiting_peers[rnd] = waiting_peers
            self._round_received_peers[rnd] = received_peers

            # Skip waiting if all expected peer models are already available.
            if waiting_peers.issubset(received_peers):
                self.ready_event.set()
    
        # Send the local model to each configured send neighbor.
        for peer_id, peer in self.send_neighbors.items():
            logger.info(f"[{self.node_id}] push params to peer_id {peer_id} (round {rnd})")
            result = push_model(
                address=peer["address"],
                node_id=self.node_id,
                rnd=rnd,
                parameters=parameters,
                num_samples=num_samples,
                timeout=self.push_model_timeout
            )
            node_store.update_send_status(rnd, peer_id, result)

        # Wait until expected peer models arrive or the timeout expires.
        start = time.time()
        logger.info(f"[{self.node_id}] Number of rounds in which peer models have been received: {len(self.peer_rnds)}")
        with self.lock:
            if not self.recv_neighbors:
                self.ready_event.set()
            elif rnd in self.peer_rnds:
                self.ready_event.set()
        completed = self.ready_event.wait(timeout=self.receive_model_timeout)
        elapsed = time.time() - start

        # Collect peer models for this round and clear its temporary tracking state.
        logger.info(f"[{self.node_id}] current peer models: {self.peer_models.keys()}")
        peer_params = {}
        with self.lock:
            if rnd in self.peer_models:
                peer_params = self.peer_models[rnd].copy()
                del self.peer_models[rnd]
                logger.info(f"[{self.node_id}] get peer params in round {rnd}")
            elif len(self.peer_models) > 0:
                # Fallback: use models from another available round if this round is missing.
                idx = next(iter(self.peer_models))
                peer_params = self.peer_models[idx].copy()
                del self.peer_models[idx]
                logger.info(f"[{self.node_id}] not found peer params in round {rnd}, so get in round {idx}")
            
            self._round_waiting_peers.pop(rnd, None)
            self._round_received_peers.pop(rnd, None)
            self.ready_event.clear()

        logger.info(
            f"[{self.node_id}] "
            f"peer aggregation: {len(peer_params)}/{len(self.recv_neighbors)} "
            f"in {elapsed:.2f}s (completed={completed}) (round {rnd})"
        )

        # Aggregate available peer models with the local model and save the result.
        mixed = self._mix(rnd, parameters, num_samples, peer_params)
        self.model_store.save("latest", mixed)
                
        return mixed

    def receive_peer_model(self, rnd: int, peer_id: str, num_samples: int, parameters: Parameters) -> None:
        """Store a received peer model and wake aggregation when ready."""
        with self.lock:
            logger.info(f"📩📩📩📩📩[{self.node_id}] Receive peer_id {peer_id} (round {rnd})")

            # Create a buffer for this round and limit the number of stored rounds.
            if rnd not in self.peer_models:
                self.peer_models[rnd] = {}
            if len(self.peer_models) > self.num_peers:
                idx = next(iter(self.peer_models))
                del self.peer_models[idx]

            # Save the received model and its sample count.
            self.peer_models[rnd][peer_id] = (fl.common.parameters_to_ndarrays(parameters), num_samples)

            received_peers = self._round_received_peers.get(rnd)
            waiting_peers = self._round_waiting_peers.get(rnd)

            # Wake aggregation when all expected peers have responded.
            if len(self.peer_models[rnd]) >= len(self.recv_neighbors) or waiting_peers.issubset(received_peers):
                self.ready_event.set()
                if rnd not in self.peer_rnds:
                    self.peer_rnds.append(rnd)

    def cleanup(self) -> None:
        """Clear temporary training state after a training session."""
        with self.lock:
            self.peer_models.clear()
            self.send_neighbors.clear()
            self.recv_neighbors.clear()
            self.peer_rnds.clear()
            self._round_waiting_peers.clear()
            self._round_received_peers.clear()
            self.ready_event.clear()

        with self._topology_update_lock:
            self.topology_cache.clear()
            # Keep pending_topology so the next session can apply it.

        self.current_round = 0
        logger.info(f"[{self.node_id}] Strategy training state cleaned up")
