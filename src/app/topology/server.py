import grpc
import threading
import uuid
from typing import Dict, Iterator, Optional
from queue import Queue, Empty

from google.protobuf import empty_pb2
from app.topology.proto import topology_pb2
from app.topology.proto import topology_pb2_grpc
from app.topology.membership_watcher import TopologyMembershipWatcher
from app.topology.custom_topology.generate_network_multigraph import generate_network_multigraph
from app.common.logger import logger
from utils.utils import load_topology


class TopologyServer(topology_pb2_grpc.TopologyServiceServicer):
    """Serve topology data and training triggers, updating topology as membership changes."""

    def __init__(
        self,
        graph_config: str,
        node_config: str,
        etcd_host: str = "127.0.0.1",
        etcd_port: int = 2379,
        graph_generation_config: Optional[str] = None,
    ):
        """Load the topology and initialize membership and subscriber state."""
        self.graph_generation_config = graph_generation_config
        self.nodes: Dict[str, str] = {}
        self.node_config = node_config
        self.topology, self.topology_nodes = load_topology(graph_config, node_config)
        self.topology_instance_id = str(uuid.uuid4())
        self.topology_version = 1
        self._last_regenerated_membership: Optional[frozenset[str]] = None
        self.known_node_metadata = {
            node_id: dict(metadata)
            for node_id, metadata in self.topology_nodes.items()
        }

        logger.info(f"topology nodes: {self.topology_nodes}")
        self.num_topology = len(self.topology)
        self.trigger_queues: Dict[str, Queue] = {}

        self.topology_lock = threading.RLock()
        self.regeneration_queue: Queue = Queue(maxsize=1)
        self.regeneration_stop_event = threading.Event()
        self.regeneration_thread: Optional[threading.Thread] = None

        self.topology_subscribers: Dict[str, set[Queue]] = {}
        self.topology_subscribers_lock = threading.Lock()

        # Nodes with metadata already available in the loaded topology/config.
        self.known_node_ids = set(self.topology_nodes.keys())
        self.membership_watcher = TopologyMembershipWatcher(
            etcd_host=etcd_host,
            etcd_port=etcd_port,
            namespace="dfl",
            known_nodes=self.known_node_ids,
        )
        self.membership_watcher.add_listener(self._on_membership_change)

    def SubscribeTopologyChanges(self, request, context):
        """Stream topology version updates to a subscribing node."""
        node_id = request.node_id
        last_seen_version = request.last_seen_version
        subscriber_queue = Queue(maxsize=1)

        # Keep one queue per stream so updates can be delivered to every subscriber.
        with self.topology_subscribers_lock:
            self.topology_subscribers.setdefault(node_id, set()).add(subscriber_queue)

        try:
            with self.topology_lock:
                current_instance_id = self.topology_instance_id
                current_version = self.topology_version

            if current_version > last_seen_version:
                yield topology_pb2.TopologyChange(
                    topology_version=current_version,
                    topology_instance_id=current_instance_id,
                )
                last_seen_version = current_version

            while context.is_active():
                try:
                    subscriber_queue.get(timeout=0.5)
                except Empty:
                    continue

                with self.topology_lock:
                    current_instance_id = self.topology_instance_id
                    current_version = self.topology_version

                if current_version > last_seen_version:
                    yield topology_pb2.TopologyChange(
                        topology_version=current_version,
                        topology_instance_id=current_instance_id,
                    )
                    last_seen_version = current_version

        finally:
            with self.topology_subscribers_lock:
                subscribers = self.topology_subscribers.get(node_id)
                if subscribers is not None:
                    subscribers.discard(subscriber_queue)
                    if not subscribers:
                        self.topology_subscribers.pop(node_id, None)

    def _notify_topology_changed(self, topology_version: int) -> None:
        """Notify all active streams that a new topology version is available."""
        with self.topology_subscribers_lock:
            subscribers = [
                queue
                for node_queues in self.topology_subscribers.values()
                for queue in node_queues
            ]

        # Replace a pending notification so subscribers always learn the newest version.
        for subscriber_queue in subscribers:
            try:
                subscriber_queue.put_nowait(topology_version)
            except Exception:
                try:
                    subscriber_queue.get_nowait()
                except Empty:
                    pass

                try:
                    subscriber_queue.put_nowait(topology_version)
                except Exception:
                    logger.warning(f"[Topology] Could not enqueue topology notification version={topology_version}")

    def start_membership_watcher(self) -> None:
        """Start membership monitoring and the optional regeneration worker."""
        if self.graph_generation_config and self.regeneration_thread is None:
            self.regeneration_stop_event.clear()
            self.regeneration_thread = threading.Thread(
                target=self._regeneration_loop,
                name="topology-regeneration-worker",
                daemon=True,
            )
            self.regeneration_thread.start()
        
        self.membership_watcher.start()

        if self.graph_generation_config:
            # Reconcile the loaded topology with current etcd membership.
            self._request_regeneration()

    def stop_membership_watcher(self) -> None:
        """Stop membership monitoring and wait for the worker to exit."""
        self.membership_watcher.stop()
        self.regeneration_stop_event.set()

        thread = self.regeneration_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)

        self.regeneration_thread = None
        logger.info("[Topology] Membership watcher stopped")

    def _on_membership_change(
        self,
        event: str,
        node_id: str,
        present_nodes: set[str],
    ) -> None:
        """Request a topology refresh when a known node changes membership."""
        logger.info(
            f"[Topology] Membership callback: event={event}, "
            f"node={node_id}, present_nodes={sorted(present_nodes)}"
        )
        if node_id not in self.known_node_ids:
            return
        
        if self.graph_generation_config:
            self._request_regeneration()

    def _request_regeneration(self) -> None:
        """Queue a regeneration request, coalescing duplicate events."""
        # Coalesce repeated events; the worker reads the latest membership snapshot.
        try:
            self.regeneration_queue.put_nowait(True)
        except Exception:
            # A regeneration is already queued; it will read the latest snapshot.
            pass

    def _regeneration_loop(self) -> None:
        """Process queued requests using the latest active-node snapshot."""
        while not self.regeneration_stop_event.is_set():
            try:
                self.regeneration_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                if self.regeneration_stop_event.is_set():
                    break

                # Read the latest snapshot, not the event's potentially stale snapshot.
                active_nodes = frozenset(
                    self.membership_watcher.get_present_nodes()
                    & self.known_node_ids
                )

                if active_nodes == self._last_regenerated_membership:
                    logger.debug("[Topology] Membership unchanged; skipping regeneration")
                    continue

                self._regenerate_topology(set(active_nodes))
            finally:
                self.regeneration_queue.task_done()

        logger.info("[Topology] Regeneration worker exited")

    def _regenerate_topology(self, active_nodes: set[str]) -> None:
        """Generate and publish a topology for the given active nodes."""
        snapshot = frozenset(active_nodes)

        if len(snapshot) < 2:
            # Keep the existing topology until enough nodes are available.
            logger.warning("[Topology] Not enough active nodes; keeping current topology")
            self._last_regenerated_membership = snapshot
            return

        try:
            logger.info(f"[Topology] Regenerating topology for {sorted(snapshot)}")

            graph_path = generate_network_multigraph(
                json_path=self.graph_generation_config,
                active_nodes=set(snapshot),
            )
            new_topology, new_topology_nodes = load_topology(graph_path, self.node_config,)

            if not new_topology or not new_topology_nodes:
                logger.warning("[Topology] Generated topology is empty; keeping current topology")
                return

            # Discard results if membership changed while the graph was being generated.
            latest_nodes = frozenset(self.membership_watcher.get_present_nodes() & self.known_node_ids)
            if latest_nodes != snapshot:
                logger.info("[Topology] Membership changed during generation; discarding stale topology")
                self._request_regeneration()
                return

            with self.topology_lock:
                self.topology = new_topology
                self.topology_nodes = new_topology_nodes
                self.num_topology = len(new_topology)
                self.topology_version += 1
                self._last_regenerated_membership = snapshot
                new_version = self.topology_version

            # Notify streams after publishing the new topology version.
            self._notify_topology_changed(new_version)
            logger.info(f"[Topology] Topology updated: version={new_version}, path={graph_path}, rounds={self.num_topology}")

        except Exception:
            logger.exception("[Topology] Regeneration failed; keeping current topology")

    def ListTopologies(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> topology_pb2.TopologyList:
        """Return all configured topologies."""
        with self.topology_lock:
            topologies = {}
            for round_id, topo in self.topology.items():
                graph = {}

                for node_id, node in topo.items():
                    graph[node_id] = topology_pb2.NodeDetails(
                        address=node["address"],
                        weight=node["weight"],
                        send_ids=node["send_ids"],
                        recv_ids=node["recv_ids"],
                    )

                topologies[str(round_id)] = topology_pb2.Topology(graph=graph)

            return topology_pb2.TopologyList(
                topologies=topologies,
                topology_instance_id=self.topology_instance_id,
                topology_version=self.topology_version,
            )

    def ListNodes(self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> topology_pb2.NodeList:
        """Return node IDs and peer addresses in the current topology."""
        with self.topology_lock:
            nodes = [
                    topology_pb2.NodeInfo(
                        node_id=node_id,
                        peer_address=self.topology_nodes[node_id]["address"],
                    )
                    for node_id in self.topology_nodes
            ]
            return topology_pb2.NodeList(nodes=nodes)

    def GetNumTopology(self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> topology_pb2.NumTopology:
        """Return the number of rounds in the current topology."""
        with self.topology_lock:
            return topology_pb2.NumTopology(num_topology=self.num_topology)
    
    def GetTopology(self, request: topology_pb2.RoundRequest, context: grpc.ServicerContext
    ) -> topology_pb2.Topology:
        """Return the topology for one round, or NOT_FOUND if it is absent."""
        with self.topology_lock:
            round_id = str(request.round_id)

            if round_id not in self.topology:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"No topology for round {round_id}",
                )

            graph = self.topology[round_id]
            return topology_pb2.Topology(graph=graph)

    def RegisterNode(self, request: topology_pb2.NodeInfo, context: grpc.ServicerContext
    ) -> topology_pb2.Ack:
        """Register a node and create its training-trigger queue."""
        self.nodes[request.node_id] = request.peer_address
        self.trigger_queues[request.node_id] = Queue()
        logger.info(f"[Topology] Registered {request.node_id}")
        return topology_pb2.Ack(ok=True)

    # Trigger stream (Node-side)
    def SubscribeTriggers(self, request: topology_pb2.NodeId, context: grpc.ServicerContext
    ) -> Iterator[topology_pb2.TrainingTrigger]:
        """Stream training triggers from the node's queue."""
        q = self.trigger_queues[request.node_id]
        while context.is_active():
            try:
                trigger = q.get(timeout=0.5)
            except Empty:
                continue
            yield trigger

    # User-triggered broadcast
    def TriggerTraining(self, request: topology_pb2.TrainingRequest, context: grpc.ServicerContext
    ) -> topology_pb2.Ack:
        """Broadcast a training-round trigger to all registered nodes."""
        logger.info(f"[Topology] Trigger training round {request.round_id}")
        for q in self.trigger_queues.values():
            q.put(
                topology_pb2.TrainingTrigger(
                    round_id=request.round_id
                )
            )
        return topology_pb2.Ack(ok=True)
