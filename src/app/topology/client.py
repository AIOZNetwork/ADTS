import grpc
import threading
from typing import Callable, Optional
from app.topology.proto import topology_pb2
from app.topology.proto import topology_pb2_grpc
from app.common.logger import logger
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToDict


class TopologyClient:
    """Manage node registration and listen for topology updates and training triggers."""

    def __init__(
        self,
        node_id: str,
        peer_address: str,
        server_address: str,
        on_topology: Callable[[str, int, dict], None],
        on_trigger: Callable[[int], None],
    ):
        self.node_id = node_id # Unique ID of this DFL node
        self.peer_address = peer_address # Address used by other nodes to reach this node
        self.server_address = server_address # Address of the topology server
        self.on_topology = on_topology # Callback invoked when a new topology snapshot is received
        self.on_trigger = on_trigger # Callback invoked when a training trigger is received

        # Create the gRPC channel and service stub for communicating with the topology server.
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = topology_pb2_grpc.TopologyServiceStub(self.channel)

        self._stop_event = threading.Event() # Signals listener threads to stop
        self._start_lock = threading.Lock() # Prevents concurrent start operations
        self._register_lock = threading.Lock() # Protects registration state and generation
        self._topology_lock = threading.Lock() # Protects the last processed topology identity
        self._trigger_thread: Optional[threading.Thread] = None # Background trigger listener
        self._topology_thread: Optional[threading.Thread] = None # Background topology listener
        self._started = False
        self._registered = False
        self._registration_generation = 0
        self._last_seen_topology_instance_id: Optional[str] = None
        self._last_seen_topology_version = 0

    def get_topology(self, round_id: int) -> dict:
        """Fetch the topology assigned to a specific training round."""
        logger.info(f"[{self.node_id}] Getting topology for round {round_id}")
        resp = self.stub.GetTopology(
            topology_pb2.RoundRequest(round_id=round_id),
            timeout=5,
        )
        return resp.graph

    def list_topologies(self) -> dict:
        """Fetch all available round topologies and convert them to dictionaries."""
        return self.get_topology_snapshot()[2]

    def get_topology_snapshot(self) -> tuple[str, int, dict]:
        """Fetch topology data together with the identity/version of that snapshot."""
        resp = self.stub.ListTopologies(Empty(), timeout=5)
        topology = {
            str(round_id): MessageToDict(
                round_topology,
                preserving_proto_field_name=True,
            )["graph"]
            for round_id, round_topology in resp.topologies.items()
        }
        return resp.topology_instance_id, resp.topology_version, topology

    def get_num_topology(self) -> int:
        """Return the number of topologies currently stored on the server."""
        resp = self.stub.GetNumTopology(Empty(), timeout=5)
        return resp.num_topology

    def start(self) -> None:
        """Start background listeners for training triggers and topology changes."""
        with self._start_lock:
            if self._started:
                return # Avoid starting duplicate listener threads
            self._started = True

        # Run each listener independently so one stream does not block the other.
        self._trigger_thread = threading.Thread(
            target=self._listen_triggers,
            daemon=True,
            name=f"{self.node_id}-trigger-listener",
        )
        self._topology_thread = threading.Thread(
            target=self._listen_topology_changes,
            daemon=True,
            name=f"{self.node_id}-topology-listener",
        )
        self._trigger_thread.start()
        self._topology_thread.start()
        logger.info(f"[{self.node_id}] Topology client started")

    def _ensure_registered(self) -> Optional[int]:
        """Register once for the current connection generation; retry on failure."""
        backoff = 1.0
        max_backoff = 15.0

        while not self._stop_event.is_set():
            with self._register_lock:
                if self._registered:
                    return self._registration_generation # Reuse the current registration

                try:
                    # Register this node before opening or restoring subscription streams.
                    self.stub.RegisterNode(
                        topology_pb2.NodeInfo(
                            node_id=self.node_id,
                            peer_address=self.peer_address,
                        ),
                        timeout=5,
                    )
                    self._registered = True
                    self._registration_generation += 1
                    generation = self._registration_generation
                    logger.info(f"[{self.node_id}] Registered with topology server (generation={generation})")
                    return generation
                except grpc.RpcError as exc:
                    logger.warning(f"[{self.node_id}] RegisterNode failed: {exc.code()}; retry in {backoff:.1f}s")
                except Exception:
                    logger.exception(f"[{self.node_id}] Unexpected RegisterNode error")

            # Wait before retrying, and exit early if shutdown is requested.
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, max_backoff)

        return None

    def _invalidate_registration(self, generation: int) -> None:
        """Invalidate only the registration associated with the failed stream."""
        with self._register_lock:
            if self._registration_generation == generation:
                self._registered = False
                logger.info(f"[{self.node_id}] Registration invalidated (generation={generation})")

    def _registration_is_current(self, generation: int) -> bool:
        """Check whether a listener is using the current registration session."""
        with self._register_lock:
            return (
                self._registered
                and self._registration_generation == generation
            )

    def _listen_triggers(self) -> None:
        """Subscribe to training triggers and reconnect when the stream fails."""
        backoff = 1.0
        max_backoff = 15.0

        while not self._stop_event.is_set():
            # Ensure registration is valid before subscribing to the trigger stream.
            generation = self._ensure_registered()
            if generation is None:
                return

            try:
                request = topology_pb2.NodeId(node_id=self.node_id)
                for trigger in self.stub.SubscribeTriggers(request):
                    if self._stop_event.is_set():
                        return
                    if not self._registration_is_current(generation):
                        break # Discard events from an outdated registration

                    # Forward the requested training round to the node's callback.
                    self.on_trigger(trigger.round_id)
                    backoff = 1.0

                if not self._stop_event.is_set():
                    logger.warning(f"[{self.node_id}] Trigger stream ended")
            except grpc.RpcError as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(f"[{self.node_id}] Trigger stream error: {exc.code()}")
            except Exception:
                if self._stop_event.is_set():
                    return
                logger.exception(f"[{self.node_id}] Unexpected trigger listener error")

            # Force registration again before reconnecting the failed stream.
            self._invalidate_registration(generation)
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, max_backoff)

    def _listen_topology_changes(self) -> None:
        """Subscribe to topology changes and fetch updated topology snapshots."""
        backoff = 1.0
        max_backoff = 15.0

        while not self._stop_event.is_set():
            generation = self._ensure_registered()
            if generation is None:
                return

            try:
                # Ask the server to replay its current version after reconnect.
                request = topology_pb2.TopologyChangeRequest(
                    node_id=self.node_id,
                    last_seen_version=0,
                )
                logger.info(f"[{self.node_id}] Subscribing to topology changes")

                for change in self.stub.SubscribeTopologyChanges(request):
                    if self._stop_event.is_set():
                        logger.info(f"[{self.node_id}] Topology listener stopped")
                        return
                    if not self._registration_is_current(generation):
                        break

                    # Notifications are wake-up signals; fetch the full snapshot and its identity.
                    instance_id, version, topology = self.get_topology_snapshot()
                    if self._stop_event.is_set():
                        return
                    if not self._registration_is_current(generation):
                        break

                    if (change.topology_instance_id != instance_id or change.topology_version != version):
                        logger.debug(
                            f"[{self.node_id}] Topology notification changed during snapshot fetch; "
                            f"notification={change.topology_instance_id}/{change.topology_version}, "
                            f"snapshot={instance_id}/{version}"
                        )

                    with self._topology_lock:
                        already_processed = (
                            instance_id == self._last_seen_topology_instance_id
                            and version <= self._last_seen_topology_version
                        )
                    if already_processed:
                        logger.info(
                            f"[{self.node_id}] Topology snapshot already processed; "
                            f"instance={instance_id}, version={version}"
                        )
                        continue

                    self.on_topology(instance_id, version, topology)
                    logger.info(
                        f"[{self.node_id}] Topology change processed: "
                        f"instance={instance_id}, version={version}"
                    )

                    with self._topology_lock:
                        if instance_id != self._last_seen_topology_instance_id:
                            self._last_seen_topology_instance_id = instance_id
                            self._last_seen_topology_version = version
                        else:
                            self._last_seen_topology_version = max(
                                self._last_seen_topology_version,
                                version,
                            )

                    logger.info(
                        f"[{self.node_id}] Staged topology snapshot "
                        f"instance={instance_id}, version={version}"
                    )
                    backoff = 1.0

                if not self._stop_event.is_set():
                    logger.warning(f"[{self.node_id}] Topology stream ended")
            except grpc.RpcError as exc:
                if self._stop_event.is_set():
                    return
                logger.warning(f"[{self.node_id}] Topology stream error: {exc.code()}")
            except Exception:
                if self._stop_event.is_set():
                    return
                logger.exception(f"[{self.node_id}] Topology listener error")

            self._invalidate_registration(generation)
            self._stop_event.wait(backoff)
            backoff = min(backoff * 2, max_backoff)

    def stop(self) -> None:
        """Stop listener threads and close the gRPC channel."""
        self._stop_event.set()
        self.channel.close()

        current = threading.current_thread()
        for thread in (self._trigger_thread, self._topology_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=2.0)

        logger.info(f"[{self.node_id}] Topology client stopped")