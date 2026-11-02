import json
import etcd3gw
import threading
from typing import Optional
from typing import Callable, Optional
from app.common.logger import logger

MembershipCallback = Callable[[str, str, set[str]], None]

class MembershipManager:
    def __init__(
        self,
        node_id: str,
        endpoint: str,
        etcd_host: str,
        etcd_port: int = 2379,
        namespace: str = "dfl",
        lease_ttl: int = 30,
    ):
        self.node_id = node_id
        self.endpoint = endpoint

        self.etcd_host = etcd_host
        self.etcd_port = etcd_port
        self.namespace = namespace
        self.lease_ttl = lease_ttl

        self.prefix = f"/{namespace}/nodes/"
        self.node_key = f"{self.prefix}{node_id}"

        self.client = None
        self.lease = None

        # A new Event is created for every lifecycle session.
        self._stop_event = threading.Event()

        self._keepalive_thread: Optional[threading.Thread] = None
        self._watch_thread: Optional[threading.Thread] = None

        self._watch_cancel = None
        self._watch_cancel_lock = threading.Lock()

        self._lease_lock = threading.Lock()

        # Serialize start() and stop().
        self._lifecycle_lock = threading.RLock()

        self._lock = threading.Lock()
        self._present_nodes: set[str] = set()
        self._started = False

        self._listeners: list[MembershipCallback] = []
        self._listeners_lock = threading.Lock()

    def add_membership_listener(self, callback: MembershipCallback) -> None:
        """Register a callback for membership changes."""
        with self._listeners_lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_membership_listener(self, callback: MembershipCallback) -> None:
        """Unregister a membership callback."""
        with self._listeners_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_membership_change(self, event: str, node_id: str) -> None:
        """Notify listeners after membership has changed."""
        present_nodes = self.get_present_nodes()
        logger.info(f"notify_membership_change: {present_nodes}")

        with self._listeners_lock:
            listeners = list(self._listeners)

        for callback in listeners:
            try:
                callback(event, node_id, present_nodes.copy())
            except Exception:
                logger.exception(
                    f"[{self.node_id}] Membership callback failed: "
                    f"event={event}, node_id={node_id}"
                )

    # Lifecycle
    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                if self._stop_event.is_set():
                    raise RuntimeError(f"[{self.node_id}] Previous membership session has not stopped completely")

                logger.info(f"[{self.node_id}] MembershipManager already started")
                return

            logger.info(f"[{self.node_id}] Starting MembershipManager (etcd={self.etcd_host}:{self.etcd_port})")

            client = None
            lease = None
            stop_event = threading.Event()

            self._stop_event = stop_event
            self._watch_cancel = None

            try:
                client = etcd3gw.client(host=self.etcd_host, port=self.etcd_port,)
                self.client = client

                # Create a new lease for this lifecycle session.
                lease = client.lease(self.lease_ttl)
                self.lease = lease

                # Register node and load initial membership.
                self._register()
                revision = self._load_initial_membership()

                # Start background workers with this session's state.
                keepalive_thread = threading.Thread(
                    target=self._keepalive_loop,
                    args=(stop_event, client, lease),
                    name=f"membership-keepalive-{self.node_id}",
                    daemon=True,
                )

                watch_thread = threading.Thread(
                    target=self._watch_loop,
                    args=(revision + 1, stop_event, client),
                    name=f"membership-watch-{self.node_id}",
                    daemon=True,
                )

                self._keepalive_thread = keepalive_thread
                self._watch_thread = watch_thread

                keepalive_thread.start()
                watch_thread.start()

                self._started = True

                logger.info(f"[{self.node_id}] MembershipManager started. present_nodes={self.get_present_nodes()}")

            except Exception:
                logger.exception(f"[{self.node_id}] Failed to start MembershipManager")
                stop_event.set()
                # self._cancel_watch()

                for thread in (self._keepalive_thread, self._watch_thread,):
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=5)

                alive_threads = [
                    thread.name
                    for thread in (self._keepalive_thread, self._watch_thread,)
                    if thread is not None and thread.is_alive()
                ]

                if alive_threads:
                    # Do not allow another session to overwrite state
                    # while an old background thread is still alive.
                    self._started = True
                    raise RuntimeError(f"[{self.node_id}] Startup rollback incomplete; threads still alive: {alive_threads}")

                # self._cleanup_etcd(client, lease)

                self.client = None
                self.lease = None
                self._keepalive_thread = None
                self._watch_thread = None

                with self._lock:
                    self._present_nodes.clear()

                raise

    def stop(self) -> None:
        if not self._started:
            return

        logger.info(f"[{self.node_id}] Stopping MembershipManager")

        self._stop_event.set()

        # Stop watch first.
        if self._watch_cancel is not None:
            try:
                self._watch_cancel()
            except Exception:
                logger.exception(f"[{self.node_id}] Failed to cancel membership watch")

        # Gracefully unregister.
        try:
            if self.client is not None:
                self.client.delete(self.node_key)
        except Exception:
            logger.exception(f"[{self.node_id}] Failed to unregister from etcd")

        # Wait for background threads.
        if (self._keepalive_thread is not None and self._keepalive_thread.is_alive()):
            self._keepalive_thread.join(timeout=2)

        if (self._watch_thread is not None and self._watch_thread.is_alive()):
            self._watch_thread.join(timeout=2)

        with self._lock:
            self._present_nodes.clear()

        self._started = False
        logger.info(f"[{self.node_id}] MembershipManager stopped")

    def _recover_lease(self, client, old_lease):
        """Create a new lease and re-register this node."""
        new_lease = client.lease(self.lease_ttl)

        value = json.dumps({
            "node_id": self.node_id,
            "endpoint": self.endpoint,
        })

        try:
            # Reattach the membership key to the new lease.
            client.put(self.node_key, value, lease=new_lease,)
        except Exception:
            # Avoid leaking a newly created lease if registration fails.
            try:
                new_lease.revoke()
            except Exception:
                logger.exception(f"[{self.node_id}] Failed to revoke unregistered recovery lease")
            raise

        # Publish the new lease only after registration succeeds.
        with self._lease_lock:
            self.lease = new_lease

        # The key is now attached to the new lease.
        with self._lock:
            self._present_nodes.add(self.node_id)

        logger.warning(f"[{self.node_id}] Membership lease recovered; re-registered at {self.node_key}")

        # Revoke the old lease after moving the key to the new lease.
        if old_lease is not None:
            try:
                old_lease.revoke()
            except Exception:
                # It may already have expired or been revoked.
                logger.warning(f"[{self.node_id}] Could not revoke old lease; it may already be expired", exc_info=True,)

        return new_lease

    # Registration
    def _register(self) -> None:
        value = json.dumps({
                "node_id": self.node_id,
                "endpoint": self.endpoint,
            })

        self.client.put(self.node_key, value, lease=self.lease,)
        logger.info(f"[{self.node_id}] Registered membership: {self.node_key} -> {self.endpoint}")

    # Initial snapshot
    def _load_initial_membership(self, client=None) -> int:
        active_client = client or self.client
        if active_client is None:
            raise RuntimeError(f"[{self.node_id}] etcd client is not initialized")

        result = active_client.get_prefix(self.prefix)

        present_nodes = set()
        max_revision = 0

        for value, metadata in result:
            key = metadata["key"]

            if isinstance(key, bytes):
                key = key.decode()

            node_id = key[len(self.prefix):]
            if node_id:
                present_nodes.add(node_id)

            revision = int(metadata["mod_revision"])
            max_revision = max(max_revision, revision)

        with self._lock:
            self._present_nodes = present_nodes

        logger.info(f"[{self.node_id}] Membership snapshot refreshed: {present_nodes}, revision={max_revision}")
        return max_revision

    # Keepalive
    def _keepalive_loop(self, stop_event, client, lease) -> None:
        interval = max(1.0, self.lease_ttl / 3)
        max_consecutive_failures = 3
        consecutive_failures = 0

        logger.info(f"[{self.node_id}] Membership keepalive started (interval={interval}s)")
        while not stop_event.wait(interval):
            try:
                ttl = lease.refresh()

                # etcd3gw returns -1 when the lease has expired.
                if ttl is not None and int(ttl) <= 0:
                    logger.warning(f"[{self.node_id}] Lease expired; attempting recovery")
                    lease = self._recover_lease(client, lease)
                    consecutive_failures = 0
                    continue

                if consecutive_failures:
                    logger.info(f"[{self.node_id}] Lease refresh recovered")
                consecutive_failures = 0

            except Exception:
                consecutive_failures += 1
                logger.exception(
                    f"[{self.node_id}] Lease refresh failed "
                    f"({consecutive_failures}/"
                    f"{max_consecutive_failures})"
                )
                # Avoid creating a new lease after a single transient error.
                if consecutive_failures < max_consecutive_failures:
                    continue

                try:
                    lease = self._recover_lease(client, lease)
                    consecutive_failures = 0
                except Exception:
                    # Keep retrying on subsequent intervals.
                    # Do not mark the node as recovered prematurely.
                    logger.exception(f"[{self.node_id}] Lease recovery failed; will retry")

    # Watch
    def _watch_loop(
        self,
        start_revision: int,
        stop_event,
        client,
    ) -> None:
        revision = start_revision

        logger.info(f"[{self.node_id}] Watch thread entered; start_revision={revision}")
        while not stop_event.is_set():
            # None means we must rebuild the membership snapshot
            # before starting another watch.
            if revision is None:
                try:
                    snapshot_revision = self._load_initial_membership(client)
                    revision = snapshot_revision + 1

                    logger.info(f"[{self.node_id}] Watch resynchronized; next_revision={revision}")
                except Exception:
                    if stop_event.is_set():
                        break
                    logger.exception(f"[{self.node_id}] Failed to reload membership snapshot; retrying")
                    stop_event.wait(1.0)
                    continue

            cancel = None
            try:
                logger.info(f"[{self.node_id}] Starting membership watch from revision={revision}")
                events, cancel = client.watch_prefix(
                    self.prefix,
                    start_revision=revision,
                )

                # Publish the cancel callback so stop() can cancel
                # the currently active watch.
                with self._watch_cancel_lock:
                    self._watch_cancel = cancel
                    should_stop = stop_event.is_set()

                if should_stop:
                    break

                for event in events:
                    if stop_event.is_set():
                        break

                    self._process_event(event)
                    event_revision = self._get_event_revision(event)
                    if event_revision is not None:
                        revision = max(revision, event_revision + 1,)

                if not stop_event.is_set():
                    # The stream ended unexpectedly. Do not blindly
                    # retry using the old revision.
                    logger.warning(f"[{self.node_id}] Membership watch ended unexpectedly; reloading snapshot")
                    revision = None

            except Exception:
                if stop_event.is_set():
                    break

                logger.exception(f"[{self.node_id}] Membership watch failed; will reload snapshot and retry")
                # Also handles watch errors caused by compacted revisions.
                revision = None

            finally:
                # Clean up this watch without clearing a newer watch's
                # callback if another iteration has already replaced it.
                if cancel is not None:
                    try:
                        cancel()
                    except Exception:
                        logger.debug(f"[{self.node_id}] Watch cancellation during cleanup failed", exc_info=True,)

                with self._watch_cancel_lock:
                    if self._watch_cancel is cancel:
                        self._watch_cancel = None

            if not stop_event.is_set() and revision is None:
                stop_event.wait(1.0)

        logger.info(f"[{self.node_id}] Membership watch stopped")

    def _process_event(self, event: dict) -> None:
        kv = event.get("kv")
        if not kv:
            return

        key = kv.get("key")
        if isinstance(key, bytes):
            key = key.decode()

        if not key or not key.startswith(self.prefix):
            return

        node_id = key[len(self.prefix):]
        if not node_id:
            return

        event_type = event.get("type")
        # PUT / CREATE / UPDATE
        if event_type != "DELETE":
            with self._lock:
                # changed = node_id not in self._present_nodes
                self._present_nodes.add(node_id)

        # DELETE
        else:
            with self._lock:
                changed = node_id in self._present_nodes
                self._present_nodes.discard(node_id)
            if changed:
                logger.info(f"[{self.node_id}] Node left: {node_id}")
                self._notify_membership_change("LEAVE", node_id)

    @staticmethod
    def _get_event_revision(event: dict) -> Optional[int]:
        kv = event.get("kv")

        if not kv:
            return None

        revision = kv.get("mod_revision")
        if revision is None:
            return None

        try:
            return int(revision)
        except (TypeError, ValueError):
            return None

    # Public membership API
    def is_present(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._present_nodes

    def get_present_nodes(self) -> set[str]:
        with self._lock:
            return set(self._present_nodes)