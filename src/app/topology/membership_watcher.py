import base64
import etcd3gw
import threading
from typing import Callable, Optional

from app.common.logger import logger


MembershipCallback = Callable[[str, str, set[str]], None]


class TopologyMembershipWatcher:
    def __init__(
        self,
        etcd_host: str,
        etcd_port: int = 2379,
        namespace: str = "dfl",
        known_nodes: Optional[set[str]] = None,
        retry_interval: float = 1.0,
    ):
        self.prefix = f"/{namespace.strip('/')}/nodes/"
        self.known_nodes = set(known_nodes or set())
        self.present_nodes: set[str] = set()
        self.listeners: list[MembershipCallback] = []
        self.retry_interval = retry_interval
        self.client = etcd3gw.client(host=etcd_host, port=etcd_port)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.watch_cancel = None
        self.started = False

    def add_listener(self, callback: MembershipCallback) -> None:
        with self.lock:
            if callback not in self.listeners:
                self.listeners.append(callback)

    def remove_listener(self, callback: MembershipCallback) -> None:
        with self.lock:
            if callback in self.listeners:
                self.listeners.remove(callback)

    def get_present_nodes(self) -> set[str]:
        with self.lock:
            return set(self.present_nodes)

    def start(self) -> None:
        with self.lock:
            if self.started:
                return
            self.stop_event.clear()
            revision = self._load_snapshot()
            self.started = True
            self.thread = threading.Thread(
                target=self._watch_loop,
                args=(revision + 1,),
                name="topology-membership-watcher",
                daemon=True,
            )
            self.thread.start()

        logger.info(
            f"[Topology] Membership watcher started; "
            f"known={sorted(self.known_nodes)}, "
            f"present={sorted(self.get_present_nodes())}"
        )

    def stop(self) -> None:
        with self.lock:
            if not self.started:
                return
            self.stop_event.set()
            cancel = self.watch_cancel
            thread = self.thread

        if cancel is not None:
            try:
                cancel()
            except Exception:
                logger.debug("[Topology] Failed to cancel membership watch", exc_info=True)

        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

        with self.lock:
            self.watch_cancel = None
            self.thread = None
            self.started = False

        logger.info("[Topology] Membership watcher stopped")

    def _load_snapshot(self) -> int:
        """Load membership snapshot and return the global etcd revision."""
        prefix = self.prefix.encode()
        range_end = bytearray(prefix)
        for i in range(len(range_end) - 1, -1, -1):
            if range_end[i] < 0xFF:
                range_end[i] += 1
                range_end = range_end[:i + 1]
                break
        else:
            range_end = bytearray(b"\0")

        response = self.client.post(
            self.client.get_url("/kv/range"),
            json={
                "key": base64.b64encode(prefix).decode(),
                "range_end": base64.b64encode(bytes(range_end)).decode(),
            },
        )

        header = response.get("header") or {}
        revision = int(header.get("revision", 0))
        present_nodes = set()

        for item in response.get("kvs") or []:
            key = base64.b64decode(item["key"]).decode()
            if key.startswith(self.prefix):
                node_id = key[len(self.prefix):]
                if node_id:
                    present_nodes.add(node_id)

        with self.lock:
            self.present_nodes = present_nodes

        logger.info(
            f"[Topology] Membership snapshot loaded: "
            f"present={sorted(present_nodes)}, revision={revision}"
        )
        return revision

    def _watch_loop(self, start_revision: int) -> None:
        revision: Optional[int] = start_revision

        while not self.stop_event.is_set():
            cancel = None
            try:
                if revision is None:
                    revision = self._load_snapshot() + 1

                events, cancel = self.client.watch_prefix(self.prefix, start_revision=revision,)
                with self.lock:
                    self.watch_cancel = cancel

                for event in events:
                    if self.stop_event.is_set():
                        break
                    self._process_event(event)
                    event_revision = self._get_event_revision(event)
                    if event_revision is not None:
                        revision = max(revision or 0, event_revision + 1)

                if not self.stop_event.is_set():
                    logger.warning("[Topology] Membership watch ended; resyncing snapshot")
                    revision = None

            except Exception:
                if self.stop_event.is_set():
                    break
                logger.exception("[Topology] Membership watch failed; retrying")
                revision = None

            finally:
                if cancel is not None:
                    try:
                        cancel()
                    except Exception:
                        logger.debug("[Topology] Watch cleanup failed", exc_info=True)
                with self.lock:
                    if self.watch_cancel is cancel:
                        self.watch_cancel = None

            if not self.stop_event.is_set() and revision is None:
                self.stop_event.wait(self.retry_interval)

        logger.info("[Topology] Membership watch thread exited")

    def _process_event(self, event: dict) -> None:
        kv = event.get("kv") or {}
        key = kv.get("key")
        if isinstance(key, bytes):
            key = key.decode()
        if not key or not key.startswith(self.prefix):
            return

        node_id = key[len(self.prefix):]
        if not node_id:
            return

        event_type = str(event.get("type", "PUT")).upper()
        callback_event = None

        with self.lock:
            was_present = node_id in self.present_nodes
            is_known = node_id in self.known_nodes

            if event_type == "DELETE":
                if not was_present:
                    return
                self.present_nodes.discard(node_id)
                if is_known:
                    callback_event = "LEAVE"
                else:
                    logger.info(f"[Topology] Unknown node left: {node_id}")

            else:
                self.present_nodes.add(node_id)
                if was_present:
                    return
                if is_known:
                    callback_event = "JOIN"
                else:
                    logger.info(f"[Topology] New node detected: {node_id}; no topology-change callback")

            present_nodes = set(self.present_nodes)
            listeners = list(self.listeners)

        if callback_event is None:
            return

        for callback in listeners:
            try:
                callback(callback_event, node_id, present_nodes)
            except Exception:
                logger.exception(f"[Topology] Membership callback failed: event={callback_event}, node={node_id}")

    @staticmethod
    def _get_event_revision(event: dict) -> Optional[int]:
        revision = (event.get("kv") or {}).get("mod_revision")
        try:
            return int(revision) if revision is not None else None
        except (TypeError, ValueError):
            return None
