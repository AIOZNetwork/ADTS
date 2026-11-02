from typing import Dict, Optional
from threading import Lock
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# Runtime state
class NodeStatus(Enum):
    STARTING = "starting"
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class RoundRecord:
    """Store training and peer communication status for one round."""
    round: int
    train_started_at: Optional[datetime] = None
    train_finished_at: Optional[datetime] = None
    send_status: Dict[str, bool] = field(default_factory=dict)
    receive_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class NodeRuntime:
    """Store the node's current runtime state."""
    node_status: NodeStatus = NodeStatus.IDLE
    current_round: int = 0
    round_records: Dict[int, RoundRecord] = field(default_factory=dict)


# Node Store
class NodeStore:
    """Manage node runtime state and per-round records."""

    def __init__(self):
        # Protect shared runtime data
        self.lock = Lock()
        # Runtime state
        self.runtime = NodeRuntime()

    def set_status(self, status: NodeStatus):
        """Update the node's runtime status."""
        with self.lock:
            self.runtime.node_status = status

    def get_status(self) -> NodeStatus:
        """Return the current node status."""
        return self.runtime.node_status

    def get_current_round(self) -> int:
        """Return the current training round."""
        return self.runtime.current_round

    def start_round(self, round_id: int):
        """Record the start of a training round."""
        with self.lock:
            self.runtime.current_round = round_id

            self.runtime.round_records[round_id] = RoundRecord(
                round=round_id,
                train_started_at=datetime.now(),
            )

    def finish_round(self, round_id: int):
        """Record the end of a training round."""
        with self.lock:
            record = self.runtime.round_records[round_id]
            record.train_finished_at = datetime.now()

    def update_send_status(self, round_id: int, peer_id: str, success: bool,):
        """Record whether sending model data to a peer succeeded."""
        with self.lock:
            self.runtime.round_records[
                round_id
            ].send_status[peer_id] = success

    def update_receive_status(self, round_id: int, peer_id: str, success: bool,):
        """Record whether receiving model data from a peer succeeded."""
        with self.lock:
            self.runtime.round_records[
                round_id
            ].receive_status[peer_id] = success

    def get_round_record(self, round: int,) -> Dict[int, RoundRecord]:
        """Return records for a specific round, or all rounds if round is -1."""
        if round == -1:
            return self.runtime.round_records
        else:
            return {round: self.runtime.round_records.get(round)}

# Shared store for node runtime state
node_store = NodeStore()