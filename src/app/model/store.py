from typing import Dict, List
from flwr.common import Parameters
from threading import Lock
import time


class ModelStore:
    """Store model parameters and their save timestamps."""

    def __init__(self):
        self.models: Dict[str, Parameters] = {}
        self.timestamps: Dict[str, float] = {}
        self.lock = Lock() # Protect shared data

    def save(self, key: str, params: Parameters) -> None:
        """Save model parameters and update their timestamp."""
        with self.lock:
            self.models[key] = params
            self.timestamps[key] = time.time()

    def get(self, key: str = "latest") -> Parameters:
        """Get a model by ID, or return the most recently saved model."""
        if key == "latest":
            return self.models[max(self.timestamps, key=self.timestamps.get)]
        return self.models[key]

    def list(self) -> List:
        """Return all saved model IDs."""
        return list(self.models.keys())