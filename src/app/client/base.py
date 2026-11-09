from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from pathlib import Path
import numpy as np


class DFLClient(ABC):
    """Base class for user-defined training and inference logic."""

    def __init__(self) -> None:
        self.node_id: str = ""
        self.training_round: int = 0 # Current training round

    @abstractmethod
    def get_parameters(self, config: Dict[str, Any]) -> list[np.ndarray]:
        """Return the current model parameters."""
        ...

    @abstractmethod
    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load model parameters."""
        ...

    @abstractmethod
    def fit(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[list[np.ndarray], int, Dict[str, Any]]:
        """Train the model and return updated parameters, sample count, and metrics."""
        ...

    @abstractmethod
    def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model and return loss, sample count, and metrics."""
        ...
    
    @abstractmethod
    def test(self, parameters: list[np.ndarray]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Test the model and return loss, sample count, and metrics."""
        ...

    @abstractmethod
    def get_round_parameters(self, round: int) -> list[np.ndarray]: 
        """Return model parameters for a specific round."""
        ...

    @abstractmethod
    def inference(self, parameters: list[np.ndarray], inputs: bytes) -> np.ndarray: 
        """Run inference on input data and return predictions."""
        ...

    @abstractmethod
    def run_inference(self, weight_path: Path, input_path: Path, output_path: Path) -> None: 
        """Run inference using saved weights and input/output files."""
        ...