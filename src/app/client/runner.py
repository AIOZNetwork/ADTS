import flwr as fl
from typing import Any, Dict, Tuple
import numpy as np
from pathlib import Path
from app.client.base import DFLClient


class ClientAdapter(fl.client.NumPyClient):
    """Adapt DFLClient to Flower's NumPyClient interface."""
    def __init__(self, user_client: DFLClient, node_id: str, ckpt_root: str):
        self.user_client = user_client
        self.node_id = node_id
        self.ckpt_root = ckpt_root

    def get_parameters(self, config: Dict[str, Any]) -> list[np.ndarray]:
        return self.user_client.get_parameters(config)
    
    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.user_client.set_parameters(parameters)

    def fit(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[list[np.ndarray], int, Dict[str, Any]]:
        self.user_client.node_id = self.node_id
        self.user_client.training_round = config["training_round"]
        self.user_client.set_parameters(parameters)
        params, num_examples, metrics = self.user_client.fit(parameters, config)
        return params, num_examples, metrics

    def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.user_client.set_parameters(parameters)
        loss, num_examples, metrics = self.user_client.evaluate(parameters, config)
        return loss, num_examples, metrics
    
    def test(self, parameters: list[np.ndarray]
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.user_client.set_parameters(parameters)
        loss, num_examples, metrics = self.user_client.test(parameters)
        return loss, num_examples, metrics
    
    def get_round_parameters(self, round: int) -> list[np.ndarray]: 
        return self.user_client.get_round_parameters(round)
    
    def inference(self, parameters: list[np.ndarray], inputs: bytes) -> np.ndarray: 
        return self.user_client.inference(parameters, inputs)

    def run_inference(self, weight_path: Path, input_path: Path, output_path: Path) -> None: 
        return self.user_client.run_inference(weight_path, input_path, output_path)