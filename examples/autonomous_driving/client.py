import os
import torch
from app.client.base import DFLClient
from examples.autonomous_driving.models.FADNet import FADNet
from examples.autonomous_driving.task import get_iterator, train, test
from typing import Any, Dict, Tuple
import numpy as np
from examples.autonomous_driving.logger import logger


class MyClient(DFLClient):
    def __init__(self, config: dict):
        self.model = FADNet()

        self.model_path = "checkpoints"
        self.server_round = 0
        
        train_path = config["flower"]["train_path"]
        test_path = config["flower"]["test_path"]
        name = config["flower"]["name"]
        batch_size = config["flower"]["batch_size"]
        
        self.node_id = config["node_id"]
        self.local_epochs = config["flower"]["local_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = get_iterator(name, train_path, self.device, batch_size=batch_size)
        self.testloader = None #get_iterator(name, test_path, self.device, batch_size=batch_size)

        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, Any]) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )

    def fit(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[list[np.ndarray], int, Dict[str, Any]]:
        logger.info(f"[{self.node_id}] Start local training")
        self.set_parameters(parameters)
        logger.info(f"[{self.node_id}] Loaded parameters for training")
        self.model.train()

        train_loss, num_samples = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        # torch.save(self.model.state_dict(), "checkpoint.pt")
        logger.info(f"[{self.node_id}] (Local training) Training loss: {train_loss}")
        self.server_round += 1

        parameters = self.get_parameters(config)

        return parameters, num_samples, {}

    def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        logger.info(f"[{self.node_id}] Start local evaluation")
        self.set_parameters(parameters)
        self.model.eval()
        if self.server_round % 100 != 0:
            logger.info(f"[{self.node_id}] No validation data provided, skipping evaluation.")
            return 0.0, 1, {}
        else:
            ckpt_path = os.path.join(self.model_path, f"{self.server_round}/{self.node_id}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            
            # loss, accuracy = test(self.model, self.testloader, self.device)
            torch.save(self.model.state_dict(), ckpt_path)
            # logger.info(f"[{self.node_id}] (Local evaluation) (round {self.server_round}) Evaluation loss: {loss}, rmse: {accuracy}")
            return 0.0, 1, {}
            # return loss, len(self.testloader.dataset), {"accuracy": accuracy}
