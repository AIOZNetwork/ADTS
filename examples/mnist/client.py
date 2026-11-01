import torch
from app.client.base import DFLClient
from examples.mnist.model import SimpleNet
from examples.mnist.dataset import load_data
from typing import Any, Dict, Tuple
import numpy as np


class MyClient(DFLClient):
    def __init__(self, config: Dict):
        self.model = SimpleNet()
        rounds, steps, train_path, test_path = config["flower"]["rounds"], config["flower"]["steps"], config["flower"]["train_path"], config["flower"]["test_path"]
        self.trainloader, self.testloader = load_data(train_path, test_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rounds = rounds
        self.steps = steps

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
        print(f"Start local training")
        self.set_parameters(parameters)
        print("Loaded parameters for training")
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        idx = 0

        for _ in range(self.rounds):  # local epochs
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                optimizer.step()
                if idx >= self.steps:
                    break
                idx += 1
        # torch.save(self.model.state_dict(), "checkpoint.pt")
        print(f"(Local training) Training loss: {loss}")

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        print(f"Start local evaluation")
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0

        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss += torch.nn.functional.cross_entropy(preds, y).item()
                correct += (preds.argmax(1) == y).sum().item()
        
        print(f"Val accuracy: {correct / len(self.testloader.dataset)}")

        return loss, len(self.testloader.dataset), {
            "accuracy": correct / len(self.testloader.dataset)
        }
