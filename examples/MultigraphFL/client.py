import os
import torch
from app.client.base import DFLClient
from typing import Any, Dict, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from examples.MultigraphFL.logger import logger
from examples.MultigraphFL.utils.utils import get_model, get_iterator
from examples.MultigraphFL.task import train, test


class MyClient(DFLClient):
    def __init__(self, config: dict):
        self.ckpt_root = config["flower"]["ckpt_root"]
        self.ckpt_round = config["flower"]["ckpt_round"]
        self.server_round = 0
        
        train_path = config["flower"]["train_path"]
        test_path = config["flower"]["test_path"]
        self.data_name = config["flower"]["name"]
        self.batch_size = config["flower"]["batch_size"]
        self.optimizer_name = "adam"
        self.lr_scheduler_name = "sqrt"
        self.initial_lr = 0.001
        
        self.node_id = config["node_id"]
        self.local_epochs = config["flower"]["local_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = get_iterator(self.data_name, train_path, self.device, batch_size=self.batch_size)
        self.testloader = get_iterator(self.data_name, test_path, self.device, batch_size=self.batch_size)

        self.model = get_model(self.data_name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                    initial_lr=self.initial_lr, coeff=1.0)
        self.model.net.to(self.device)

        self.transforms = Compose([Resize(28),
                            ToTensor(),
                            Normalize((0.1307,), (0.3081,))
                          ])

    def get_parameters(self, config: Dict[str, Any]) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.net.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = dict(zip(self.model.net.state_dict().keys(), parameters))
        self.model.net.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )

    def fit(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[list[np.ndarray], int, Dict[str, Any]]:
        logger.info(f"[{self.node_id}] Start local training")
        self.set_parameters(parameters)
        logger.info(f"[{self.node_id}] Loaded parameters for training")
        self.model.net.train()

        #train
        train_loss, num_samples = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        logger.info(f"[{self.node_id}] (Local training) Training loss: {train_loss}")
        self.server_round += 1

        parameters = self.get_parameters(config)
        model_size = sum(param.nbytes for param in parameters) * 8
        logger.info(f"[{self.node_id}] model size (in bit): {model_size}")

        #save ckpt
        if self.server_round % self.ckpt_round == 0:
            ckpt_path = (self.checkpoint_dir / f"{self.node_id}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.net.state_dict(), ckpt_path)

        return parameters, num_samples, {}

    def evaluate(self, parameters: list[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        logger.info(f"[{self.node_id}] Start local evaluation")
        self.set_parameters(parameters)
        self.model.net.eval()
        if self.server_round % self.ckpt_round != 0:
            logger.info(f"[{self.node_id}] No validation data provided, skipping evaluation.")
            return 0.0, 1, {}
        else:
            ckpt_path = (self.checkpoint_dir / f"{self.node_id}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.net.state_dict(), ckpt_path)
            return 0.0, 1, {}
    
    def test(self, parameters: list[np.ndarray]) -> Tuple[float, int, Dict[str, Any]]:
        logger.info(f"[{self.node_id}] [Test] Prepare")

        # Init model
        model = get_model(self.data_name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                    initial_lr=self.initial_lr, coeff=1.0)
        model.net.to(self.device)

        # Load model
        state_dict = dict(zip(model.net.state_dict().keys(), parameters))
        model.net.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )
        model.net.eval()
        logger.info(f"[{self.node_id}] [Test] Loaded model")

        # Test
        loss, accuracy = test(model, self.testloader, self.device)
        logger.info(f"[{self.node_id}] [Test] Loss: {loss}, accuracy: {accuracy}, dataset: {len(self.testloader.dataset)}")
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
    
    def inference(self, parameters: list[np.ndarray], inputs: bytes) -> np.ndarray:
        logger.info(f"[{self.node_id}] [inference] Prepare")

        # Init model
        model = get_model(self.data_name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                    initial_lr=self.initial_lr, coeff=1.0)
        model.net.to(self.device)
        model.net.eval()
        logger.info(f"[{self.node_id}] [inference] Init model")

        # Load model
        state_dict = dict(zip(model.net.state_dict().keys(), parameters))
        model.net.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )
        logger.info(f"[{self.node_id}] [inference] Load model")

        # Inference
        image = Image.open(io.BytesIO(inputs)).convert("RGB")
        image = self.transforms(image).to(self.device)
        image = torch.unsqueeze(image, dim=0)
        logger.info(f"[{self.node_id}] [inference] image {image.shape}")
        output = model.net(image)
        logger.info(f"[{self.node_id}] [inference] output: {output}, {output.shape}")
        return output.cpu().detach().numpy()

    def run_inference(self, weight_path: Path, input_path: Path, output_path: Path) -> None: 
        logger.info(f"[{self.node_id}] [inference] Prepare")
        
        # Init model
        model = get_model(self.data_name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                    initial_lr=self.initial_lr, coeff=1.0)
        model.net.to(self.device)
        model.net.eval()
        logger.info(f"[{self.node_id}] [inference] Init model")

        # Load weights
        if weight_path.is_file():
            logger.info(f"[{self.node_id}] Load weight {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
        elif weight_path.is_dir():
            checkpoints = sorted(weight_path.glob("*.pt"))
            if len(checkpoints) == 0:
                raise RuntimeError(f"No weight found in {weight_path}")

            logger.info(f"[{self.node_id}] Average {len(checkpoints)} weights")
            avg_state_dict = None
            for ckpt in checkpoints:
                logger.info(f"Load {ckpt}")
                current_state = torch.load(ckpt,map_location="cpu")

                if avg_state_dict is None:
                    avg_state_dict = { k: v.float() for k, v in current_state.items()}
                else:
                    for k in avg_state_dict:
                        avg_state_dict[k] += current_state[k].float()

            # arithmetic mean
            for k in avg_state_dict:
                avg_state_dict[k] /= len(checkpoints)
            state_dict = avg_state_dict
        else:
            raise ValueError(f"Invalid weight path: {weight_path}")

        model.net.load_state_dict(state_dict, strict=True)
        logger.info(f"[{self.node_id}] Load model done")

        # Prepare inputs
        if input_path.is_file():
            inputs = [input_path]

        elif input_path.is_dir():
            inputs = sorted([f for f in input_path.iterdir() if f.is_file()])
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        # Inference
        with torch.no_grad():
            for file in inputs:
                logger.info(f"[{self.node_id}] Infer {file.name}")

                image = Image.open(file).convert("RGB")
                image = self.transforms(image).to(self.device)
                tensor = torch.unsqueeze(image, dim=0)
                
                output = model.net(tensor)
                result = (output.cpu().numpy())
                logger.info(f"file {file}: {result.shape}")
                if output_path:
                    save_file = (output_path / f"{file.stem}.npy")
                    np.save(save_file, result)
                else:
                    logger.info(f"{file.name}: {result}")

        logger.info(f"[{self.node_id}] Inference finished")
    
    def get_round_parameters(self, round: int) -> list[np.ndarray]:
        logger.info("Start to get_round_parameters")
        ckpt_path = os.path.join(self.ckpt_root, f"{round}/{self.node_id}.pt")
        state_dict = torch.load(ckpt_path, weights_only=True)
        model = get_model(self.data_name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                    initial_lr=self.initial_lr, coeff=1.0)
        model.net.to(self.device)
        model.net.load_state_dict(state_dict)
        model.net.eval()
        logger.info("End to get_round_parameters")
        return [val.cpu().numpy() for val in model.net.state_dict().values()]