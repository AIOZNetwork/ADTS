import torch
from tqdm import tqdm
from examples.autonomous_driving.models.FADNet import FADNet
from examples.autonomous_driving.task import get_iterator, train
from examples.autonomous_driving.logger import logger


class MyClient:
    def __init__(self):
        self.node_id = "train"
        self.model = FADNet()

        self.model_path = "checkpoints"
        
        train_path = "benchmark/FADNet/data/driving_carla/gaia/train/train.npz"
        name = "driving_carla"
        batch_size = 32
        
        self.local_epochs = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = get_iterator(name, train_path, self.device, batch_size=batch_size)

        self.model.to(self.device)

    def fit(self, idx):
        print(f"[{self.node_id}] Start local training")
        self.model.train()

        train_loss, num_samples = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        if idx % 100 == 0:
            torch.save(self.model.state_dict(), f"checkpoint{idx}.pt")
        print(f"[{self.node_id}] (Local training) Training loss: {train_loss}")

if __name__ == "__main__":
    client = MyClient()
    for idx in tqdm(range(1600)):
        client.fit(idx)
