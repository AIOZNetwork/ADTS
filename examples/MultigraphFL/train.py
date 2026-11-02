import torch
from tqdm import tqdm
from examples.MultigraphFL.logger import logger
from examples.MultigraphFL.utils.utils import get_model, get_iterator
from examples.MultigraphFL.task import train


class MyClient:
    def __init__(self):
        self.node_id = "train"

        self.model_path = "checkpoints"
        
        train_path = "benchmark/MultigraphFL/data/femnist/gaia/train/train.pkl"
        name = "femnist"
        batch_size = 128
        optimizer_name = "adam"
        lr_scheduler_name = "sqrt"
        initial_lr = 0.001
        
        self.local_epochs = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = get_iterator(name, train_path, self.device, batch_size=batch_size)

        self.model = get_model(name, self.device, self.trainloader, epoch_size=None,
                    optimizer_name=optimizer_name, lr_scheduler=lr_scheduler_name,
                    initial_lr=initial_lr, coeff=1.0)

        self.model.net.to(self.device)

    def fit(self, idx):
        print(f"[{self.node_id}] Start local training")
        self.model.net.train()

        train_loss, num_samples = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        if (idx+1) % 400 == 0:
            torch.save(self.model.net.state_dict(), f"checkpoints/checkpoint{idx}.pt")
        print(f"[{self.node_id}] (Local training) Training loss: {train_loss}")

if __name__ == "__main__":
    client = MyClient()
    for idx in tqdm(range(6400)):
        client.fit(idx)
