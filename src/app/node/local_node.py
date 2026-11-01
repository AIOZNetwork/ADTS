from pathlib import Path
from app.config.schema import NodeConfig
from app.client.runner import ClientAdapter
from app.config.loader import load_user_client

class LocalNode:
    """Run local inference using the user-defined client."""
    def __init__(self, cfg: NodeConfig):
        user_client = load_user_client(cfg)
        self.flower_client = ClientAdapter(user_client)

    def inference(self, weight_path: Path, input_path: Path, output_path: Path):
        outputs = self.flower_client.run_inference(weight_path, input_path, output_path)
        return outputs