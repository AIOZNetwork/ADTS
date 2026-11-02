import importlib
from app.client.base import DFLClient

def load_user_client(cfg: dict) -> DFLClient:
    """Load and initialize the user-defined client from config."""
    module = importlib.import_module(cfg["client"]["module"])  # Import client module
    cls = getattr(module, cfg["client"]["class"]) # Get client class
    return cls(cfg) # Create and return client instance
