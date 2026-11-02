from typing import List, Tuple
import flwr as fl
from flwr.server.strategy.aggregate import aggregate

class ModelMixer:
    """Aggregate model parameters from multiple nodes."""

    @staticmethod
    def mix(models: List[Tuple[fl.common.Parameters, int]]) -> fl.common.Parameters:
        """Mix models using sample-weighted averaging.

        Args:
            models: List of (model parameters, number of samples) pairs.

        Returns:
            Aggregated model parameters.
        """

        # Convert Flower parameters to NumPy arrays.
        nds = [
            (fl.common.parameters_to_ndarrays(models[i][0]), models[i][1])
            for i in range(len(models))
        ]

        # Aggregate models using sample-weighted averaging.
        mixed = aggregate(nds)

        # Convert the aggregated arrays back to Flower parameters.
        return fl.common.ndarrays_to_parameters(mixed)
