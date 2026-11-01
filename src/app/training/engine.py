import time
import flwr as fl

from app.strategy.graphfed import GraphFedStrategy
from app.client.base import DFLClient
from app.node.store import node_store, NodeStatus
from app.common.logger import logger

class TrainingEngine:
    """Coordinate local client training and strategy-based parameter aggregation."""

    def __init__(self, strategy: GraphFedStrategy, client: DFLClient):
        """Create an engine with the strategy and client used for training."""
        self.strategy = strategy
        self.client = client

    def run_round(self, rnd: int):
        """Train the client for one round, aggregate parameters, and update the model."""
        node_store.start_round(rnd)
        config = {"training_round": rnd}
        parameters = self.client.get_parameters(config)
        parameters, num_samples, _ = self.client.fit(parameters, config)

        # Convert between Flower's parameter format and the client's NumPy arrays.
        parameters = fl.common.ndarrays_to_parameters(parameters)
        parameters = self.strategy.aggregate_fit(
            rnd,
            parameters,
            num_samples,
        )
        parameters = fl.common.parameters_to_ndarrays(parameters)
        self.client.set_parameters(parameters)
        node_store.finish_round(rnd)
        return parameters

    def run(self, num_rounds: int):
        """Run the requested rounds, log elapsed times, and mark the node idle."""
        train_times = {}
        for rnd in range(1, num_rounds + 1):
            # Measure each round independently for the total and average duration.
            start_time = time.time()
            _ = self.run_round(rnd)
            end_time = time.time()
            train_times[rnd] = end_time - start_time
        total_time = sum(train_times.values())
        avg_time = total_time / len(train_times)

        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Average time per round: {avg_time:.2f} seconds")
        
        node_store.set_status(status = NodeStatus.IDLE)