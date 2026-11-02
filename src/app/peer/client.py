import grpc
import time
import flwr as fl
from flwr.common import (
    Parameters
)
from typing import Dict, List
from app.peer.proto import peer_pb2, peer_pb2_grpc
from app.common.logger import logger
from app.common.constants import MAX_MODEL_SIZE, PUSH_MODEL_TIMEOUT


class PeerClient:
    """Client for communicating with other DFL nodes over gRPC."""

    def __init__(self, address: str):
        # Create a gRPC channel with support for large model messages.
        self.stub = peer_pb2_grpc.PeerFLStub(
            grpc.insecure_channel(
                    address,
                    options=[
                ("grpc.max_send_message_length", MAX_MODEL_SIZE),    # 100 MB
                ("grpc.max_receive_message_length", MAX_MODEL_SIZE), # 100 MB
            ])
        )

    def inference_with_model(self, parameters: List[bytes], inputs: bytes) -> bytes:
        """Run inference on a peer using the provided model parameters."""
        resp = self.stub.InferenceWithModel(
                peer_pb2.InferenceRequest(parameters=parameters, inputs=inputs)
            )
        return resp.outputs

    def evaluate_with_model(self, parameters: List[bytes]) -> Dict[str, float]:
        """Evaluate the model on a peer and return its metrics."""
        resp = self.stub.EvaluateWithModel(
                peer_pb2.EvalWithModelRequest(parameters=parameters)
            )
        return resp.metrics
    
    def get_model(self, model_key: str) -> List[bytes]:
        """Retrieve model parameters from a peer."""
        resp = self.stub.GetModel(
            peer_pb2.ModelRequest(model_key=model_key)
        )
        return resp.parameters

    def get_node_info(self, round: int) -> Dict:
        """Retrieve a peer's information for a specific round."""
        resp = self.stub.GetNodeInfo(peer_pb2.RoundRequest(round=round))
        return resp


def push_model(
        address: str, 
        node_id: str, 
        rnd: int, 
        parameters: Parameters, 
        num_samples: int,
        timeout: float = PUSH_MODEL_TIMEOUT,) -> None:
    """Send model parameters and metadata to a peer, returning success status."""

    # Create a gRPC client for the target peer.
    stub = peer_pb2_grpc.PeerFLStub(
        grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", MAX_MODEL_SIZE),    # 100 MB
                ("grpc.max_receive_message_length", MAX_MODEL_SIZE), # 100 MB
            ])
    )

    # Convert Flower parameters into serialized NumPy tensors.
    tensors = [
        fl.common.ndarray_to_bytes(arr)
        for arr in fl.common.parameters_to_ndarrays(parameters)
    ]

    # Attach round, sample count, and model metadata.
    meta = peer_pb2.ModelMetadata(
        node_id=node_id,
        round=rnd,
        num_samples=num_samples,
        model_hash="na",
    )

    update = peer_pb2.ModelUpdate(
        meta=meta,
        tensors=tensors,
    )

    try:
        # Send the model update and wait for the peer's acknowledgment.
        start_time = time.time()
        ack = stub.PushModel(update, timeout=timeout)
        logger.info(f"[{node_id}] PushModel took {time.time() - start_time}s, response: {ack}")
    except grpc.RpcError as exc:
        # Log gRPC errors such as unavailable peers or request timeouts.
        status_code = exc.code()
        logger.warning(
            f"[{node_id}] PushModel to {address} failed "
            f"(round={rnd}, status={status_code}, "
            f"details={exc.details()})"
        )
        return False
    except:
        logger.warning(f"Node unavailable to push model")
        return False
    return True
