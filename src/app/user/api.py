"""Authenticated user-facing API for training, evaluation, inference, and node info."""

import os
import grpc
import flwr as fl
import base64
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.protobuf.json_format import MessageToDict

from app.user.model import TrainRequest, EvaluateRequest, \
    InferenceRequest, NodeInfoRequest, \
    NodeInfoResponse
from app.user.utils import get_registered_nodes
from app.common.logger import logger
from app.peer.mixer import ModelMixer
from app.peer.client import PeerClient
from app.topology.proto import topology_pb2
from app.topology.proto import topology_pb2_grpc
from app.user.constants import TOPO_ADDRESS

load_dotenv()

app = FastAPI()
API_TOKEN = os.getenv("API_TOKEN")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security),):
    """Reject requests without the configured bearer token."""
    # Treat a missing server-side token as invalid configuration, not as open access.
    if not API_TOKEN or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/train", dependencies=[Depends(verify_token)])
def trigger_training(request: TrainRequest):
    """Ask the topology service to trigger a training round."""
    round_id = request.round_id
    channel = grpc.insecure_channel(TOPO_ADDRESS)
    stub = topology_pb2_grpc.TopologyServiceStub(channel)

    # The RPC currently triggers round 1, while the response echoes the requested ID.
    stub.TriggerTraining(
        topology_pb2.TrainingRequest(round_id=1)
    )
    return {"ok": True, "round_id": round_id}

@app.post("/evaluate", dependencies=[Depends(verify_token)])
def evaluate(request: EvaluateRequest):
    """Mix selected nodes' models and evaluate the result on a node."""
    model_key = request.model_key
    nodes = request.nodes
    registered_nodes = get_registered_nodes(TOPO_ADDRESS)
    logger.info(f"registered_nodes: {registered_nodes}")
    # An empty node list means evaluate using every currently registered node.
    target_nodes = nodes or list(registered_nodes.keys())
    logger.info(f"target_nodes: {target_nodes}")

    models = []
    for node_id in target_nodes:
        logger.info(f"{node_id}: {registered_nodes[node_id]}")
        client = PeerClient(registered_nodes[node_id])
        tensors = client.get_model(model_key)
        tensors = [
            fl.common.bytes_to_ndarray(t) for t in tensors
        ]
        params=fl.common.ndarrays_to_parameters(tensors)
        # Give each selected model equal weight when building the mixed model.
        models.append((params, 1))

    mixed = ModelMixer.mix(models)
    # Convert the mixed Flower parameters back to bytes for the peer API.
    mixed = [
        fl.common.ndarray_to_bytes(arr)
        for arr in fl.common.parameters_to_ndarrays(mixed)
    ]

    client = PeerClient(registered_nodes[target_nodes[0]])
    # Evaluate the shared model on the first selected node.
    results = client.evaluate_with_model(mixed)

    return {
        "model_key": model_key,
        "results": results,
    }

@app.post("/inference", dependencies=[Depends(verify_token)])
def inference(
    request: InferenceRequest
):
    """Mix selected nodes' models and run inference on the input data."""
    model_key = request.model_key
    nodes = request.nodes
    # The request carries input data as base64; peers expect its decoded bytes.
    raw_bytes = base64.b64decode(request.data_b64)

    registered_nodes = get_registered_nodes(TOPO_ADDRESS)
    logger.info(f"registered_nodes: {registered_nodes}")
    # Use all registered nodes unless the request explicitly selects a subset.
    target_nodes = nodes or list(registered_nodes.keys())
    logger.info(f"target_nodes: {target_nodes}")

    models = []
    for node_id in target_nodes:
        logger.info(f"{node_id}: {registered_nodes[node_id]}")
        client = PeerClient(registered_nodes[node_id])
        tensors = client.get_model(model_key)
        tensors = [
            fl.common.bytes_to_ndarray(t) for t in tensors
        ]
        params=fl.common.ndarrays_to_parameters(tensors)
        # Mix selected node models before running inference.
        models.append((params, 1))
    
    mixed = ModelMixer.mix(models)
    # Serialize the mixed parameters in the format expected by the peer.
    mixed = [
        fl.common.ndarray_to_bytes(arr)
        for arr in fl.common.parameters_to_ndarrays(mixed)
    ]

    client = PeerClient(registered_nodes[target_nodes[0]])

    logger.info("Start inference")
    results = client.inference_with_model(mixed, raw_bytes)
    logger.info("End inference")

    return {
        "model_key": model_key,
        # Return inference bytes as base64 so they can be encoded in JSON.
        "results_b64": base64.b64encode(results).decode("ascii"),
    }

@app.post("/node_info", response_model=NodeInfoResponse, dependencies=[Depends(verify_token)])
def node_info(request: NodeInfoRequest):
    """Fetch a node's information and return it in the API response format."""
    try:
        logger.info(f"Node info: {request.node_address}")
        client = PeerClient(request.node_address)
        results = client.get_node_info(request.round)
        # Keep protobuf field names stable in the API response.
        results = MessageToDict(
            results,
            preserving_proto_field_name=True,
        )

        logger.info(f"Node info: {results}")
        return NodeInfoResponse(
            status="success",
            data=results,
            message=f"Get node info successfully!"
        )
    except:
        # Return the failure details using the same response model as success.
        error_details = traceback.format_exc()
        logger.info(f"Error when get node info: {error_details}")
        return NodeInfoResponse(
            status="failed",
            data=None,
            message=f"Error when get node info: {error_details}"
        )