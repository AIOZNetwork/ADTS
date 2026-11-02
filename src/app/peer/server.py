import grpc
import traceback
import flwr as fl
from concurrent import futures

from app.client.base import DFLClient
from app.model.store import ModelStore
from app.peer.proto import peer_pb2, peer_pb2_grpc
from app.strategy.graphfed import GraphFedStrategy
from app.common.logger import logger
from app.common.constants import MAX_MODEL_SIZE, MAX_PEER_WORKERS
from app.node.store import node_store


class PeerService(peer_pb2_grpc.PeerFLServicer):
    """Handle gRPC requests for peer-to-peer model exchange and inference."""

    def __init__(self, strategy: GraphFedStrategy, user_client: DFLClient, store: ModelStore):
        self.strategy = strategy
        self.user_client = user_client
        self.store = store

    def InferenceWithModel(self, request: peer_pb2.InferenceRequest, context: grpc.ServicerContext
        ) -> peer_pb2.InferenceResponse:
        """Run inference using the provided model parameters and input data."""

        # Deserialize model parameters and run inference.
        params = [ fl.common.bytes_to_ndarray(t) for t in request.parameters ]
        inputs = request.inputs
        outputs = self.user_client.inference(params, inputs)

        # Serialize predictions for the gRPC response.
        outputs = fl.common.ndarray_to_bytes(outputs)

        return peer_pb2.InferenceResponse(outputs=outputs)

    def EvaluateWithModel(self, request: peer_pb2.EvalWithModelRequest, context: grpc.ServicerContext
        ) -> peer_pb2.EvalResponse:
        """Evaluate the provided model and return its metrics."""
        params = [fl.common.bytes_to_ndarray(t) for t in request.parameters]
        _, _, metrics = self.user_client.test(params)
        return peer_pb2.EvalResponse(metrics=metrics)
    
    def GetModel(self, request: peer_pb2.ModelRequest, context: grpc.ServicerContext
        ) -> peer_pb2.ModelResponse:
        """Return the latest model or parameters saved for a specific round."""
        if request.model_key == "latest":
            logger.info(f"[PeerService] GetModel: latest")
            model = self.store.get(request.model_key) #Parameters
            model = fl.common.parameters_to_ndarrays(model)
        else:
            logger.info(f"[PeerService] GetModel: round {request.model_key}")
            model = self.user_client.get_round_parameters(request.model_key)

        # Serialize model parameters for transmission.
        parameters = [
            fl.common.ndarray_to_bytes(arr)
            for arr in model
        ]
        return peer_pb2.ModelResponse(
            parameters=parameters
        )

    def PushModel(self, request: peer_pb2.ModelUpdate, context: grpc.ServicerContext
        ) -> peer_pb2.Ack:
        """Receive a peer's model update and pass it to the strategy."""
        try:
            # Deserialize tensors and forward the model to the strategy.
            tensors = [fl.common.bytes_to_ndarray(t) for t in request.tensors]

            self.strategy.receive_peer_model(
                rnd=request.meta.round,
                peer_id=request.meta.node_id,
                num_samples=request.meta.num_samples,
                parameters=fl.common.ndarrays_to_parameters(tensors),
            )
            return peer_pb2.Ack(ok=True)
        except:
            # Log the error and notify the sender that the update failed.
            error_details = traceback.format_exc()
            logger.error(f"PushModel error: {error_details}")
            return peer_pb2.Ack(ok=False)

    def GetNodeInfo(self, request: peer_pb2.RoundRequest, context: grpc.ServicerContext
            ) -> peer_pb2.NodeStatusResponse:
        """Return the node's status and requested round records."""
        response = peer_pb2.NodeStatusResponse(
            current_round=node_store.get_current_round(),
            status=node_store.get_status().name,
        )

        # Convert round records into the gRPC response format.
        for rnd, record in node_store.get_round_record(request.round).items():
            start, finish = "", ""
            if record.train_started_at is not None:
                start = record.train_started_at.isoformat()
            if record.train_finished_at is not None:
                finish = record.train_finished_at.isoformat()

            response.round_records[rnd].CopyFrom(
                peer_pb2.RoundRecord(
                    round=record.round,
                    train_started_at=start,
                    train_finished_at=finish,
                    send_status=record.send_status,
                    receive_status=record.receive_status,
                )
            )

        return response


def start_peer_server(strategy: GraphFedStrategy, address: str, user_client: DFLClient, model_store: ModelStore) -> grpc.Server:
    """Start the gRPC server for peer communication."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=MAX_PEER_WORKERS),
        options=[
            ("grpc.max_send_message_length", MAX_MODEL_SIZE),
            ("grpc.max_receive_message_length", MAX_MODEL_SIZE),
        ],
    )

    # Register the peer service and start listening for requests.
    peer_pb2_grpc.add_PeerFLServicer_to_server(
        PeerService(strategy, user_client, model_store), server
    )
    server.add_insecure_port(address)
    server.start()
    logger.info(f"[Peer] listening on {address}")
    return server
