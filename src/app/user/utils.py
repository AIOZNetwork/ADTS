import grpc
from typing import Dict
from google.protobuf.empty_pb2 import Empty
from app.topology.proto import topology_pb2, topology_pb2_grpc

def get_registered_nodes(topology_addr: str) -> Dict[str, str]:
    """Fetch registered nodes as a mapping from node IDs to peer addresses."""
    channel = grpc.insecure_channel(topology_addr)
    stub = topology_pb2_grpc.TopologyServiceStub(channel)

    resp = stub.ListNodes(Empty())
    # Convert the protobuf list into a lookup map for API handlers.
    return {n.node_id: n.peer_address for n in resp.nodes}