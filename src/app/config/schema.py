from typing import TypedDict

class NodeInfo(TypedDict):
    address: str
    weight: float

class FlowerConfig(TypedDict):
    host: str
    port: int
    rounds: int
    delay: int


class PeerConfig(TypedDict):
    host: str
    port: int


class NodeConfig(TypedDict):
    node_id: str
    flower: FlowerConfig
    peer: PeerConfig
    topology: str