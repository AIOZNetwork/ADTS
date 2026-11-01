from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Literal, TypeVar, Generic

T = TypeVar("T")

class TrainRequest(BaseModel):
    round_id: int

class NodeInfoRequest(BaseModel):
    node_address: str
    round: int

class EvaluateRequest(BaseModel):
    model_key: str = "latest"
    nodes: Optional[List[str]] = None

class InferenceRequest(BaseModel):
    data_b64: str
    model_key: str = "lastest"
    nodes: Optional[List[str]] = None

class BaseResponse(BaseModel, Generic[T]):
    status: Literal["success", "failed"]
    data: T | None
    message: str

class RoundRecord(BaseModel):
    round: int
    train_started_at: datetime
    train_finished_at: datetime
    send_status: Dict[str, bool]
    receive_status: Dict[str, bool]

class NodeInfoData(BaseModel):
    current_round: int
    status: str
    round_records: Dict[str, RoundRecord]

NodeInfoResponse = BaseResponse[NodeInfoData]