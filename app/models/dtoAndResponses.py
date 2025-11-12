from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class EmbedRequest(BaseModel):
    message: str

class EmbedResponse(BaseModel):
    status: str
    message: str
    file_base64: str
    payload_size: int
    capacity_used: float
    file_type: str

class ExtractResponse(BaseModel):
    status: str
    message: Optional[str]
    message_length: int

class MetricDetail(BaseModel):
    name: str
    value: float
    explanation: str
    is_suspicious: bool
    severity: str
    category: str = "confirmation"

class SteganalysisResponse(BaseModel):
    status: str
    is_infected: bool
    confidence: float
    lsb_probability: float
    verdict: str
    metrics: List[MetricDetail]
    summary: Dict[str, Any]