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

# Modelos de respuesta
class ExtractResponse(BaseModel):
    status: str
    message: str
    message_length: int
    notes: str = None

class BatchExtractItem(BaseModel):
    index: int
    filename: str
    status: str
    message: str
    message_length: int
    notes: str = None
    error: str = None

class BatchExtractResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[BatchExtractItem]

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

from pydantic import BaseModel
from typing import Optional

class DocumentEmbedRequest(BaseModel):
    user_id: str
    password: Optional[str] = None

class DocumentEmbedResponse(BaseModel):
    status: str
    message: str
    file_base64: str
    file_type: str  # "image" o "audio"
    original_filename: str
    original_size: int
    compressed_size: int
    payload_size: int
    capacity_used: float
    is_password_protected: bool
    user_id: str

class DocumentExtractResponse(BaseModel):
    status: str
    message: str
    document_base64: str
    original_filename: str
    document_size: int
    mime_type: str
    user_id: str
    embedded_at: str

class BatchDocumentExtractItem(BaseModel):
    index: int
    filename: str
    status: str
    document_base64: Optional[str] = None
    original_filename: Optional[str] = None
    document_size: Optional[int] = None
    mime_type: Optional[str] = None
    user_id: Optional[str] = None
    embedded_at: Optional[str] = None
    notes: Optional[str] = None
    error: Optional[str] = None


class BatchDocumentExtractResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[BatchDocumentExtractItem]