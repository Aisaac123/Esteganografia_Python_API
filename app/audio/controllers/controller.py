# ------------------------------------------------------------
# ENDPOINTS PARA AUDIO
# ------------------------------------------------------------
from app.audio.service.service import AudioSteganographyEngine
from app.libs.utils import save_upload_file, calculate_audio_capacity, file_to_base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter
import os

import time

OUTPUT_DIR = "generated_audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/audio", tags=["Audio Steganography"])

from app.models.dtoAndResponses import EmbedResponse, EmbedRequest, ExtractResponse, SteganalysisResponse, MetricDetail


@router.post("/stego/embed", response_model=EmbedResponse)
async def embed_audio_message(
    audio: UploadFile = File(...),
    data: EmbedRequest = Depends()
):
    """Ocultar mensaje en audio usando LSB"""
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    input_path = save_upload_file(audio, ".wav")
    capacity = calculate_audio_capacity(input_path)
    message_size = len(data.message.encode('utf-8'))

    if message_size > capacity:
        os.unlink(input_path)
        raise HTTPException(400, f"Mensaje demasiado grande. Capacidad: {capacity} bytes, Mensaje: {message_size} bytes")

    filename = f"stego_audio_{int(time.time())}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    success = AudioSteganographyEngine.hide_message(input_path, data.message, output_path)
    os.unlink(input_path)

    if not success:
        raise HTTPException(500, "Error al ocultar mensaje en el audio")

    return EmbedResponse(
        status="success",
        message="Mensaje ocultado exitosamente en audio",
        file_base64=file_to_base64(output_path),
        payload_size=message_size,
        capacity_used=round((message_size / capacity) * 100, 2),
        file_type="audio"
    )

@router.post("/stego/extract", response_model=ExtractResponse)
async def extract_audio_message(audio: UploadFile = File(...)):
    """Extraer mensaje oculto de audio"""
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    input_path = save_upload_file(audio, ".wav")
    extracted_message = AudioSteganographyEngine.reveal_message(input_path)
    os.unlink(input_path)

    return ExtractResponse(
        status="success",
        message=extracted_message,
        message_length=len(extracted_message) if extracted_message else 0
    )

@router.post("/steganalysis/analyze", response_model=SteganalysisResponse)
async def analyze_audio(audio: UploadFile = File(...)):
    """Análisis de esteganografía en audio"""
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    tmp_path = save_upload_file(audio, ".wav")

    # Primero: Intentar extraer mensaje
    extracted_message = AudioSteganographyEngine.reveal_message(tmp_path)

    if extracted_message and len(extracted_message.strip()) > 0:
        confirmation_metric = MetricDetail(
            name="🎯 EXTRACCIÓN EXITOSA",
            value=100.0,
            explanation=f"✅ SE ENCONTRÓ MENSAJE OCULTO: '{extracted_message[:100]}{'...' if len(extracted_message) > 100 else ''}'",
            is_suspicious=True,
            severity="high",
            category="confirmation"
        )
        os.unlink(tmp_path)

        return SteganalysisResponse(
            status="success",
            is_infected=True,
            confidence=100.0,
            lsb_probability=100.0,
            verdict="🔴 INFECTADA - MENSAJE ENCONTRADO",
            metrics=[confirmation_metric],
            summary={
                "detection_method": "Direct Message Extraction",
                "message_found": True,
                "message_length": len(extracted_message),
                "file_type": "audio"
            }
        )

    # Para audio, por ahora solo tenemos extracción directa
    extraction_metric = MetricDetail(
        name="Extracción Directa",
        value=0.0,
        explanation="❌ No se pudo extraer mensaje directamente. Análisis estadístico de audio no implementado.",
        is_suspicious=False,
        severity="low",
        category="confirmation"
    )

    os.unlink(tmp_path)

    return SteganalysisResponse(
        status="success",
        is_infected=False,
        confidence=85.0,
        lsb_probability=15.0,
        verdict="🟢 LIMPIA",
        metrics=[extraction_metric],
        summary={
            "detection_method": "Direct Extraction Only",
            "message_found": False,
            "file_type": "audio",
            "note": "Análisis estadístico de audio no implementado"
        }
    )