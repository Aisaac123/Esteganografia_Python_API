# ------------------------------------------------------------
# ENDPOINTS PARA AUDIO
# ------------------------------------------------------------
import asyncio
import concurrent.futures
from typing import List
from app.audio.service.service import AudioSteganographyEngine
from app.libs.utils import save_upload_file, calculate_audio_capacity, file_to_base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter
import os
import time

OUTPUT_DIR = "generated_audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/audio", tags=["Audio Steganography"])

from app.models.dtoAndResponses import (
    EmbedResponse,
    EmbedRequest,
    ExtractResponse,
    SteganalysisResponse,
    MetricDetail,
    BatchExtractResponse,
    BatchExtractItem
)


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

    if extracted_message is None or extracted_message == "":
        return ExtractResponse(
            status="success",
            message="",
            message_length=0,
            notes="No se encontró ningún mensaje oculto en el audio"
        )

    return ExtractResponse(
        status="success",
        message=extracted_message,
        message_length=len(extracted_message)
    )


# ========================================
# NUEVO: ENDPOINT DE EXTRACCIÓN POR LOTES (BATCH) PARA AUDIO
# ========================================
@router.post("/stego/extract-batch", response_model=BatchExtractResponse)
async def extract_batch_audio_messages(audios: List[UploadFile] = File(...)):
    """
    Extraer mensajes ocultos de múltiples audios en paralelo

    Límite recomendado: 30 audios por request (son más pesados que imágenes)
    """

    # Validar límite de audios
    if len(audios) > 30:
        raise HTTPException(400, "Máximo 30 audios por request")

    # Validar que todos sean audios
    for idx, audio in enumerate(audios):
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(400, f"El archivo en posición {idx} no es un audio válido")

    # Procesar en paralelo usando ThreadPoolExecutor
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Crear tareas para cada audio (menos workers porque audio es más pesado)
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_audio,
                idx,
                audio
            )
            for idx, audio in enumerate(audios)
        ]

        # Esperar todas las tareas
        results = await asyncio.gather(*tasks)

    # Contar éxitos y fallos
    successful = sum(1 for r in results if r.status == "success" and not r.error)
    failed = sum(1 for r in results if r.error)

    return BatchExtractResponse(
        total=len(audios),
        successful=successful,
        failed=failed,
        results=results
    )


# ========================================
# FUNCIÓN AUXILIAR: Procesar un audio individual
# ========================================
def process_single_audio(index: int, audio: UploadFile) -> BatchExtractItem:
    """
    Procesa un audio individual de forma síncrona
    Esta función se ejecutará en un thread separado
    """
    input_path = None

    try:
        # Guardar archivo temporal
        input_path = save_upload_file(audio, ".wav")

        # Extraer mensaje
        extracted_message = AudioSteganographyEngine.reveal_message(input_path)

        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        # Sin mensaje oculto
        if extracted_message is None or extracted_message == "":
            return BatchExtractItem(
                index=index,
                filename=audio.filename,
                status="success",
                message="",
                message_length=0,
                notes="No se encontró ningún mensaje oculto"
            )

        # Mensaje extraído exitosamente
        return BatchExtractItem(
            index=index,
            filename=audio.filename,
            status="success",
            message=extracted_message,
            message_length=len(extracted_message)
        )

    except Exception as e:
        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        return BatchExtractItem(
            index=index,
            filename=audio.filename,
            status="error",
            message="",
            message_length=0,
            error=str(e)
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