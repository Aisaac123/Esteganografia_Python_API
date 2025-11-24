# ------------------------------------------------------------
# ENDPOINTS PARA AUDIO
# ------------------------------------------------------------
import asyncio
import base64
import concurrent.futures
from typing import List, Optional
from app.audio.service.service import AudioSteganographyEngine
from app.libs.utils import save_upload_file, calculate_audio_capacity, file_to_base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter, Form
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
    BatchExtractItem, BatchDocumentExtractItem, BatchDocumentExtractResponse
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
    """
    Análisis de esteganografía en audio que detecta:
    1. Mensajes de texto ocultos
    2. Documentos completos ocultos
    """
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    tmp_path = save_upload_file(audio, ".wav")

    # ========================================
    # PASO 1: INTENTAR EXTRAER CONTENIDO
    # ========================================
    extracted_content = AudioSteganographyEngine.reveal_message(tmp_path)

    # ========================================
    # PASO 2: VERIFICAR SI ES UN DOCUMENTO
    # ========================================
    document_found = False
    document_info = None
    is_text_message = False

    if extracted_content and len(extracted_content.strip()) > 0:
        # PRIMERO: Intentar decodificar como documento
        try:
            doc_info = extract_document_payload(extracted_content, password=None)
            # ✅ Si llegamos aquí, ES un documento
            document_found = True
            is_text_message = False
            document_info = {
                "filename": doc_info["original_filename"],
                "size": doc_info["original_size"],
                "mime_type": doc_info["mime_type"],
                "user_id": doc_info["user_id"],
                "embedded_at": doc_info["embedded_at"],
                "is_password_protected": False
            }
        except ValueError as e:
            # Si falla con error de contraseña, ES un documento protegido
            if "contraseña incorrecta" in str(e).lower() or "password" in str(e).lower() or "incorrect password" in str(
                    e).lower():
                document_found = True
                is_text_message = False
                document_info = {
                    "filename": "Documento protegido",
                    "is_password_protected": True,
                    "error": "Documento requiere contraseña para ser extraído"
                }
            else:
                # Otro tipo de error = NO es documento
                document_found = False
                is_text_message = True
        except Exception:
            # ✅ Si falla con cualquier otro error = NO es documento, es texto plano
            document_found = False
            is_text_message = True

    # ========================================
    # CASO 1: SE ENCONTRÓ UN DOCUMENTO
    # ========================================
    if document_found and document_info:
        metrics = []

        if document_info.get("is_password_protected"):
            doc_metric = MetricDetail(
                name="🔒 DOCUMENTO PROTEGIDO DETECTADO",
                value=100.0,
                explanation=f"✅ SE ENCONTRÓ UN DOCUMENTO OCULTO PROTEGIDO CON CONTRASEÑA. Se requiere la contraseña correcta para extraerlo.",
                is_suspicious=True,
                severity="high",
                category="confirmation"
            )
        else:
            doc_metric = MetricDetail(
                name="📄 DOCUMENTO EXTRAÍDO",
                value=100.0,
                explanation=f"✅ SE ENCONTRÓ DOCUMENTO OCULTO:\n"
                            f"- Nombre: {document_info['filename']}\n"
                            f"- Tamaño: {document_info['size']} bytes\n"
                            f"- Tipo: {document_info['mime_type']}\n"
                            f"- Usuario: {document_info['user_id']}\n"
                            f"- Fecha: {document_info['embedded_at']}",
                is_suspicious=True,
                severity="high",
                category="confirmation"
            )

        metrics.append(doc_metric)
        os.unlink(tmp_path)

        return SteganalysisResponse(
            status="success",
            is_infected=True,
            confidence=100.0,
            lsb_probability=100.0,
            verdict="🔴 INFECTADA - DOCUMENTO ENCONTRADO",
            metrics=metrics,
            summary={
                "detection_method": "Direct Document Extraction",
                "content_type": "document",
                "document_found": True,
                "is_password_protected": document_info.get("is_password_protected", False),
                "document_filename": document_info.get("filename"),
                "document_size": document_info.get("size"),
                "document_mime_type": document_info.get("mime_type"),
                "user_id": document_info.get("user_id"),
                "embedded_at": document_info.get("embedded_at"),
                "file_type": "audio",
                "recommendation": "✅ CONFIRMADO: El audio contiene un documento oculto. Use el endpoint /stego/extract-document para extraerlo." if not document_info.get(
                    "is_password_protected") else "🔒 El audio contiene un documento protegido. Proporcione la contraseña en /stego/extract-document."
            }
        )

    # ========================================
    # CASO 2: SE ENCONTRÓ UN MENSAJE DE TEXTO SIMPLE
    # ========================================
    if is_text_message and extracted_content and len(extracted_content.strip()) > 0:
        confirmation_metric = MetricDetail(
            name="💬 MENSAJE DE TEXTO EXTRAÍDO",
            value=100.0,
            explanation=f"✅ SE ENCONTRÓ MENSAJE OCULTO: '{extracted_content[:100]}{'...' if len(extracted_content) > 100 else ''}'",
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
                "content_type": "text",
                "message_found": True,
                "document_found": False,
                "message_length": len(extracted_content),
                "message_preview": extracted_content[:500],  # ✅ Aumentado a 500 caracteres
                "file_type": "audio"
            }
        )

    # ========================================
    # CASO 3: NO SE ENCONTRÓ CONTENIDO
    # ========================================
    extraction_metric = MetricDetail(
        name="Extracción Directa",
        value=0.0,
        explanation="❌ No se pudo extraer contenido directamente (ni mensaje ni documento). Análisis estadístico de audio no implementado.",
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
            "detection_method": "Direct Extraction Only (Text + Document)",
            "content_extraction_attempted": True,
            "message_found": False,
            "document_found": False,
            "file_type": "audio",
            "note": "Análisis estadístico de audio no implementado"
        }
    )

from app.libs.utils import (
    prepare_document_payload,
    extract_document_payload,
    save_upload_file,
    calculate_audio_capacity,
    file_to_base64
)
from app.models.dtoAndResponses import (
    DocumentEmbedResponse,
    DocumentExtractResponse
)


@router.post("/stego/embed-document", response_model=DocumentEmbedResponse)
async def embed_document_in_audio(
        audio: UploadFile = File(..., description="Audio portador (WAV)"),
        document: UploadFile = File(..., description="Documento a ocultar"),
        user_id: str = Form("Anonymous", description="ID del usuario"),
        password: Optional[str] = Form(None, description="Contraseña opcional")
):
    """
    Oculta un documento completo en audio con:
    - Compresión automática (si no es ZIP)
    - Cifrado AES-256-GCM
    - Contraseña opcional
    """
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    # Leer documento
    doc_bytes = await document.read()

    # Preparar payload (comprimir + cifrar)
    try:
        final_payload, stats = prepare_document_payload(
            doc_bytes,
            document.filename,
            document.content_type,
            (user_id or "Anonymous"),
            password
        )
    except Exception as e:
        raise HTTPException(500, f"Error al preparar documento: {str(e)}")

    # Verificar capacidad
    input_path = save_upload_file(audio, ".wav")
    capacity = calculate_audio_capacity(input_path)
    payload_size = len(final_payload.encode('utf-8'))

    if payload_size > capacity:
        os.unlink(input_path)
        raise HTTPException(
            400,
            f"Documento demasiado grande. Capacidad: {capacity} bytes, Necesario: {payload_size} bytes"
        )

    # Embeber
    filename = f"stego_doc_audio_{int(time.time())}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    success = AudioSteganographyEngine.hide_message(input_path, final_payload, output_path)
    os.unlink(input_path)

    if not success:
        raise HTTPException(500, "Error al embeber documento en audio")

    return DocumentEmbedResponse(
        status="success",
        message="Documento embebido exitosamente en audio",
        file_base64=file_to_base64(output_path),
        file_type="audio",
        original_filename=document.filename,
        original_size=stats["original_size"],
        compressed_size=stats["compressed_size"],
        payload_size=stats["payload_size"],
        capacity_used=round((payload_size / capacity) * 100, 2),
        is_password_protected=stats["is_password_protected"],
        user_id=user_id
    )


@router.post("/stego/extract-document", response_model=DocumentExtractResponse)
async def extract_document_from_audio(
        audio: UploadFile = File(..., description="Audio con documento oculto"),
        password: Optional[str] = Form(None, description="Contraseña si está protegido")
):
    """
    Extrae un documento oculto de un audio.
    Requiere contraseña si el documento fue protegido.
    """
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Solo se permiten archivos de audio")

    input_path = save_upload_file(audio, ".wav")

    try:
        # Extraer payload
        extracted_payload = AudioSteganographyEngine.reveal_message(input_path)
        os.unlink(input_path)

        if not extracted_payload:
            raise HTTPException(404, "No se encontró ningún documento oculto")

        # Descifrar y descomprimir
        try:
            doc_info = extract_document_payload(extracted_payload, password)
        except ValueError as e:
            raise HTTPException(401, str(e))

        # Retornar documento
        return DocumentExtractResponse(
            status="success",
            message="Documento extraído exitosamente",
            document_base64=base64.b64encode(doc_info["document_bytes"]).decode('utf-8'),
            original_filename=doc_info["original_filename"],
            document_size=doc_info["original_size"],
            mime_type=doc_info["mime_type"],
            user_id=doc_info["user_id"],
            embedded_at=doc_info["embedded_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(input_path):
            os.unlink(input_path)
        raise HTTPException(500, f"Error al extraer documento: {str(e)}")


@router.post("/stego/extract-document-batch", response_model=BatchDocumentExtractResponse)
async def extract_batch_documents_from_audio(
        audios: List[UploadFile] = File(...),
        password: Optional[str] = Form(None, description="Contraseña para documentos protegidos")
):
    """
    Extraer documentos ocultos de múltiples audios en paralelo

    Límite: 30 audios por request (los audios son más pesados)
    """

    if len(audios) > 30:
        raise HTTPException(400, "Máximo 30 audios por request")

    for idx, audio in enumerate(audios):
        if not audio.content_type.startswith("audio/"):
            raise HTTPException(400, f"El archivo en posición {idx} no es un audio válido")

    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_document_audio,
                idx,
                audio,
                password
            )
            for idx, audio in enumerate(audios)
        ]

        results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.status == "success" and not r.error)
    failed = sum(1 for r in results if r.error)

    return BatchDocumentExtractResponse(
        total=len(audios),
        successful=successful,
        failed=failed,
        results=results
    )


def process_single_document_audio(index: int, audio: UploadFile, password: Optional[str]) -> BatchDocumentExtractItem:
    """
    Procesa un audio individual para extraer documento
    """
    input_path = None

    try:
        input_path = save_upload_file(audio, ".wav")
        extracted_payload = AudioSteganographyEngine.reveal_message(input_path)

        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        if not extracted_payload:
            return BatchDocumentExtractItem(
                index=index,
                filename=audio.filename,
                status="success",
                notes="No se encontró ningún documento oculto"
            )

        try:
            doc_info = extract_document_payload(extracted_payload, password)
        except ValueError as e:
            return BatchDocumentExtractItem(
                index=index,
                filename=audio.filename,
                status="error",
                error=str(e)
            )

        return BatchDocumentExtractItem(
            index=index,
            filename=audio.filename,
            status="success",
            document_base64=base64.b64encode(doc_info["document_bytes"]).decode('utf-8'),
            original_filename=doc_info["original_filename"],
            document_size=doc_info["original_size"],
            mime_type=doc_info["mime_type"],
            user_id=doc_info["user_id"],
            embedded_at=doc_info["embedded_at"]
        )

    except Exception as e:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        return BatchDocumentExtractItem(
            index=index,
            filename=audio.filename,
            status="error",
            error=str(e)
        )