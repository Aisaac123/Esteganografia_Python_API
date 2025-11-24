# ------------------------------------------------------------
# ROUTER UNIFICADO PARA EXTRACCIÓN BATCH GLOBAL
# ------------------------------------------------------------
import asyncio
import base64
import concurrent.futures
from typing import List, Optional
from fastapi import UploadFile, File, HTTPException, APIRouter, Form
import os

from stegano import lsb
from app.audio.service.service import AudioSteganographyEngine
from app.libs.utils import (
    save_upload_file,
    extract_document_payload
)
from app.models.dtoAndResponses import (
    BatchExtractItem,
    BatchDocumentExtractItem
)
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["Unified Steganography"])


# ========================================
# MODELOS DE RESPUESTA
# ========================================
class UnifiedExtractItem(BaseModel):
    """Resultado de extracción unificada para un archivo individual"""
    index: int
    filename: str
    file_type: str  # "image" o "audio"
    status: str  # "success" o "error"

    # Para mensajes de texto
    content_type: Optional[str] = None  # "text", "document", o None
    message: Optional[str] = None
    message_length: Optional[int] = None

    # Para documentos
    document_found: bool = False
    document_base64: Optional[str] = None
    original_filename: Optional[str] = None
    document_size: Optional[int] = None
    mime_type: Optional[str] = None
    user_id: Optional[str] = None
    embedded_at: Optional[str] = None
    is_password_protected: bool = False

    # Información adicional
    notes: Optional[str] = None
    error: Optional[str] = None


class UnifiedBatchExtractResponse(BaseModel):
    """Respuesta de extracción batch unificada"""
    total: int
    successful: int
    failed: int
    images_processed: int
    audios_processed: int
    messages_found: int
    documents_found: int
    results: List[UnifiedExtractItem]


# ========================================
# ENDPOINT PRINCIPAL: EXTRACCIÓN BATCH UNIFICADA
# ========================================
@router.post("/extract-batch", response_model=UnifiedBatchExtractResponse)
async def unified_batch_extract(
        files: List[UploadFile] = File(...),
        password: Optional[str] = Form(None, description="Contraseña para documentos protegidos")
):
    """
    🔥 EXTRACCIÓN BATCH UNIFICADA 🔥

    Procesa múltiples archivos (imágenes y audios) en paralelo y extrae:
    - Mensajes de texto ocultos
    - Documentos completos ocultos

    Detecta automáticamente el tipo de archivo y aplica el método correcto.

    Límites:
    - Máximo 50 archivos por request
    - Soporta: PNG, JPG, JPEG, WAV
    """

    # Validar límite
    if len(files) > 50:
        raise HTTPException(400, "Máximo 50 archivos por request")

    # Validar tipos de archivo
    valid_image_types = ["image/png", "image/jpeg", "image/jpg"]
    valid_audio_types = ["audio/wav", "audio/x-wav", "audio/wave"]

    for idx, file in enumerate(files):
        is_image = any(file.content_type.startswith(t.split('/')[0]) for t in valid_image_types)
        is_audio = any(file.content_type.startswith(t.split('/')[0]) for t in valid_audio_types)

        if not is_image and not is_audio:
            raise HTTPException(
                400,
                f"Archivo en posición {idx} tiene tipo inválido: {file.content_type}. "
                f"Soportados: PNG, JPG, JPEG, WAV"
            )

    # Procesar en paralelo
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                process_unified_file,
                idx,
                file,
                password
            )
            for idx, file in enumerate(files)
        ]

        results = await asyncio.gather(*tasks)

    # Calcular estadísticas
    successful = sum(1 for r in results if r.status == "success" and not r.error)
    failed = sum(1 for r in results if r.error)
    images_processed = sum(1 for r in results if r.file_type == "image")
    audios_processed = sum(1 for r in results if r.file_type == "audio")
    messages_found = sum(1 for r in results if r.content_type == "text")
    documents_found = sum(1 for r in results if r.document_found)

    return UnifiedBatchExtractResponse(
        total=len(files),
        successful=successful,
        failed=failed,
        images_processed=images_processed,
        audios_processed=audios_processed,
        messages_found=messages_found,
        documents_found=documents_found,
        results=results
    )


# ========================================
# FUNCIÓN AUXILIAR: PROCESAR ARCHIVO UNIFICADO
# ========================================
def process_unified_file(
        index: int,
        file: UploadFile,
        password: Optional[str]
) -> UnifiedExtractItem:
    """
    Procesa un archivo individual (imagen o audio) y extrae contenido oculto.

    Flujo:
    1. Detecta tipo de archivo (imagen/audio)
    2. Extrae contenido con el método apropiado
    3. Intenta decodificar como documento
    4. Si falla, lo trata como mensaje de texto
    """
    input_path = None

    try:
        # ========================================
        # PASO 1: DETECTAR TIPO DE ARCHIVO
        # ========================================
        is_image = file.content_type.startswith("image/")
        is_audio = file.content_type.startswith("audio/")

        file_type = "image" if is_image else "audio"

        # ========================================
        # PASO 2: GUARDAR ARCHIVO TEMPORAL
        # ========================================
        if is_audio:
            input_path = save_upload_file(file, ".wav")
        else:
            input_path = save_upload_file(file)

        # ========================================
        # PASO 3: EXTRAER CONTENIDO
        # ========================================
        extracted_content = None

        if is_image:
            try:
                extracted_content = lsb.reveal(input_path)
            except Exception:
                extracted_content = None
        else:  # audio
            extracted_content = AudioSteganographyEngine.reveal_message(input_path)

        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        # ========================================
        # PASO 4: SI NO HAY CONTENIDO
        # ========================================
        if not extracted_content or len(extracted_content.strip()) == 0:
            return UnifiedExtractItem(
                index=index,
                filename=file.filename,
                file_type=file_type,
                status="success",
                content_type=None,
                notes=f"No se encontró contenido oculto en el {file_type}"
            )

        # ========================================
        # PASO 5: INTENTAR DECODIFICAR COMO DOCUMENTO
        # ========================================
        try:
            doc_info = extract_document_payload(extracted_content, password)

            # ✅ ES UN DOCUMENTO
            return UnifiedExtractItem(
                index=index,
                filename=file.filename,
                file_type=file_type,
                status="success",
                content_type="document",
                document_found=True,
                document_base64=base64.b64encode(doc_info["document_bytes"]).decode('utf-8'),
                original_filename=doc_info["original_filename"],
                document_size=doc_info["original_size"],
                mime_type=doc_info["mime_type"],
                user_id=doc_info["user_id"],
                embedded_at=doc_info["embedded_at"],
                is_password_protected=False
            )

        except ValueError as e:
            # ========================================
            # PASO 6: VERIFICAR SI ES DOCUMENTO PROTEGIDO
            # ========================================
            error_msg = str(e).lower()
            if "contraseña incorrecta" in error_msg or "password" in error_msg or "incorrect password" in error_msg:
                return UnifiedExtractItem(
                    index=index,
                    filename=file.filename,
                    file_type=file_type,
                    status="success",
                    content_type="document",
                    document_found=True,
                    is_password_protected=True,
                    notes="Documento protegido con contraseña. Proporcione la contraseña correcta."
                )
            else:
                # Otro tipo de ValueError = NO es documento, pasar a texto
                pass

        except Exception:
            # Cualquier otro error = NO es documento, pasar a texto
            pass

        # ========================================
        # PASO 7: ES UN MENSAJE DE TEXTO
        # ========================================
        return UnifiedExtractItem(
            index=index,
            filename=file.filename,
            file_type=file_type,
            status="success",
            content_type="text",
            message=extracted_content,
            message_length=len(extracted_content),
            document_found=False
        )

    except Exception as e:
        # ========================================
        # ERROR GENERAL
        # ========================================
        # Limpiar archivo temporal si existe
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        # Determinar tipo de archivo
        file_type = "image" if file.content_type.startswith("image/") else "audio"

        return UnifiedExtractItem(
            index=index,
            filename=file.filename,
            file_type=file_type,
            status="error",
            error=str(e)
        )


# ========================================
# ENDPOINT ADICIONAL: ESTADÍSTICAS RÁPIDAS
# ========================================
@router.post("/quick-scan")
async def quick_scan_files(files: List[UploadFile] = File(...)):
    """
    🔍 ESCANEO RÁPIDO

    Analiza archivos sin extraer contenido completo.
    Solo verifica si tienen contenido oculto (Sí/No).

    Útil para validaciones rápidas de grandes lotes.
    """

    if len(files) > 100:
        raise HTTPException(400, "Máximo 100 archivos para escaneo rápido")

    results = []

    for idx, file in enumerate(files):
        input_path = None
        try:
            is_audio = file.content_type.startswith("audio/")
            file_type = "audio" if is_audio else "image"

            # Guardar temporal
            if is_audio:
                input_path = save_upload_file(file, ".wav")
                content = AudioSteganographyEngine.reveal_message(input_path)
            else:
                input_path = save_upload_file(file)
                content = lsb.reveal(input_path)

            # Limpiar
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)

            has_content = content and len(content.strip()) > 0

            results.append({
                "index": idx,
                "filename": file.filename,
                "file_type": file_type,
                "has_hidden_content": has_content
            })

        except Exception as e:
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)

            results.append({
                "index": idx,
                "filename": file.filename,
                "file_type": "unknown",
                "has_hidden_content": False,
                "error": str(e)
            })

    files_with_content = sum(1 for r in results if r.get("has_hidden_content"))

    return {
        "total": len(files),
        "files_with_content": files_with_content,
        "files_clean": len(files) - files_with_content,
        "results": results
    }