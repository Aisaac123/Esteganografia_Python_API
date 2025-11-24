import asyncio
import base64
import concurrent
from typing import List

from fastapi import UploadFile, File, HTTPException, Depends, APIRouter
import os

import time
import tempfile
from PIL import Image

# Steganografía para imágenes
from stegano import lsb

from app.image.service.service import AdvancedSteganalysisEngine
from app.libs.utils import save_upload_file, calculate_image_capacity, file_to_base64
from app.models.dtoAndResponses import EmbedRequest, EmbedResponse, ExtractResponse, SteganalysisResponse, MetricDetail, \
    BatchExtractResponse, BatchExtractItem, BatchDocumentExtractResponse, BatchDocumentExtractItem

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/image", tags=["Audio Steganography"])

@router.post("/stego/embed", response_model=EmbedResponse)

async def embed_image_message(
    image: UploadFile = File(...),
    data: EmbedRequest = Depends()
):
    """Ocultar mensaje en imagen usando LSB"""
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

    input_path = save_upload_file(image)
    capacity = calculate_image_capacity(input_path)
    message_size = len(data.message.encode('utf-8'))

    if message_size > capacity:
        os.unlink(input_path)
        raise HTTPException(400, f"Mensaje demasiado grande. Capacidad: {capacity} bytes, Mensaje: {message_size} bytes")

    filename = f"stego_image_{int(time.time())}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    secret_img = lsb.hide(input_path, data.message)
    secret_img.save(output_path)
    os.unlink(input_path)

    return EmbedResponse(
        status="success",
        message="Mensaje ocultado exitosamente en imagen",
        file_base64=file_to_base64(output_path),
        payload_size=message_size,
        capacity_used=round((message_size / capacity) * 100, 2),
        file_type="image"
    )


@router.post("/stego/extract", response_model=ExtractResponse)
async def extract_image_message(image: UploadFile = File(...)):
    """Extraer mensaje oculto de imagen"""
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

    input_path = save_upload_file(image)

    try:
        extracted_message = lsb.reveal(input_path)

        if extracted_message is None or extracted_message == "":
            return ExtractResponse(
                status="success",
                message="",
                message_length=0,
                notes="No se encontró ningún mensaje oculto en la imagen"
            )

        os.unlink(input_path)

        return ExtractResponse(
            status="success",
            message=extracted_message,
            message_length=len(extracted_message)
        )

    except IndexError as e:
        os.unlink(input_path)
        return ExtractResponse(
            status="success",
            message="",
            message_length=0,
            notes="La imagen no contiene un mensaje oculto o está corrupta"
        )
    except Exception as e:
        os.unlink(input_path)
        print(f"Error extracting message: {str(e)}")
        raise HTTPException(500, f"Error al procesar la imagen: {str(e)}")


# ========================================
# NUEVO: ENDPOINT DE EXTRACCIÓN POR LOTES (BATCH)
# ========================================
@router.post("/stego/extract-batch", response_model=BatchExtractResponse)
async def extract_batch_messages(images: List[UploadFile] = File(...)):
    """
    Extraer mensajes ocultos de múltiples imágenes en paralelo

    Límite recomendado: 50 imágenes por request
    """

    # Validar límite de imágenes
    if len(images) > 50:
        raise HTTPException(400, "Máximo 50 imágenes por request")

    # Validar que todas sean imágenes
    for idx, image in enumerate(images):
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, f"El archivo en posición {idx} no es una imagen válida")

    # Procesar en paralelo usando ThreadPoolExecutor
    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Crear tareas para cada imagen
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_image,
                idx,
                image
            )
            for idx, image in enumerate(images)
        ]

        # Esperar todas las tareas
        results = await asyncio.gather(*tasks)

    # Contar éxitos y fallos
    successful = sum(1 for r in results if r.status == "success" and not r.error)
    failed = sum(1 for r in results if r.error)

    return BatchExtractResponse(
        total=len(images),
        successful=successful,
        failed=failed,
        results=results
    )


# ========================================
# FUNCIÓN AUXILIAR: Procesar una imagen individual
# ========================================
def process_single_image(index: int, image: UploadFile) -> BatchExtractItem:
    """
    Procesa una imagen individual de forma síncrona
    Esta función se ejecutará en un thread separado
    """
    input_path = None

    try:
        # Guardar archivo temporal
        input_path = save_upload_file(image)

        # Extraer mensaje
        extracted_message = lsb.reveal(input_path)

        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        # Sin mensaje oculto
        if extracted_message is None or extracted_message == "":
            return BatchExtractItem(
                index=index,
                filename=image.filename,
                status="success",
                message="",
                message_length=0,
                notes="No se encontró ningún mensaje oculto"
            )

        # Mensaje extraído exitosamente
        return BatchExtractItem(
            index=index,
            filename=image.filename,
            status="success",
            message=extracted_message,
            message_length=len(extracted_message)
        )

    except IndexError:
        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        return BatchExtractItem(
            index=index,
            filename=image.filename,
            status="success",
            message="",
            message_length=0,
            notes="La imagen no contiene un mensaje oculto o está corrupta"
        )

    except Exception as e:
        # Limpiar archivo temporal
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        return BatchExtractItem(
            index=index,
            filename=image.filename,
            status="error",
            message="",
            message_length=0,
            error=str(e)
        )


@router.post("/steganalysis/analyze", response_model=SteganalysisResponse)
async def analyze_image(image: UploadFile = File(...)):
    """
    Sistema de detección mejorado que intenta extraer:
    1. Mensajes de texto ocultos
    2. Documentos completos ocultos
    Si encuentra contenido → 100% de confianza
    Si no → análisis estadístico normal
    """

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

    # Guardar imagen temporal
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    with open(tmp_path, "wb") as f:
        f.write(await image.read())

    # ========================================
    # PASO 1: INTENTAR EXTRAER CONTENIDO
    # ========================================
    extracted_content = None
    try:
        extracted_content = lsb.reveal(tmp_path)
    except Exception:
        extracted_content = None

    # ========================================
    # PASO 2: VERIFICAR SI ES UN DOCUMENTO OCULTO
    # ========================================
    document_found = False
    document_info = None
    is_text_message = False

    if extracted_content and len(extracted_content.strip()) > 0:
        # PRIMERO: Intentar decodificar como documento (tiene estructura específica)
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
                # Otro tipo de error (formato inválido, etc.) = NO es documento
                document_found = False
                is_text_message = True
        except Exception as ex:
            # ✅ Si falla con cualquier otro error (JSON decode, etc.) = NO es documento, es texto plano
            document_found = False
            is_text_message = True

    # ========================================
    # CASO 1: SE ENCONTRÓ UN DOCUMENTO
    # ========================================
    if document_found and document_info:
        metrics = []

        # Métrica de documento encontrado
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
                "recommendation": "✅ CONFIRMADO: La imagen contiene un documento oculto. Use el endpoint /stego/extract-document para extraerlo." if not document_info.get(
                    "is_password_protected") else "🔒 La imagen contiene un documento protegido. Proporcione la contraseña en /stego/extract-document."
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
                "recommendation": "✅ CONFIRMADO: La imagen contiene un mensaje de texto oculto."
            }
        )

    # ========================================
    # CASO 3: NO SE ENCONTRÓ CONTENIDO - ANÁLISIS ESTADÍSTICO
    # ========================================
    img = Image.open(tmp_path).convert("RGB")

    # Métricas críticas
    critical_metrics = [
        AdvancedSteganalysisEngine.advanced_lsb_detector(tmp_path),
        AdvancedSteganalysisEngine.enhanced_sample_pair(tmp_path),
    ]

    # Métricas secundarias
    secondary_metrics = [
        AdvancedSteganalysisEngine.channel_uniformity_detector(tmp_path),
    ]

    # Métricas de soporte
    support_metrics = [
        AdvancedSteganalysisEngine.chi_square_attack(tmp_path),
        AdvancedSteganalysisEngine.histogram_attack(tmp_path),
        AdvancedSteganalysisEngine.rs_analysis(tmp_path),
        AdvancedSteganalysisEngine.pixel_difference_analysis(tmp_path),
        AdvancedSteganalysisEngine.noise_index(tmp_path),
        AdvancedSteganalysisEngine.fourier_spectrum_consistency(tmp_path),
    ]

    os.unlink(tmp_path)

    # Lógica de decisión
    critical_suspicious = sum(1 for m in critical_metrics if m.is_suspicious)
    secondary_suspicious = sum(1 for m in secondary_metrics if m.is_suspicious)

    if critical_suspicious == 2:
        is_infected = True
        confidence = 95.0
        verdict = "🔴 INFECTADA (Alta Confianza)"
        recommendation = "⚠️ ALERTA MÁXIMA: Las 2 métricas críticas detectaron esteganografía LSB. Probabilidad >95% de que contenga contenido oculto (mensaje o documento)."

    elif critical_suspicious == 1 and secondary_suspicious >= 1:
        is_infected = True
        confidence = 85.0
        verdict = "🔴 INFECTADA (Confianza Media-Alta)"
        recommendation = "⚠️ ALERTA: Detección positiva en métrica crítica confirmada por métrica secundaria. Probabilidad >85% de esteganografía."

    elif critical_suspicious == 1:
        is_infected = True
        confidence = 70.0
        verdict = "🟡 SOSPECHOSA"
        recommendation = "⚠️ PRECAUCIÓN: Una métrica crítica detectó anomalías. Recomendado análisis adicional."

    else:
        is_infected = False
        confidence = 90.0
        verdict = "🟢 LIMPIA"
        recommendation = "✓ La imagen pasó todas las verificaciones críticas. No se detectaron signos de esteganografía LSB."

    lsb_probability = confidence if is_infected else (100 - confidence)

    all_metrics = critical_metrics + secondary_metrics + support_metrics

    # Métrica de extracción fallida
    extraction_metric = MetricDetail(
        name="Extracción Directa",
        value=0.0,
        explanation="❌ No se pudo extraer contenido directamente (ni mensaje ni documento). Procediendo con análisis estadístico.",
        is_suspicious=False,
        severity="low",
        category="confirmation"
    )
    all_metrics.insert(0, extraction_metric)

    summary = {
        "detection_method": "Hybrid: Extraction (Text + Document) + Statistical Analysis",
        "content_extraction_attempted": True,
        "message_found": False,
        "document_found": False,
        "critical_suspicious": critical_suspicious,
        "critical_total": len(critical_metrics),
        "secondary_suspicious": secondary_suspicious,
        "secondary_total": len(secondary_metrics),
        "support_info": f"{len(support_metrics)} additional metrics computed",
        "recommendation": recommendation,
        "confidence_level": "Very High" if confidence > 90 else "High" if confidence > 80 else "Medium"
    }

    return SteganalysisResponse(
        status="success",
        is_infected=is_infected,
        confidence=confidence,
        lsb_probability=lsb_probability,
        verdict=verdict,
        metrics=all_metrics,
        summary=summary
    )

from app.libs.utils import (
    prepare_document_payload,
    extract_document_payload,
    save_upload_file,
    calculate_image_capacity,
    file_to_base64
)
from app.models.dtoAndResponses import (
    DocumentEmbedResponse,
    DocumentExtractResponse
)
from fastapi import Form
from typing import Optional


@router.post("/stego/embed-document", response_model=DocumentEmbedResponse)
async def embed_document_in_image(
        image: UploadFile = File(..., description="Imagen portadora"),
        document: UploadFile = File(..., description="Documento a ocultar"),
        user_id: str = Form("Anonymous", description="ID del usuario"),
        password: Optional[str] = Form(None, description="Contraseña opcional")
):
    """
    Oculta un documento completo en una imagen con:
    - Compresión automática (si no es ZIP)
    - Cifrado AES-256-GCM
    - Contraseña opcional
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

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
    input_path = save_upload_file(image)
    capacity = calculate_image_capacity(input_path)
    payload_size = len(final_payload.encode('utf-8'))

    if payload_size > capacity:
        os.unlink(input_path)
        raise HTTPException(
            400,
            f"Documento demasiado grande. Capacidad: {capacity} bytes, Necesario: {payload_size} bytes"
        )

    # Embeber con LSB
    filename = f"stego_doc_{int(time.time())}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    secret_img = lsb.hide(input_path, final_payload)
    secret_img.save(output_path)
    os.unlink(input_path)

    return DocumentEmbedResponse(
        status="success",
        message="Documento embebido exitosamente en imagen",
        file_base64=file_to_base64(output_path),
        file_type="image",
        original_filename=document.filename,
        original_size=stats["original_size"],
        compressed_size=stats["compressed_size"],
        payload_size=stats["payload_size"],
        capacity_used=round((payload_size / capacity) * 100, 2),
        is_password_protected=stats["is_password_protected"],
        user_id=user_id
    )


@router.post("/stego/extract-document", response_model=DocumentExtractResponse)
async def extract_document_from_image(
        image: UploadFile = File(..., description="Imagen con documento oculto"),
        password: Optional[str] = Form(None, description="Contraseña si está protegido")
):
    """
    Extrae un documento oculto de una imagen.
    Requiere contraseña si el documento fue protegido.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

    input_path = save_upload_file(image)

    try:
        # Extraer payload
        extracted_payload = lsb.reveal(input_path)
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
async def extract_batch_documents(
        images: List[UploadFile] = File(...),
        password: Optional[str] = Form(None, description="Contraseña para documentos protegidos")
):
    """
    Extraer documentos ocultos de múltiples imágenes en paralelo

    Límite: 50 imágenes por request
    """

    if len(images) > 50:
        raise HTTPException(400, "Máximo 50 imágenes por request")

    for idx, image in enumerate(images):
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, f"El archivo en posición {idx} no es una imagen válida")

    loop = asyncio.get_event_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_document_image,
                idx,
                image,
                password
            )
            for idx, image in enumerate(images)
        ]

        results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r.status == "success" and not r.error)
    failed = sum(1 for r in results if r.error)

    return BatchDocumentExtractResponse(
        total=len(images),
        successful=successful,
        failed=failed,
        results=results
    )


def process_single_document_image(index: int, image: UploadFile, password: Optional[str]) -> BatchDocumentExtractItem:
    """
    Procesa una imagen individual para extraer documento
    """
    input_path = None

    try:
        input_path = save_upload_file(image)
        extracted_payload = lsb.reveal(input_path)

        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        if not extracted_payload:
            return BatchDocumentExtractItem(
                index=index,
                filename=image.filename,
                status="success",
                notes="No se encontró ningún documento oculto"
            )

        try:
            doc_info = extract_document_payload(extracted_payload, password)
        except ValueError as e:
            return BatchDocumentExtractItem(
                index=index,
                filename=image.filename,
                status="error",
                error=str(e)
            )

        return BatchDocumentExtractItem(
            index=index,
            filename=image.filename,
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
            filename=image.filename,
            status="error",
            error=str(e)
        )