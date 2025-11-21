import asyncio
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
    BatchExtractResponse, BatchExtractItem

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


# ========================================
# FUNCIÓN AUXILIAR: Guardar archivo subido
# ========================================
def save_upload_file(upload_file: UploadFile) -> str:
    """Guarda el archivo subido temporalmente y retorna la ruta"""
    import tempfile
    import shutil

    # Crear archivo temporal
    suffix = os.path.splitext(upload_file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

@router.post("/steganalysis/analyze", response_model=SteganalysisResponse)
async def analyze_image(image: UploadFile = File(...)):
    """
    Sistema de detección mejorado: Primero intenta extraer mensaje
    Si encuentra mensaje → 100% de confianza
    Si no → análisis estadístico normal
    """

    if not image.content_type.startswith("image/"):
        raise HTTPException(400, "Solo se permiten imágenes")

    # Guardar imagen temporal
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    with open(tmp_path, "wb") as f:
        f.write(await image.read())

    # ========================================
    # PRIMERO: INTENTAR EXTRAER MENSAJE
    # ========================================
    extracted_message = None
    try:
        extracted_message = lsb.reveal(tmp_path)
    except Exception:
        extracted_message = None

    # Si se encontró un mensaje, dar 100% de confianza
    if extracted_message and len(extracted_message.strip()) > 0:
        # Crear métrica de confirmación
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
                "message_preview": extracted_message[:100],
                "recommendation": "✅ CONFIRMADO: La imagen contiene un mensaje oculto extraíble."
            }
        )

    # ========================================
    # SEGUNDO: ANÁLISIS ESTADÍSTICO (si no se encontró mensaje)
    # ========================================
    img = Image.open(tmp_path).convert("RGB")

    # ========================================
    # MÉTRICAS CRÍTICAS (las que deciden)
    # ========================================
    critical_metrics = [
        AdvancedSteganalysisEngine.advanced_lsb_detector(tmp_path),
        AdvancedSteganalysisEngine.enhanced_sample_pair(tmp_path),
    ]

    # ========================================
    # MÉTRICAS SECUNDARIAS (confirmación)
    # ========================================
    secondary_metrics = [
        AdvancedSteganalysisEngine.channel_uniformity_detector(tmp_path),
    ]

    # ========================================
    # MÉTRICAS DE SOPORTE (solo informativas)
    # ========================================
    support_metrics = [
        AdvancedSteganalysisEngine.chi_square_attack(tmp_path),
        AdvancedSteganalysisEngine.histogram_attack(tmp_path),
        AdvancedSteganalysisEngine.rs_analysis(tmp_path),
        AdvancedSteganalysisEngine.pixel_difference_analysis(tmp_path),
        AdvancedSteganalysisEngine.noise_index(tmp_path),
        AdvancedSteganalysisEngine.fourier_spectrum_consistency(tmp_path),
    ]

    os.unlink(tmp_path)

    # ========================================
    # LÓGICA DE DECISIÓN
    # ========================================

    # Contar críticas sospechosas
    critical_suspicious = sum(1 for m in critical_metrics if m.is_suspicious)

    # Contar secundarias sospechosas
    secondary_suspicious = sum(1 for m in secondary_metrics if m.is_suspicious)

    # REGLAS DE DECISIÓN:
    # 1. Si AMBAS críticas son sospechosas → INFECTADA (confianza 95%)
    # 2. Si 1 crítica + 1 secundaria sospechosas → INFECTADA (confianza 85%)
    # 3. Si solo 1 crítica sospechosa → SOSPECHOSA (confianza 70%)
    # 4. Resto → LIMPIA

    if critical_suspicious == 2:
        is_infected = True
        confidence = 95.0
        verdict = "🔴 INFECTADA (Alta Confianza)"
        recommendation = "⚠️ ALERTA MÁXIMA: Las 2 métricas críticas detectaron esteganografía LSB. Probabilidad >95% de que contenga mensaje oculto."

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

    # Calcular probabilidad LSB
    lsb_probability = confidence if is_infected else (100 - confidence)

    # Combinar todas las métricas para mostrar
    all_metrics = critical_metrics + secondary_metrics + support_metrics

    # Agregar métrica de extracción fallida
    extraction_metric = MetricDetail(
        name="Extracción Directa",
        value=0.0,
        explanation="❌ No se pudo extraer mensaje directamente. Procediendo con análisis estadístico.",
        is_suspicious=False,
        severity="low",
        category="confirmation"
    )
    all_metrics.insert(0, extraction_metric)

    # Estadísticas
    summary = {
        "detection_method": "Hybrid: Extraction + Statistical Analysis",
        "message_extraction_attempted": True,
        "message_found": False,
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