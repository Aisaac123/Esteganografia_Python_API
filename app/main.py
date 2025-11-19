"""
FastAPI Steganography & Steganalysis API - MULTIFORMAT
Soporte para Imágenes + Audio
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from app.audio.controllers.controller import router as audio_router
from app.image.controllers.controller import router as image_router
from PIL import Image



from app.libs.utils import file_to_base64, save_upload_file, calculate_audio_capacity

app = FastAPI(
    title="KeyNography API",
    description="API educativa sobre la esteganografía con soporte para imágenes y audio",
    version="0.0.1"
)

os.makedirs("generated_audios", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar controladores
app.include_router(image_router)
app.include_router(audio_router)

# ------------------------------------------------------------
# ENDPOINTS GENERALES
# ------------------------------------------------------------

@app.get("/stego/capacity")
async def calculate_capacity(file: UploadFile = File(...)):
    """Calcula capacidad de almacenamiento"""
    input_path = save_upload_file(file)

    if file.content_type.startswith("image/"):
        img = Image.open(input_path)
        width, height = img.size
        channels = len(img.getbands())
        capacity_bits = width * height * channels
        capacity_bytes = capacity_bits // 8
        file_type = "image"
    elif file.content_type.startswith("audio/"):
        capacity_bytes = calculate_audio_capacity(input_path)
        file_type = "audio"
    else:
        os.unlink(input_path)
        raise HTTPException(400, "Tipo de archivo no soportado")

    capacity_kb = capacity_bytes / 1024
    os.unlink(input_path)

    return {
        "status": "success",
        "file_type": file_type,
        "capacity_bytes": capacity_bytes,
        "capacity_kb": round(capacity_kb, 2)
    }

@app.get("/")
def root():
    return {
        "status": "running",
        "version": "5.0.0 - Multiformat Edition",
        "supported_formats": ["images", "audio"],
        "endpoints": {
            "image_embed": "/image/stego/embed",
            "image_extract": "/image/stego/extract",
            "image_analysis": "/image/steganalysis/analyze",
            "audio_embed": "/audio/stego/embed",
            "audio_extract": "/audio/stego/extract",
            "audio_analysis": "/audio/steganalysis/analyze",
            "capacity": "/stego/capacity"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "supported_formats": ["PNG/BMP images", "WAV audio"],
        "extraction_first": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)