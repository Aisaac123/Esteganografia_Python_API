import base64
import tempfile
import wave

from fastapi import UploadFile
from PIL import Image


def file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def save_upload_file(upload_file: UploadFile, suffix: str = ".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload_file.file.read())
    tmp.close()
    return tmp.name

def calculate_image_capacity(image_path: str) -> int:
    img = Image.open(image_path)
    width, height = img.size
    return (width * height * 3) // 8

def calculate_audio_capacity(audio_path: str) -> int:
    """Calcula capacidad de almacenamiento en audio WAV"""
    try:
        with wave.open(audio_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            sample_width = audio_file.getsampwidth()
            return (frames * sample_width) // 8
    except Exception:
        return 0