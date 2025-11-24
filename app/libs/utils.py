import base64
import tempfile
import wave
import zipfile
import io
import os
import json
from typing import Optional, Tuple
from fastapi import UploadFile
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


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


# ========================================
# NUEVAS UTILIDADES PARA DOCUMENTOS
# ========================================

class CryptoEngine:
    """Motor de cifrado AES-256-GCM para esteganografía de documentos"""

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Deriva una clave de 256 bits desde una contraseña usando PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode('utf-8'))

    @staticmethod
    def encrypt(data: bytes, password: Optional[str] = None) -> dict:
        """
        Cifra datos con AES-256-GCM
        Si hay password: usa PBKDF2 para derivar la clave
        Si no: usa clave maestra fija (para cifrado base sin password)
        """
        salt = os.urandom(16)
        iv = os.urandom(12)

        if password:
            key = CryptoEngine.derive_key(password, salt)
        else:
            master_password = "STEGO_MASTER_KEY_2024_SECURE"
            key = CryptoEngine.derive_key(master_password, salt)

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        return {
            "salt": base64.b64encode(salt).decode('utf-8'),
            "iv": base64.b64encode(iv).decode('utf-8'),
            "tag": base64.b64encode(encryptor.tag).decode('utf-8'),
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8')
        }

    @staticmethod
    def decrypt(encrypted_data: dict, password: Optional[str] = None) -> bytes:
        """Descifra datos con AES-256-GCM"""
        salt = base64.b64decode(encrypted_data["salt"])
        iv = base64.b64decode(encrypted_data["iv"])
        tag = base64.b64decode(encrypted_data["tag"])
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])

        if password:
            key = CryptoEngine.derive_key(password, salt)
        else:
            master_password = "STEGO_MASTER_KEY_2024_SECURE"
            key = CryptoEngine.derive_key(master_password, salt)

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext


class CompressionEngine:
    """Motor de compresión ZIP para documentos"""

    @staticmethod
    def is_already_compressed(filename: str) -> bool:
        """Verifica si el archivo ya está comprimido"""
        compressed_extensions = ['.zip', '.rar', '.7z', '.gz', '.tar']
        ext = os.path.splitext(filename)[1].lower()
        return ext in compressed_extensions

    @staticmethod
    def compress_file(file_bytes: bytes, original_filename: str) -> bytes:
        """Comprime un archivo a ZIP en memoria con compresión máxima"""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            zip_file.writestr(original_filename, file_bytes)

        return zip_buffer.getvalue()

    @staticmethod
    def decompress_file(zip_bytes: bytes) -> Tuple[bytes, str]:
        """
        Descomprime un archivo ZIP y retorna (contenido, nombre_original)
        """
        zip_buffer = io.BytesIO(zip_bytes)

        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            filenames = zip_file.namelist()
            if not filenames:
                raise ValueError("El archivo ZIP está vacío")

            original_filename = filenames[0]
            file_content = zip_file.read(original_filename)

            return file_content, original_filename


def prepare_document_payload(
        doc_bytes: bytes,
        original_filename: str,
        mime_type: str,
        user_id: Optional[str] = "Anonymous",
        password: Optional[str] = None
) -> Tuple[str, dict]:
    """
    Prepara un documento para embeber: comprime, cifra y retorna payload + stats

    Returns:
        (payload_string, statistics_dict)
    """
    import time

    original_size = len(doc_bytes)

    # Comprimir si no está comprimido
    if CompressionEngine.is_already_compressed(original_filename):
        compressed_bytes = doc_bytes
        was_compressed = False
    else:
        compressed_bytes = CompressionEngine.compress_file(doc_bytes, original_filename)
        was_compressed = True

    compressed_size = len(compressed_bytes)

    # Crear metadata
    metadata = {
        "version": "1.0",
        "type": "document",
        "original_filename": original_filename,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "was_compressed": was_compressed,
        "mime_type": mime_type,
        "user_id": user_id,
        "embedded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_password": password is not None,
        "data": base64.b64encode(compressed_bytes).decode('utf-8')
    }

    metadata_json = json.dumps(metadata).encode('utf-8')

    # Cifrar
    encrypted_payload = CryptoEngine.encrypt(metadata_json, password)
    final_payload = json.dumps(encrypted_payload)

    # Estadísticas
    stats = {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "payload_size": len(final_payload),
        "was_compressed": was_compressed,
        "is_password_protected": password is not None
    }

    return final_payload, stats


def extract_document_payload(
        encrypted_payload: str,
        password: Optional[str] = None
) -> dict:
    """
    Extrae y descifra un documento desde un payload cifrado

    Returns:
        {
            "document_bytes": bytes,
            "original_filename": str,
            "original_size": int,
            "mime_type": str,
            "user_id": str,
            "embedded_at": str
        }

    Raises:
        ValueError: Si la contraseña es incorrecta o si falta cuando es requerida
    """
    try:
        encrypted_data = json.loads(encrypted_payload)
        decrypted_bytes = CryptoEngine.decrypt(encrypted_data, password)
    except Exception as e:
        if password:
            raise ValueError("Contraseña incorrecta")
        else:
            raise ValueError("Este documento requiere contraseña")

    # Parsear metadata
    metadata = json.loads(decrypted_bytes.decode('utf-8'))

    # Verificar contraseña requerida
    if metadata.get("has_password", False) and not password:
        raise ValueError("Este documento requiere contraseña")

    # Descomprimir
    compressed_data = base64.b64decode(metadata["data"])

    if metadata.get("was_compressed", False):
        original_bytes, _ = CompressionEngine.decompress_file(compressed_data)
    else:
        original_bytes = compressed_data

    return {
        "document_bytes": original_bytes,
        "original_filename": metadata["original_filename"],
        "original_size": metadata["original_size"],
        "mime_type": metadata["mime_type"],
        "user_id": metadata["user_id"],
        "embedded_at": metadata["embedded_at"]
    }