# Steganografía para audio
import wave
import audioop
from typing import Optional


class AudioSteganographyEngine:
    """Motor de esteganografía para archivos de audio WAV"""

    @staticmethod
    def hide_message(audio_path: str, message: str, output_path: str) -> bool:
        """
        Oculta un mensaje en archivo de audio usando LSB
        """
        try:
            with wave.open(audio_path, 'rb') as audio_file:
                params = audio_file.getparams()
                frames = audio_file.readframes(audio_file.getnframes())

            audio_data = bytearray(frames)
            message_bits = []

            for char in message:
                bits = format(ord(char), '08b')
                message_bits.extend([int(bit) for bit in bits])

            message_bits.extend([0, 0, 0, 0, 0, 0, 0, 0])

            if len(message_bits) > len(audio_data):
                return False

            for i in range(len(message_bits)):
                audio_data[i] = (audio_data[i] & 0xFE) | message_bits[i]

            with wave.open(output_path, 'wb') as output_file:
                output_file.setparams(params)
                output_file.writeframes(bytes(audio_data))

            return True

        except Exception as e:
            print(f"Error ocultando mensaje en audio: {e}")
            return False

    @staticmethod
    def reveal_message(audio_path: str) -> Optional[str]:
        """
        Extrae mensaje oculto de archivo de audio
        """
        try:
            with wave.open(audio_path, 'rb') as audio_file:
                frames = audio_file.readframes(audio_file.getnframes())

            audio_data = bytearray(frames)
            message_bits = []

            for byte in audio_data:
                message_bits.append(byte & 1)

            message = ""
            for i in range(0, len(message_bits), 8):
                if i + 8 > len(message_bits):
                    break

                byte_bits = message_bits[i:i+8]
                char_code = int(''.join(map(str, byte_bits)), 2)

                if char_code == 0:
                    break

                message += chr(char_code)

            return message if message else None

        except Exception as e:
            print(f"Error extrayendo mensaje de audio: {e}")
            return None
