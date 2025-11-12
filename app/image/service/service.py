# Stegoanálisis
from scipy import stats, fftpack
from skimage import color, filters, measure
from skimage.metrics import structural_similarity as ssim

from app.models.dtoAndResponses import MetricDetail
import numpy as np
from PIL import Image
import cv2

class AdvancedSteganalysisEngine:
    """Motor con umbrales ajustados para mayor precisión"""

    @staticmethod
    def advanced_lsb_detector(image_path: str) -> MetricDetail:
        """
        DETECTOR PRINCIPAL - Analiza distribución LSB con contexto estadístico
        Esta es la métrica MÁS CONFIABLE
        """
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        # Extraer LSB de cada canal
        lsb_layers = []
        for i in range(3):
            channel = pixels[:, :, i]
            lsb = channel & 1
            lsb_layers.append(lsb)

        # CLAVE: Analizar entropía de los LSB
        # LSB natural tiene baja entropía (patrones)
        # LSB con mensaje tiene alta entropía (aleatorio)

        entropies = []
        for lsb_layer in lsb_layers:
            # Calcular entropía de bloques 8x8
            h, w = lsb_layer.shape
            block_entropies = []

            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    block = lsb_layer[i:i + 8, j:j + 8].flatten()
                    # Entropía de Shannon
                    values, counts = np.unique(block, return_counts=True)
                    probabilities = counts / len(block)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    block_entropies.append(entropy)

            entropies.append(np.mean(block_entropies))

        avg_entropy = float(np.mean(entropies))

        # UMBRAL: Entropía > 0.95 = datos aleatorios (mensaje oculto)
        # Entropía < 0.85 = patrones naturales
        is_suspicious = avg_entropy > 0.92
        severity = "high" if avg_entropy > 0.95 else "medium" if avg_entropy > 0.92 else "low"

        if is_suspicious:
            explanation = f"🔴 INFECTADA: Entropía LSB={avg_entropy:.4f} > 0.92. Los bits LSB son DEMASIADO aleatorios. En imágenes naturales, los LSB tienen patrones (entropía ~0.7-0.85). Aleatoriedad alta indica mensaje cifrado oculto."
        else:
            explanation = f"✓ LIMPIA: Entropía LSB={avg_entropy:.4f} ≤ 0.92. Los bits LSB mantienen patrones naturales."

        return MetricDetail(
            name="LSB Entropy Analysis (CRITICAL)",
            value=avg_entropy,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity,
            category = "critical"
        )

    @staticmethod
    def enhanced_sample_pair(image_path: str) -> MetricDetail:
        """
        DETECTOR SECUNDARIO - SPA mejorado
        MUY CONFIABLE cuando da positivo
        """
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        spa_scores = []

        for i in range(3):
            channel = pixels[:, :, i].flatten()

            # Pares consecutivos
            pairs = [(channel[j], channel[j + 1]) for j in range(0, len(channel) - 1, 2)]

            # Contar pares con mismo LSB
            same_lsb = sum(1 for p in pairs if (p[0] & 1) == (p[1] & 1))

            spa_score = same_lsb / len(pairs) if len(pairs) > 0 else 0.5
            spa_scores.append(spa_score)

        avg_spa = float(np.mean(spa_scores))

        # UMBRAL CRÍTICO: > 0.90 es casi imposible naturalmente
        # En imágenes naturales: 0.40-0.60
        # En LSB stego: > 0.85 (los pares se vuelven muy similares)

        is_suspicious = avg_spa > 0.85
        severity = "high" if avg_spa > 0.90 else "medium" if avg_spa > 0.85 else "low"

        if is_suspicious:
            explanation = f"🔴 INFECTADA: SPA={avg_spa:.4f} > 0.85. El {avg_spa * 100:.1f}% de pares vecinos tienen el mismo LSB. Esto es ESTADÍSTICAMENTE IMPOSIBLE en imágenes naturales (esperado: ~50%). Clara evidencia de manipulación LSB."
        else:
            explanation = f"✓ LIMPIA: SPA={avg_spa:.4f} ≤ 0.85. Los pares de píxeles mantienen distribución LSB natural."

        return MetricDetail(
            name="Sample Pair Analysis (CRITICAL)",
            value=avg_spa,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity,
            category="critical"
        )

    @staticmethod
    def channel_uniformity_detector(image_path: str) -> MetricDetail:
        """
        DETECTOR TERCIARIO - Uniformidad RGB
        Útil para detectar embedding selectivo
        """
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        variances = [np.var(pixels[:, :, i]) for i in range(3)]
        max_var = max(variances)
        min_var = min(variances)
        uniformity = float(min_var / max_var) if max_var > 0 else 1.0

        # CRÍTICO: < 0.1 indica que un canal fue fuertemente modificado
        is_suspicious = uniformity < 0.1
        severity = "high" if uniformity < 0.05 else "medium" if uniformity < 0.1 else "low"

        if is_suspicious:
            explanation = f"🔴 SOSPECHOSO: Uniformidad={uniformity:.4f} < 0.1. Los canales RGB tienen varianzas EXTREMADAMENTE diferentes ({variances}). Uno o más canales fueron selectivamente modificados, típico de LSB stego que ataca solo canal azul."
        else:
            explanation = f"✓ NORMAL: Uniformidad={uniformity:.4f} ≥ 0.1. Canales RGB balanceados."

        return MetricDetail(
            name="Channel Uniformity (SECONDARY)",
            value=uniformity,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity,
            category="critical"
        )

    # MÉTRICAS DE SOPORTE (informativas pero no decisivas)
    @staticmethod
    def chi_square_attack(image_path: str) -> MetricDetail:
        """Chi-Square CORREGIDO - Detecta solo manipulación LSB real"""
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        p_values = []
        for i in range(3):
            channel_data = pixels[:, :, i].flatten()

            # CORRECCIÓN: Analizar PARES vs IMPARES correctamente
            even_pixels = channel_data[channel_data % 2 == 0]
            odd_pixels = channel_data[channel_data % 2 == 1]

            # Si hay muy pocos píxeles de algún tipo, saltar
            if len(even_pixels) < 100 or len(odd_pixels) < 100:
                continue

            # Histograma de valores pares e impares
            hist_even, _ = np.histogram(even_pixels, bins=128, range=(0, 256))
            hist_odd, _ = np.histogram(odd_pixels, bins=128, range=(0, 256))

            # Chi-square test: compara distribuciones
            # En LSB stego, los histogramas par/impar son casi idénticos
            observed = hist_even
            expected = hist_odd + 1e-10  # Evitar división por 0

            # Normalizar para que tengan el mismo total
            observed = observed * (np.sum(expected) / np.sum(observed))

            chi_sq = np.sum(((observed - expected) ** 2) / expected)

            # Grados de libertad
            df = len(hist_even) - 1
            p_value = 1 - stats.chi2.cdf(chi_sq, df=df)

            # CRÍTICO: p-value MUY ALTO = sospechoso (distribuciones demasiado similares)
            # p-value bajo = normal (distribuciones diferentes como debe ser)
            p_values.append(p_value)

        if len(p_values) == 0:
            avg_p_value = 0.5
        else:
            avg_p_value = float(np.mean(p_values))

        # UMBRAL INVERTIDO: p-value ALTO es sospechoso
        is_suspicious = avg_p_value > 0.95  # Distribuciones casi idénticas = LSB
        severity = "high" if avg_p_value > 0.99 else "medium" if avg_p_value > 0.95 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: p-value={avg_p_value:.4f} > 0.95. Los histogramas par/impar son DEMASIADO similares, indicando que LSB fue manipulado para igualarlos artificialmente."
        else:
            explanation = f"✓ NORMAL: p-value={avg_p_value:.4f}. Los histogramas par/impar mantienen diferencias naturales."

        return MetricDetail(
            name="Chi-Square Attack",
            value=avg_p_value,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity,
            category="info"
        )

    @staticmethod
    def detect_text_image(image_path: str) -> bool:
        """Detecta si la imagen contiene texto predominante"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar bordes (texto tiene muchos bordes)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Calcular varianza de bloques pequeños
        h, w = gray.shape
        block_vars = []
        for i in range(0, h - 16, 16):
            for j in range(0, w - 16, 16):
                block = gray[i:i + 16, j:j + 16]
                block_vars.append(np.var(block))

        # Texto tiene alta varianza en algunos bloques, baja en otros
        var_std = np.std(block_vars)

        # Alto contraste también indica texto
        contrast = gray.max() - gray.min()

        # Umbral: si tiene bordes densos + alta varianza + alto contraste = texto
        is_text = edge_density > 0.15 and var_std > 500 and contrast > 200

        return is_text

    @staticmethod
    def lsb_analysis(image_path: str) -> MetricDetail:
        """Analiza balance LSB - MEJORADO con contexto de brillo"""
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        # Calcular brillo promedio de la imagen
        avg_brightness = float(np.mean(pixels))

        ratios = []
        for i in range(3):
            channel_data = pixels[:, :, i].flatten()
            lsb_data = channel_data & 1
            ones_ratio = np.mean(lsb_data)
            ratios.append(ones_ratio)

        avg_ratio = float(np.mean(ratios))
        deviation = abs(avg_ratio - 0.5)

        # NUEVO: Ajustar umbral según brillo
        # Imágenes brillantes (>200) naturalmente tienen más 1s en LSB
        # Imágenes oscuras (<55) naturalmente tienen más 0s en LSB
        if avg_brightness > 200:  # Imagen muy clara
            threshold = 0.15  # Más tolerante
        elif avg_brightness < 55:  # Imagen muy oscura
            threshold = 0.15  # Más tolerante
        else:  # Imagen normal
            threshold = 0.08  # Estándar

        is_suspicious = deviation > threshold

        if is_suspicious:
            severity = "high" if deviation > threshold * 1.5 else "medium"
        else:
            severity = "low"

        if is_suspicious:
            explanation = f"⚠️ ANÓMALO: LSB ratio={avg_ratio:.4f} (desviación: {deviation * 100:.2f}%, brillo: {avg_brightness:.0f}). Balance anormal para esta imagen."
        else:
            explanation = f"✓ NORMAL: LSB ratio={avg_ratio:.4f} (brillo: {avg_brightness:.0f}). Balance esperado para imagen {'clara' if avg_brightness > 200 else 'oscura' if avg_brightness < 55 else 'normal'}."

        return MetricDetail(
            name="LSB Bit Balance",
            value=avg_ratio,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def sample_pair_analysis(image_path: str) -> MetricDetail:
        """Sample Pair Analysis - MEJORADO con contexto"""
        img = Image.open(image_path).convert('L')
        pixels = np.array(img).flatten()

        # Calcular brillo para ajustar expectativa
        avg_brightness = float(np.mean(pixels))

        pairs = [(pixels[i], pixels[i + 1]) for i in range(0, len(pixels) - 1, 2)]
        x_same_lsb = sum(1 for p in pairs if (p[0] & 1) == (p[1] & 1))

        spa_score = float(x_same_lsb / len(pairs)) if len(pairs) > 0 else 0.5
        deviation = abs(spa_score - 0.5)

        # NUEVO: Ajustar según brillo (igual que LSB)
        if avg_brightness > 200 or avg_brightness < 55:
            threshold = 0.15
        else:
            threshold = 0.08

        is_suspicious = deviation > threshold
        severity = "medium" if is_suspicious else "low"  # NUNCA high

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: SPA={spa_score:.4f} (brillo: {avg_brightness:.0f}). Pares LSB anormales."
        else:
            explanation = f"✓ NORMAL: SPA={spa_score:.4f}. Pares LSB naturales."

        return MetricDetail(
            name="Sample Pair Analysis",
            value=spa_score,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def rs_analysis(image_path: str) -> MetricDetail:
        """RS Steganalysis - MEJORADO"""
        img = Image.open(image_path).convert('L')
        pixels = np.array(img).flatten()

        def flip_lsb(arr):
            return arr ^ 1

        block_size = 8
        blocks = [pixels[i:i + block_size] for i in range(0, len(pixels), block_size)
                  if len(pixels[i:i + block_size]) == block_size]

        regular = 0
        singular = 0

        # Aumentar muestra para mayor precisión
        for block in blocks[:min(5000, len(blocks))]:  # 2000 → 5000
            original_var = np.var(block)
            flipped = flip_lsb(block)
            flipped_var = np.var(flipped)

            if flipped_var > original_var:
                regular += 1
            elif flipped_var < original_var:
                singular += 1

        total = regular + singular
        rs_ratio = float(regular / total) if total > 0 else 0.5

        # UMBRAL MÁS ESTRICTO
        deviation = abs(rs_ratio - 0.5)
        is_suspicious = deviation > 0.20  # 0.15 → 0.20
        severity = "high" if deviation > 0.30 else "medium" if deviation > 0.20 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: RS={rs_ratio:.4f}. Desbalance R/S muy significativo."
        else:
            explanation = f"✓ NORMAL: RS={rs_ratio:.4f}. Balance R/S adecuado."

        return MetricDetail(
            name="RS Analysis",
            value=rs_ratio,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def histogram_attack(image_path: str) -> MetricDetail:
        """Histograma par-impar - CORREGIDO"""
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        correlations = []
        for i in range(3):
            channel_data = pixels[:, :, i].flatten()

            # Separar valores pares e impares
            even_vals = channel_data[channel_data % 2 == 0]
            odd_vals = channel_data[channel_data % 2 == 1]

            # Histograma
            hist_even, _ = np.histogram(even_vals, bins=128, range=(0, 256))
            hist_odd, _ = np.histogram(odd_vals, bins=128, range=(0, 256))

            # Asegurar misma longitud
            min_len = min(len(hist_even), len(hist_odd))
            hist_even = hist_even[:min_len]
            hist_odd = hist_odd[:min_len]

            # Verificar varianza (evitar correlación de arrays constantes)
            if np.std(hist_even) < 1 or np.std(hist_odd) < 1:
                similarity = 0.0  # No hay información útil
            else:
                similarity = np.corrcoef(hist_even, hist_odd)[0, 1]

            correlations.append(similarity)

        avg_corr = float(np.mean(correlations))

        # Correlación ALTA = sospechoso (los histogramas NO deberían ser similares)
        is_suspicious = avg_corr > 0.98
        severity = "high" if avg_corr > 0.99 else "medium" if avg_corr > 0.98 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: Correlación={avg_corr:.4f} > 0.98. Histogramas par/impar artificialmente similares."
        else:
            explanation = f"✓ NORMAL: Correlación={avg_corr:.4f}. Histogramas mantienen independencia."

        return MetricDetail(
            name="Histogram Par-Impar Analysis",
            value=avg_corr,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def pixel_difference_analysis(image_path: str) -> MetricDetail:
        """Diferencias de píxeles - AJUSTADO"""
        img = Image.open(image_path).convert('L')
        pixels = np.array(img, dtype=np.float32)

        diff_h = np.abs(np.diff(pixels, axis=1))
        diff_v = np.abs(np.diff(pixels, axis=0))
        avg_diff = float(np.mean([np.mean(diff_h), np.mean(diff_v)]))

        # UMBRAL AJUSTADO: más tolerante
        is_suspicious = avg_diff > 25  # Antes: 15
        severity = "high" if avg_diff > 35 else "medium" if avg_diff > 25 else "low"

        if is_suspicious:
            explanation = f"⚠️ ANÓMALO: Diferencia={avg_diff:.2f} > 25. Transiciones irregulares."
        else:
            explanation = f"✓ NORMAL: Diferencia={avg_diff:.2f}. Transiciones suaves."

        return MetricDetail(
            name="Pixel Difference Analysis",
            value=avg_diff,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def color_channel_uniformity(image_path: str) -> MetricDetail:
        """Uniformidad de canales - AJUSTADO"""
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        variances = [np.var(pixels[:, :, i]) for i in range(3)]
        max_var = max(variances)
        min_var = min(variances)
        uniformity = float(min_var / max_var) if max_var > 0 else 1.0

        # UMBRAL AJUSTADO: más tolerante
        is_suspicious = uniformity < 0.5  # Antes: 0.7
        severity = "high" if uniformity < 0.3 else "medium" if uniformity < 0.5 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: Uniformidad={uniformity:.4f} < 0.5. Canales desbalanceados."
        else:
            explanation = f"✓ NORMAL: Uniformidad={uniformity:.4f}. Canales balanceados."

        return MetricDetail(
            name="Color Channel Uniformity",
            value=uniformity,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def noise_index(image_path: str) -> MetricDetail:
        """Índice de ruido - AJUSTADO"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean_filter = cv2.blur(gray, (5, 5))
        noise_map = np.abs(gray.astype(np.float32) - mean_filter.astype(np.float32))
        noise_level = float(np.mean(noise_map))

        # UMBRAL AJUSTADO: más tolerante
        is_suspicious = noise_level > 12  # Antes: 8
        severity = "high" if noise_level > 18 else "medium" if noise_level > 12 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: Ruido={noise_level:.2f} > 12. Ruido artificial detectado."
        else:
            explanation = f"✓ NORMAL: Ruido={noise_level:.2f}. Textura coherente."

        return MetricDetail(
            name="Noise Index",
            value=noise_level,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def fourier_spectrum_consistency(image_path: str) -> MetricDetail:
        """Análisis FFT - AJUSTADO"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f_transform = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        high_freq_region = magnitude[0:center_h//2, 0:center_w//2]
        low_freq_region = magnitude[center_h-20:center_h+20, center_w-20:center_w+20]

        high_energy = np.mean(high_freq_region)
        low_energy = np.mean(low_freq_region)
        freq_ratio = float(high_energy / low_energy) if low_energy > 0 else 0

        # UMBRAL AJUSTADO: más tolerante
        is_suspicious = freq_ratio > 0.5  # Antes: 0.3
        severity = "high" if freq_ratio > 0.8 else "medium" if freq_ratio > 0.5 else "low"

        if is_suspicious:
            explanation = f"⚠️ ANÓMALO: Fourier ratio={freq_ratio:.4f} > 0.5. Altas frecuencias anormales."
        else:
            explanation = f"✓ NORMAL: Fourier ratio={freq_ratio:.4f}. Espectro natural."

        return MetricDetail(
            name="Fourier Spectrum Consistency",
            value=freq_ratio,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )

    @staticmethod
    def ssim_self_similarity(image_path: str) -> MetricDetail:
        """SSIM entre bloques - AJUSTADO"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        if h < 50 or w < 50:
            return MetricDetail(
                name="SSIM Self-Similarity",
                value=1.0,
                explanation="⚠️ Imagen muy pequeña para análisis SSIM.",
                is_suspicious=False,
                severity="low"
            )

        block1 = gray[0:h//2, 0:w//2]
        block2 = gray[h//2:, w//2:]

        min_h = min(block1.shape[0], block2.shape[0])
        min_w = min(block1.shape[1], block2.shape[1])
        block1 = block1[:min_h, :min_w]
        block2 = block2[:min_h, :min_w]

        similarity = float(ssim(block1, block2))

        # UMBRAL AJUSTADO: más estricto
        is_suspicious = similarity < 0.3  # Antes: 0.5
        severity = "high" if similarity < 0.2 else "medium" if similarity < 0.3 else "low"

        if is_suspicious:
            explanation = f"⚠️ SOSPECHOSO: SSIM={similarity:.4f} < 0.3. Baja similitud estructural."
        else:
            explanation = f"✓ NORMAL: SSIM={similarity:.4f}. Coherencia estructural adecuada."

        return MetricDetail(
            name="SSIM Self-Similarity",
            value=similarity,
            explanation=explanation,
            is_suspicious=is_suspicious,
            severity=severity
        )
