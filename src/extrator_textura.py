"""
Módulo dos DESCRITORES DE TEXTURA.

Enquanto o descritor de Fourier captura a FORMA (o contorno), aqui
descrevemos a TEXTURA da superfície do objeto: a forma como os níveis
de cinza se distribuem e se repetem. Duas técnicas complementares:

    - GLCM (Gray-Level Co-occurrence Matrix): estatísticas de SEGUNDA
      ordem. Mede com que frequência pares de intensidades aparecem a
      uma certa distância/direção. Ótimo para "rugosidade" global.

    - LBP (Local Binary Patterns): descritor LOCAL e robusto a variações
      de iluminação. Codifica, para cada pixel, o padrão de vizinhança
      (mais claro/mais escuro). O histograma desses padrões resume a
      microtextura.

O vetor final de textura é a CONCATENAÇÃO de GLCM + LBP.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# --- Parâmetros do LBP ---
# P = número de vizinhos amostrados ao redor do pixel central.
# R = raio do círculo de vizinhança.
# 'uniform' agrupa padrões em rotacionalmente invariantes + ruído,
# gerando um histograma compacto (P + 2 bins) e robusto.
LBP_P = 8
LBP_R = 1
LBP_METODO = "uniform"

# Propriedades extraídas da GLCM (5 medidas pedidas no enunciado).
PROPRIEDADES_GLCM = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def extrair_glcm(cinza, mascara=None):
    """Extrai estatísticas de textura via matriz de co-ocorrência (GLCM).

    Args:
        cinza (numpy.ndarray): imagem em tons de cinza (0-255).
        mascara (numpy.ndarray, opcional): se fornecida, zera o fundo para
            que a textura medida venha majoritariamente do objeto.

    Returns:
        numpy.ndarray: vetor com as 5 propriedades da GLCM
            [contrast, dissimilarity, homogeneity, energy, correlation].
    """
    imagem = cinza.copy()
    if mascara is not None:
        # Aplica a máscara: fora do objeto vira 0, focando a textura no alvo.
        imagem = np.where(mascara > 0, imagem, 0).astype(np.uint8)

    # Reduzir de 256 para 32 níveis de cinza torna a GLCM menor, mais
    # robusta a ruído e mais rápida de calcular, sem perder o padrão de textura.
    niveis = 32
    imagem_quant = (imagem.astype(np.uint16) * niveis // 256).astype(np.uint8)

    # Calculamos a GLCM em 1 pixel de distância e em 4 direções (0°, 45°,
    # 90°, 135°). Usar várias direções e depois a média torna o descritor
    # aproximadamente invariante à rotação.
    glcm = graycomatrix(
        imagem_quant,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=niveis,
        symmetric=True,   # par (i,j) conta igual a (j,i): direção indiferente
        normed=True,      # normaliza para probabilidades (some 1)
    )

    # Para cada propriedade, tiramos a média entre as 4 direções.
    caracteristicas = []
    for prop in PROPRIEDADES_GLCM:
        valor = graycoprops(glcm, prop).mean()
        caracteristicas.append(valor)

    return np.array(caracteristicas, dtype=np.float64)


def extrair_lbp(cinza, mascara=None):
    """Extrai o histograma normalizado de Local Binary Patterns (LBP).

    Args:
        cinza (numpy.ndarray): imagem em tons de cinza.
        mascara (numpy.ndarray, opcional): restringe o histograma aos
            pixels do objeto, ignorando o fundo.

    Returns:
        numpy.ndarray: histograma normalizado de tamanho (LBP_P + 2).
    """
    # Mapa LBP: cada pixel recebe um código baseado em sua vizinhança.
    lbp = local_binary_pattern(cinza, LBP_P, LBP_R, method=LBP_METODO)

    # No método 'uniform' os rótulos vão de 0 a P+1 (P+2 categorias).
    n_bins = LBP_P + 2

    if mascara is not None:
        # Considera apenas os códigos LBP dentro do objeto.
        valores = lbp[mascara > 0]
        # Se a máscara estiver vazia, evitamos um histograma degenerado.
        if valores.size == 0:
            valores = lbp.ravel()
    else:
        valores = lbp.ravel()

    histograma, _ = np.histogram(valores, bins=n_bins, range=(0, n_bins))

    # Normaliza o histograma para somar 1 -> invariância ao tamanho do
    # objeto (número de pixels) e comparabilidade entre imagens.
    histograma = histograma.astype(np.float64)
    soma = histograma.sum()
    if soma > 0:
        histograma /= soma

    return histograma


def mapa_lbp(cinza):
    """Retorna o MAPA LBP (imagem) para fins de visualização.

    Diferente de `extrair_lbp` (que devolve o histograma), aqui devolvemos
    a imagem de códigos LBP, útil para mostrar visualmente a microtextura.
    """
    return local_binary_pattern(cinza, LBP_P, LBP_R, method=LBP_METODO)


def extrair_textura(cinza, mascara=None):
    """Concatena GLCM + LBP em um único vetor de textura.

    Args:
        cinza (numpy.ndarray): imagem em tons de cinza.
        mascara (numpy.ndarray, opcional): máscara do objeto.

    Returns:
        numpy.ndarray: vetor de textura = [5 medidas GLCM | (P+2) bins LBP].
    """
    glcm = extrair_glcm(cinza, mascara)
    lbp = extrair_lbp(cinza, mascara)
    return np.concatenate([glcm, lbp])
