"""
Descritores de textura: GLCM + LBP.

Enquanto o Fourier descreve a forma, aqui a ideia e descrever a textura
da superficie.
- GLCM: estatisticas de segunda ordem sobre pares de pixels.
- LBP: histograma dos padroes binarios locais, robusto a iluminacao.

O vetor final de textura e a concatenacao dos dois.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# parametros do LBP: P vizinhos, raio R, metodo uniform (histograma com P+2 bins)
LBP_P = 8
LBP_R = 1
LBP_METODO = "uniform"

# as 5 medidas da GLCM pedidas no enunciado
PROPRIEDADES_GLCM = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def extrair_glcm(cinza, mascara=None):
    """Extrai as 5 estatisticas da GLCM."""
    imagem = cinza.copy()
    if mascara is not None:
        # zera o fundo pra medir a textura so do objeto
        imagem = np.where(mascara > 0, imagem, 0).astype(np.uint8)

    # reduz de 256 pra 32 niveis: GLCM menor, mais rapida e menos sensivel a ruido
    niveis = 32
    imagem_quant = (imagem.astype(np.uint16) * niveis // 256).astype(np.uint8)

    # 1 pixel de distancia, 4 direcoes (depois tiro a media -> quase invariante a rotacao)
    glcm = graycomatrix(
        imagem_quant,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=niveis,
        symmetric=True,
        normed=True,
    )

    caracteristicas = []
    for prop in PROPRIEDADES_GLCM:
        valor = graycoprops(glcm, prop).mean()
        caracteristicas.append(valor)

    return np.array(caracteristicas, dtype=np.float64)


def extrair_lbp(cinza, mascara=None):
    """Histograma normalizado do LBP."""
    lbp = local_binary_pattern(cinza, LBP_P, LBP_R, method=LBP_METODO)

    # no metodo uniform os rotulos vao de 0 a P+1
    n_bins = LBP_P + 2

    if mascara is not None:
        valores = lbp[mascara > 0]
        # se a mascara veio vazia, usa a imagem toda pra nao quebrar
        if valores.size == 0:
            valores = lbp.ravel()
    else:
        valores = lbp.ravel()

    histograma, _ = np.histogram(valores, bins=n_bins, range=(0, n_bins))

    # normaliza pra somar 1
    histograma = histograma.astype(np.float64)
    soma = histograma.sum()
    if soma > 0:
        histograma /= soma

    return histograma


def mapa_lbp(cinza):
    """Retorna a imagem do LBP (pra visualizar, diferente do histograma)."""
    return local_binary_pattern(cinza, LBP_P, LBP_R, method=LBP_METODO)


def extrair_textura(cinza, mascara=None):
    """Junta GLCM + LBP num vetor so."""
    glcm = extrair_glcm(cinza, mascara)
    lbp = extrair_lbp(cinza, mascara)
    return np.concatenate([glcm, lbp])
