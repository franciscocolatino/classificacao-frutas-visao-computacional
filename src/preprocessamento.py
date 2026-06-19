"""
Pré-processamento das imagens.

Pega a imagem original e gera uma versao em cinza e uma mascara binaria
do objeto. Essa mascara depois e usada tanto pelo descritor de forma
(contorno) quanto para recortar a regiao da textura.

Etapas: resize 256x256, cinza, blur gaussiano, Otsu e morfologia.
"""

import cv2
import numpy as np

TAMANHO_PADRAO = (256, 256)


def carregar_imagem(caminho):
    """Le a imagem do disco (BGR)."""
    imagem = cv2.imread(caminho, cv2.IMREAD_COLOR)
    # cv2.imread nao lanca erro, so retorna None quando falha
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return imagem


def redimensionar(imagem, tamanho=TAMANHO_PADRAO):
    """Redimensiona para um tamanho fixo (deixa as imagens comparaveis)."""
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)


def para_cinza(imagem):
    """Converte de BGR para cinza."""
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def filtro_gaussiano(cinza, kernel=(5, 5)):
    """Blur gaussiano pra tirar um pouco do ruido antes de segmentar."""
    return cv2.GaussianBlur(cinza, kernel, sigmaX=0)


def segmentar_otsu(suavizada):
    """Segmenta objeto/fundo com Otsu."""
    # uso BINARY_INV porque a fruta costuma ser mais escura que o fundo,
    # entao invertendo o objeto fica branco (255)
    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binaria


def operacoes_morfologicas(binaria, kernel_tamanho=5):
    """Limpa a mascara: abertura (tira ruido) e fechamento (fecha buracos)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_tamanho, kernel_tamanho)
    )
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)
    return fechamento


def preprocessar(caminho):
    """Roda o pre-processamento inteiro a partir de um caminho de arquivo."""
    imagem = carregar_imagem(caminho)
    imagem = redimensionar(imagem)
    cinza = para_cinza(imagem)
    suavizada = filtro_gaussiano(cinza)
    binaria = segmentar_otsu(suavizada)
    mascara = operacoes_morfologicas(binaria)

    return {
        "original": imagem,
        "cinza": cinza,
        "suavizada": suavizada,
        "mascara": mascara,
    }


def preprocessar_array(imagem):
    """Igual ao preprocessar, mas recebe a imagem ja carregada (usado no demo)."""
    imagem = redimensionar(imagem)
    cinza = para_cinza(imagem)
    suavizada = filtro_gaussiano(cinza)
    binaria = segmentar_otsu(suavizada)
    mascara = operacoes_morfologicas(binaria)

    return {
        "original": imagem,
        "cinza": cinza,
        "suavizada": suavizada,
        "mascara": mascara,
    }
