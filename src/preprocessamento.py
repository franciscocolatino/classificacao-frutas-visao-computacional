"""
Módulo de PRÉ-PROCESSAMENTO de imagens.

Objetivo: transformar a imagem bruta (colorida, com ruído e fundo) em
uma versão padronizada e em uma MÁSCARA BINÁRIA limpa do objeto.
Essa máscara é o ponto de partida tanto para o descritor de forma
(contorno) quanto para recortar a região de interesse para a textura.

Etapas (todas com OpenCV):
    1. Redimensionar para 256x256  -> padroniza a escala de entrada
    2. Escala de cinza             -> textura/forma não dependem de cor
    3. Filtro Gaussiano            -> suaviza ruído antes de segmentar
    4. Limiarização de Otsu        -> separa objeto do fundo automaticamente
    5. Morfologia (abertura+fech.) -> remove ruído e fecha buracos da máscara
"""

import cv2
import numpy as np

# Tamanho padrão de trabalho. Fixar a dimensão garante que descritores
# extraídos de imagens diferentes sejam comparáveis entre si.
TAMANHO_PADRAO = (256, 256)


def carregar_imagem(caminho):
    """Lê uma imagem do disco em formato BGR (padrão do OpenCV).

    Args:
        caminho (str): caminho do arquivo de imagem.

    Returns:
        numpy.ndarray: imagem colorida (H, W, 3).

    Raises:
        FileNotFoundError: se o OpenCV não conseguir ler o arquivo
            (caminho inválido, formato não suportado ou arquivo corrompido).
    """
    imagem = cv2.imread(caminho, cv2.IMREAD_COLOR)
    # cv2.imread NÃO lança exceção: ele retorna None em caso de falha.
    # Por isso validamos explicitamente para falhar de forma clara.
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return imagem


def redimensionar(imagem, tamanho=TAMANHO_PADRAO):
    """Redimensiona a imagem para um tamanho fixo.

    Padronizar o tamanho é essencial: descritores como GLCM/LBP e o
    contorno dependem da quantidade de pixels, então todas as imagens
    precisam estar na mesma escala.

    Args:
        imagem (numpy.ndarray): imagem de entrada.
        tamanho (tuple): (largura, altura) de destino.

    Returns:
        numpy.ndarray: imagem redimensionada.
    """
    # INTER_AREA é a melhor interpolação para REDUZIR imagens (menos artefatos).
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)


def para_cinza(imagem):
    """Converte BGR para escala de cinza.

    Trabalhamos com forma e textura, que independem da cor. A escala de
    cinza reduz a imagem a um único canal de intensidade, simplificando
    os cálculos seguintes.
    """
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def filtro_gaussiano(cinza, kernel=(5, 5)):
    """Aplica desfoque Gaussiano para reduzir ruído.

    O ruído de alta frequência atrapalha a limiarização (gera buracos e
    pontos isolados). Suavizar antes torna a segmentação mais estável.

    Args:
        cinza (numpy.ndarray): imagem em tons de cinza.
        kernel (tuple): tamanho do kernel (ímpar em cada dimensão).
    """
    # O desvio-padrão 0 faz o OpenCV calculá-lo a partir do tamanho do kernel.
    return cv2.GaussianBlur(cinza, kernel, sigmaX=0)


def segmentar_otsu(suavizada):
    """Segmenta o objeto do fundo usando limiarização de Otsu.

    Otsu escolhe AUTOMATICAMENTE o limiar que melhor separa duas
    populações de intensidade (objeto x fundo), maximizando a variância
    entre classes. Ideal quando não sabemos o limiar ideal de antemão.

    Returns:
        numpy.ndarray: imagem binária (0 ou 255).
    """
    # THRESH_OTSU calcula o limiar; o valor "0" passado é ignorado por causa disso.
    # THRESH_BINARY_INV é usado porque, em fotos de frutas, o objeto costuma ser
    # mais escuro que o fundo claro -> invertendo, o objeto vira branco (255).
    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binaria


def operacoes_morfologicas(binaria, kernel_tamanho=5):
    """Limpa a máscara binária com abertura seguida de fechamento.

    - ABERTURA (erosão -> dilatação): remove pontos brancos isolados (ruído).
    - FECHAMENTO (dilatação -> erosão): fecha pequenos buracos dentro do objeto.

    A ordem abertura->fechamento primeiro elimina o lixo externo e depois
    consolida o objeto, resultando em uma máscara sólida e bem definida.

    Args:
        binaria (numpy.ndarray): máscara binária de entrada.
        kernel_tamanho (int): lado do elemento estruturante quadrado.

    Returns:
        numpy.ndarray: máscara binária limpa.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_tamanho, kernel_tamanho)
    )
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)
    return fechamento


def preprocessar(caminho):
    """Executa o pipeline completo de pré-processamento de uma imagem.

    Orquestra todas as etapas e devolve os resultados intermediários,
    úteis tanto para os extratores quanto para as visualizações.

    Args:
        caminho (str): caminho da imagem.

    Returns:
        dict: com as chaves
            'original'  -> imagem colorida redimensionada (BGR)
            'cinza'     -> escala de cinza
            'suavizada' -> após filtro Gaussiano
            'mascara'   -> máscara binária final (objeto = 255)
    """
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
    """Versão do pré-processamento que recebe um array já em memória.

    Usada pela demonstração (demo.py), que gera imagens sintéticas e não
    precisa lê-las do disco. Evita duplicação de lógica.

    Args:
        imagem (numpy.ndarray): imagem colorida (BGR) já carregada.

    Returns:
        dict: mesmo formato de `preprocessar`.
    """
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
