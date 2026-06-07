"""
Módulo do DESCRITOR DE FOURIER (descritor de FORMA).

Ideia central: o CONTORNO de um objeto é uma curva fechada. Se
percorrermos essa curva e tratarmos cada ponto (x, y) como um número
complexo z = x + i*y, obtemos um SINAL 1D periódico. Aplicando a FFT
nesse sinal, os coeficientes resultantes ("descritores de Fourier")
capturam a forma global do objeto de maneira compacta.

Vantagem: com poucos coeficientes já representamos bem a forma, e é
fácil torná-los invariantes a translação, escala e rotação — exatamente
o que queremos para classificar formas independentemente de como a
imagem foi capturada.
"""

import cv2
import numpy as np


def encontrar_maior_contorno(mascara):
    """Encontra o maior contorno externo presente na máscara binária.

    Assumimos que o objeto de interesse é a maior região conectada.
    Pegar o maior contorno descarta automaticamente pequenos ruídos
    que possam ter sobrado da segmentação.

    Args:
        mascara (numpy.ndarray): imagem binária (objeto = 255).

    Returns:
        numpy.ndarray: contorno no formato (N, 1, 2) do OpenCV.

    Raises:
        ValueError: se NENHUM contorno for encontrado (máscara vazia).
    """
    # RETR_EXTERNAL pega só os contornos externos (ignora buracos internos).
    # CHAIN_APPROX_NONE mantém TODOS os pontos do contorno — precisamos da
    # curva completa para que a FFT represente a forma com fidelidade.
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contornos:
        # Tratamento crítico: sem contorno não há forma a descrever.
        raise ValueError("Nenhum contorno encontrado na máscara (objeto vazio).")

    # max por área: o objeto principal é a maior região fechada.
    maior = max(contornos, key=cv2.contourArea)
    return maior


def contorno_para_complexo(contorno):
    """Converte um contorno (lista de pontos x,y) em um sinal complexo.

    Cada ponto vira z = x + i*y. A sequência de z's é o sinal 1D que
    será transformado pela FFT.

    Args:
        contorno (numpy.ndarray): contorno do OpenCV, formato (N, 1, 2).

    Returns:
        numpy.ndarray: vetor complexo de tamanho N.
    """
    # squeeze remove a dimensão extra do meio -> (N, 2)
    pontos = contorno.squeeze()

    # Caso raro: contorno com um único ponto vira shape (2,). Garantimos 2D.
    if pontos.ndim == 1:
        pontos = pontos.reshape(1, 2)

    # x na parte real, y na parte imaginária.
    complexo = pontos[:, 0].astype(np.float64) + 1j * pontos[:, 1].astype(np.float64)
    return complexo


def descritor_fourier(mascara, n_coeficientes=15):
    """Calcula o descritor de Fourier normalizado a partir da máscara.

    Passos e o PORQUÊ de cada normalização:

    1. FFT do contorno complexo -> coeficientes de Fourier.
    2. Invariância à TRANSLAÇÃO: descartamos o coeficiente 0 (componente
       DC), que representa apenas a posição (centro) do objeto.
    3. Invariância à ROTAÇÃO e ao ponto inicial: usamos o MÓDULO (abs)
       dos coeficientes, pois rotação/deslocamento do ponto de partida
       afetam apenas a FASE, não a magnitude.
    4. Invariância à ESCALA: dividimos todos os módulos pelo módulo do
       primeiro coeficiente, normalizando o "tamanho" da forma.

    Args:
        mascara (numpy.ndarray): máscara binária do objeto.
        n_coeficientes (int): quantos coeficientes (após o DC) manter.

    Returns:
        numpy.ndarray: vetor de descritores de tamanho `n_coeficientes`.
    """
    contorno = encontrar_maior_contorno(mascara)
    sinal = contorno_para_complexo(contorno)

    # FFT do sinal complexo do contorno.
    coef = np.fft.fft(sinal)

    # (2) Remove a componente DC (índice 0) -> invariância à translação.
    coef = coef[1:]

    # Se o contorno for muito curto, completamos com zeros para manter o
    # tamanho do vetor de saída constante (evita erro no classificador).
    if len(coef) < n_coeficientes + 1:
        coef = np.pad(coef, (0, n_coeficientes + 1 - len(coef)))

    # (3) Magnitude -> invariância à rotação e ao ponto inicial.
    magnitudes = np.abs(coef)

    # (4) Normalização pela escala. O primeiro coeficiente (agora índice 0)
    # carrega a "amplitude" dominante da forma; dividir por ele torna o
    # descritor invariante ao tamanho. Protegemos contra divisão por zero.
    fator_escala = magnitudes[0] if magnitudes[0] != 0 else 1.0
    magnitudes = magnitudes / fator_escala

    # Mantemos os N coeficientes seguintes (descartamos o usado na normalização).
    descritores = magnitudes[1 : n_coeficientes + 1]

    return descritores


def assinatura_distancia_centroide(mascara, n_pontos=180):
    """Calcula a assinatura de forma (distância ao centroide) para plotar.

    Esta função é usada APENAS para visualização: para cada ponto do
    contorno, medimos a distância até o centro do objeto. O gráfico
    resultante é a "assinatura da forma" — uma forma intuitiva de mostrar
    visualmente o que o descritor de Fourier resume numericamente.

    Args:
        mascara (numpy.ndarray): máscara binária.
        n_pontos (int): número de pontos reamostrados da curva.

    Returns:
        numpy.ndarray: vetor de distâncias normalizadas (tamanho n_pontos).
    """
    contorno = encontrar_maior_contorno(mascara)
    pontos = contorno.squeeze().astype(np.float64)
    if pontos.ndim == 1:
        pontos = pontos.reshape(1, 2)

    centro = pontos.mean(axis=0)
    distancias = np.sqrt(((pontos - centro) ** 2).sum(axis=1))

    # Reamostra para um número fixo de pontos (interpolação linear) para
    # que a assinatura tenha sempre o mesmo comprimento ao comparar formas.
    indices_orig = np.linspace(0, 1, len(distancias))
    indices_novos = np.linspace(0, 1, n_pontos)
    distancias = np.interp(indices_novos, indices_orig, distancias)

    # Normaliza pelo máximo -> invariância à escala, bom para visualização.
    maximo = distancias.max() if distancias.max() != 0 else 1.0
    return distancias / maximo
