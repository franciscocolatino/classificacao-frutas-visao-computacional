"""
Descritor de Fourier (forma).

A ideia: o contorno do objeto e uma curva fechada. Tratando cada ponto
(x, y) como um complexo z = x + i*y, da pra aplicar a FFT e usar os
coeficientes como descritor da forma. Com algumas normalizacoes eles
ficam invariantes a translacao, rotacao e escala.
"""

import cv2
import numpy as np


def encontrar_maior_contorno(mascara):
    """Pega o maior contorno externo da mascara (assume que e o objeto)."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contornos:
        raise ValueError("Nenhum contorno encontrado na máscara (objeto vazio).")

    maior = max(contornos, key=cv2.contourArea)
    return maior


def contorno_para_complexo(contorno):
    """Transforma os pontos (x, y) do contorno num sinal complexo z = x + i*y."""
    pontos = contorno.squeeze()  # (N, 1, 2) -> (N, 2)

    # contorno de um ponto so vira shape (2,); garante 2D
    if pontos.ndim == 1:
        pontos = pontos.reshape(1, 2)

    complexo = pontos[:, 0].astype(np.float64) + 1j * pontos[:, 1].astype(np.float64)
    return complexo


def descritor_fourier(mascara, n_coeficientes=15):
    """Calcula o descritor de Fourier normalizado.

    Normalizacoes:
    - translacao: descarta o coeficiente DC (indice 0)
    - rotacao/ponto inicial: usa o modulo dos coeficientes
    - escala: divide tudo pelo primeiro coeficiente
    """
    contorno = encontrar_maior_contorno(mascara)
    sinal = contorno_para_complexo(contorno)

    coef = np.fft.fft(sinal)

    # remove a componente DC (translacao)
    coef = coef[1:]

    # contorno curto demais: completa com zeros pra manter o tamanho do vetor
    if len(coef) < n_coeficientes + 1:
        coef = np.pad(coef, (0, n_coeficientes + 1 - len(coef)))

    # modulo -> invariante a rotacao e ao ponto inicial
    magnitudes = np.abs(coef)

    # normaliza pela escala (cuidado com divisao por zero)
    fator_escala = magnitudes[0] if magnitudes[0] != 0 else 1.0
    magnitudes = magnitudes / fator_escala

    # fica com os N seguintes (o usado na normalizacao sai fora)
    descritores = magnitudes[1 : n_coeficientes + 1]

    return descritores


def assinatura_distancia_centroide(mascara, n_pontos=180):
    """Assinatura de forma (distancia ao centroide). So pra visualizacao.

    Para cada ponto do contorno mede a distancia ate o centro. O grafico
    disso e uma forma intuitiva de mostrar o que o Fourier resume.
    """
    contorno = encontrar_maior_contorno(mascara)
    pontos = contorno.squeeze().astype(np.float64)
    if pontos.ndim == 1:
        pontos = pontos.reshape(1, 2)

    centro = pontos.mean(axis=0)
    distancias = np.sqrt(((pontos - centro) ** 2).sum(axis=1))

    # reamostra pra ter sempre o mesmo numero de pontos
    indices_orig = np.linspace(0, 1, len(distancias))
    indices_novos = np.linspace(0, 1, n_pontos)
    distancias = np.interp(indices_novos, indices_orig, distancias)

    maximo = distancias.max() if distancias.max() != 0 else 1.0
    return distancias / maximo
