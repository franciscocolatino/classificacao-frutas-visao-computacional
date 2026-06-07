"""
Módulo do CLASSIFICADOR e da MONTAGEM DO VETOR DE CARACTERÍSTICAS.

Aqui juntamos os três descritores em um único vetor e definimos o
modelo de Machine Learning que aprende a separar as classes de frutas.

Modelo: Pipeline do Scikit-Learn com
    StandardScaler -> SVM (kernel RBF)

Por que essa combinação?
    - StandardScaler: os descritores têm ESCALAS muito diferentes
      (ex.: 'contrast' da GLCM x bins do LBP entre 0 e 1). O SVM é
      sensível à escala, então padronizar (média 0, desvio 1) é
      indispensável para que nenhuma característica domine as demais.
    - SVM com kernel RBF: lida bem com fronteiras de decisão NÃO
      lineares e funciona muito bem em datasets pequenos/médios, como
      é o caso de uma atividade acadêmica.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import extrator_fourier, extrator_textura, preprocessamento


def montar_vetor_caracteristicas(resultado_preproc, n_fourier=15):
    """Constrói o vetor de características final de UMA imagem.

    Concatena, nesta ordem:
        [ Descritores de Fourier | GLCM | LBP ]

    Args:
        resultado_preproc (dict): saída de `preprocessamento.preprocessar`,
            contendo 'cinza' e 'mascara'.
        n_fourier (int): número de coeficientes de Fourier.

    Returns:
        numpy.ndarray: vetor de características 1D.
    """
    cinza = resultado_preproc["cinza"]
    mascara = resultado_preproc["mascara"]

    # FORMA (contorno + FFT)
    fourier = extrator_fourier.descritor_fourier(mascara, n_coeficientes=n_fourier)
    # TEXTURA (GLCM + LBP já concatenados internamente)
    textura = extrator_textura.extrair_textura(cinza, mascara)

    return np.concatenate([fourier, textura])


def extrair_caracteristicas_de_caminho(caminho, n_fourier=15):
    """Pré-processa uma imagem do disco e devolve seu vetor de características.

    Conveniência que encadeia pré-processamento + extração. Trata erros
    de leitura/segmentação propagando exceções claras para quem chamou.

    Args:
        caminho (str): caminho da imagem.
        n_fourier (int): número de coeficientes de Fourier.

    Returns:
        numpy.ndarray: vetor de características da imagem.
    """
    dados = preprocessamento.preprocessar(caminho)
    return montar_vetor_caracteristicas(dados, n_fourier=n_fourier)


def criar_classificador():
    """Cria o pipeline StandardScaler + SVM (RBF).

    Returns:
        sklearn.pipeline.Pipeline: modelo pronto para `.fit`/`.predict`.
    """
    modelo = Pipeline(
        steps=[
            # Etapa 1: padroniza cada característica (média 0, desvio 1).
            ("escala", StandardScaler()),
            # Etapa 2: SVM com kernel RBF.
            #   - C=10: penaliza erros de classificação (margem mais justa).
            #   - gamma='scale': ajusta a influência de cada amostra com base
            #     na variância dos dados (boa escolha padrão).
            #   - probability=True: permite estimar probabilidades se preciso.
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )
    return modelo


def treinar(modelo, X_treino, y_treino):
    """Treina o classificador.

    Args:
        modelo (Pipeline): pipeline criado por `criar_classificador`.
        X_treino (numpy.ndarray): matriz de características (amostras x atributos).
        y_treino (array-like): rótulos das amostras.

    Returns:
        Pipeline: o próprio modelo treinado (para encadeamento).
    """
    modelo.fit(X_treino, y_treino)
    return modelo


def prever(modelo, X):
    """Faz a predição das classes para novas amostras.

    Args:
        modelo (Pipeline): modelo treinado.
        X (numpy.ndarray): matriz de características.

    Returns:
        numpy.ndarray: rótulos previstos.
    """
    return modelo.predict(X)
