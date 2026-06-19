"""
Montagem do vetor de caracteristicas e o classificador.

Junta os tres descritores num vetor unico e treina um SVM.
Uso um Pipeline com StandardScaler + SVM (RBF): o scaler e importante
porque os descritores tem escalas bem diferentes (ex.: contrast da GLCM
vs bins do LBP entre 0 e 1) e o SVM e sensivel a isso.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import extrator_fourier, extrator_textura, preprocessamento


def montar_vetor_caracteristicas(resultado_preproc, n_fourier=15):
    """Monta o vetor de uma imagem: [Fourier | GLCM | LBP]."""
    cinza = resultado_preproc["cinza"]
    mascara = resultado_preproc["mascara"]

    fourier = extrator_fourier.descritor_fourier(mascara, n_coeficientes=n_fourier)
    textura = extrator_textura.extrair_textura(cinza, mascara)

    return np.concatenate([fourier, textura])


def extrair_caracteristicas_de_caminho(caminho, n_fourier=15):
    """Pre-processa a imagem do disco e ja devolve o vetor de caracteristicas."""
    dados = preprocessamento.preprocessar(caminho)
    return montar_vetor_caracteristicas(dados, n_fourier=n_fourier)


def criar_classificador():
    """Cria o pipeline StandardScaler + SVM RBF."""
    modelo = Pipeline(
        steps=[
            ("escala", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]
    )
    return modelo


def treinar(modelo, X_treino, y_treino):
    """Treina o modelo."""
    modelo.fit(X_treino, y_treino)
    return modelo


def prever(modelo, X):
    """Prediz as classes."""
    return modelo.predict(X)
