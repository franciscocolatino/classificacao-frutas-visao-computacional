"""
Pacote `src` do projeto de Classificação de Frutas.

Reúne os módulos do pipeline de Visão Computacional:
    - preprocessamento : limpeza e segmentação das imagens
    - extrator_fourier : descritor de FORMA (contorno + FFT)
    - extrator_textura : descritores de TEXTURA (GLCM + LBP)
    - classificador    : pipeline StandardScaler + SVM (RBF)
    - avaliacao        : métricas, relatórios e visualizações
"""
