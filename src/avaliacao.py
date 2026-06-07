"""
Módulo de AVALIAÇÃO e VISUALIZAÇÕES.

Reúne tudo que gera saídas para análise e para a apresentação:
    - Métricas: acurácia, precisão, recall e F1-score.
    - Matriz de confusão (imagem PNG com seaborn).
    - Relatório de métricas exportado em CSV.
    - Gráfico de barras de F1-score por classe.
    - Visualizações do pipeline: etapas de pré-processamento,
      assinatura de forma (Fourier) e mapa de textura LBP.

Usa o backend "Agg" do matplotlib para funcionar mesmo em servidores
sem interface gráfica (apenas salva arquivos, não abre janelas).
"""

import math
import os

import cv2
import matplotlib

matplotlib.use("Agg")  # backend não-interativo: gera arquivos, não janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import extrator_fourier, extrator_textura


def garantir_pasta(pasta):
    """Cria a pasta de resultados se ela ainda não existir."""
    os.makedirs(pasta, exist_ok=True)


def calcular_metricas(y_verdadeiro, y_previsto):
    """Calcula as métricas globais de classificação.

    Usamos média 'weighted' (ponderada pelo nº de amostras de cada classe)
    para que classes desbalanceadas não distorçam o resultado global.

    Returns:
        dict: {'acuracia', 'precisao', 'recall', 'f1'}.
    """
    return {
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
        "precisao": precision_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
        "recall": recall_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
        "f1": f1_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
    }


def salvar_relatorio_csv(y_verdadeiro, y_previsto, classes, caminho_csv):
    """Gera o relatório de classificação (por classe) e salva em CSV.

    O `classification_report` do sklearn já calcula precisão, recall,
    F1 e suporte por classe — exportamos isso como tabela.

    Returns:
        pandas.DataFrame: o relatório, caso queira inspecionar em memória.
    """
    relatorio = classification_report(
        y_verdadeiro,
        y_previsto,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(relatorio).transpose()
    df.to_csv(caminho_csv, index=True, encoding="utf-8")
    return df


def plotar_matriz_confusao(y_verdadeiro, y_previsto, classes, caminho_png):
    """Plota e salva a matriz de confusão como heatmap colorido (seaborn).

    A matriz de confusão mostra, para cada classe real (linha), quantas
    amostras foram previstas em cada classe (coluna). A diagonal são os
    acertos; fora da diagonal estão as confusões entre classes.
    """
    matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=list(range(len(classes))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,        # escreve o número em cada célula
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_f1_por_classe(y_verdadeiro, y_previsto, classes, caminho_png):
    """Plota um gráfico de barras com o F1-score de cada classe.

    Permite enxergar rapidamente QUAIS classes o modelo classifica bem
    e quais são mais difíceis (barras mais baixas).
    """
    f1_por_classe = f1_score(
        y_verdadeiro,
        y_previsto,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )

    plt.figure(figsize=(8, 5))
    barras = plt.bar(classes, f1_por_classe, color=sns.color_palette("viridis", len(classes)))
    plt.ylim(0, 1.05)
    plt.ylabel("F1-score")
    plt.title("F1-score por Classe")
    plt.xticks(rotation=30, ha="right")

    # Escreve o valor acima de cada barra para leitura fácil na apresentação.
    for barra, valor in zip(barras, f1_por_classe):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.02,
            f"{valor:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_etapas_preprocessamento(dados, caminho_png):
    """Mostra lado a lado: original -> após filtro -> após segmentação.

    Visualização-chave para a banca: deixa explícito o efeito de cada
    etapa do pré-processamento sobre a imagem.

    Args:
        dados (dict): saída de `preprocessamento.preprocessar(_array)`.
        caminho_png (str): onde salvar a figura.
    """
    fig, eixos = plt.subplots(1, 3, figsize=(14, 5))

    # OpenCV usa BGR; matplotlib espera RGB. Convertemos só para exibir.
    original_rgb = dados["original"][:, :, ::-1]
    eixos[0].imshow(original_rgb)
    eixos[0].set_title("1. Original (256x256)")

    eixos[1].imshow(dados["suavizada"], cmap="gray")
    eixos[1].set_title("2. Cinza + Filtro Gaussiano")

    eixos[2].imshow(dados["mascara"], cmap="gray")
    eixos[2].set_title("3. Segmentação (Otsu + Morfologia)")

    for eixo in eixos:
        eixo.axis("off")

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_assinatura_fourier(dados, caminho_png):
    """Plota a 'assinatura de forma' (distância ao centroide) do objeto.

    É a representação visual e intuitiva do que o descritor de Fourier
    resume numericamente: como a borda do objeto varia ao longo do contorno.
    """
    assinatura = extrator_fourier.assinatura_distancia_centroide(dados["mascara"])

    plt.figure(figsize=(9, 4))
    plt.plot(assinatura, color="darkorange", linewidth=2)
    plt.title("Descritor de Forma — Assinatura (distância ao centroide)")
    plt.xlabel("Ponto ao longo do contorno")
    plt.ylabel("Distância normalizada")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_mapa_lbp(dados, caminho_png):
    """Plota o mapa de textura LBP ao lado da imagem em cinza."""
    lbp = extrator_textura.mapa_lbp(dados["cinza"])

    fig, eixos = plt.subplots(1, 2, figsize=(11, 5))
    eixos[0].imshow(dados["cinza"], cmap="gray")
    eixos[0].set_title("Imagem em Cinza")
    eixos[0].axis("off")

    # 'nipy_spectral' realça as diferentes regiões de microtextura.
    eixos[1].imshow(lbp, cmap="nipy_spectral")
    eixos[1].set_title("Mapa de Textura (LBP)")
    eixos[1].axis("off")

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_amostras_preditas(
    caminhos, y_verdadeiro, y_previsto, classes, caminho_png,
    n=15, colunas=5, apenas_erros=False, seed=42,
):
    """Mostra uma GALERIA de imagens de teste com 'Real' vs 'Previsto'.

    É a forma mais intuitiva de apresentar o resultado: vê-se a imagem real
    e a decisão do modelo. Acertos ficam em verde e erros em vermelho.

    Args:
        caminhos (array-like): caminhos das imagens de teste (alinhados a y).
        y_verdadeiro (array-like): rótulos reais (índices das classes).
        y_previsto (array-like): rótulos previstos pelo modelo.
        classes (list[str]): nomes das classes.
        caminho_png (str): onde salvar a figura.
        n (int): nº máximo de imagens exibidas.
        colunas (int): imagens por linha na grade.
        apenas_erros (bool): se True, mostra somente as classificações erradas.
        seed (int): semente para amostragem reprodutível.
    """
    caminhos = np.asarray(caminhos)
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_previsto = np.asarray(y_previsto)

    indices = np.arange(len(caminhos))
    if apenas_erros:
        # Seleciona apenas onde o modelo errou (real != previsto).
        indices = indices[y_verdadeiro != y_previsto]

    # Caso não haja nada a mostrar (ex.: nenhum erro), gera um aviso visual.
    if len(indices) == 0:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Nenhuma amostra para exibir\n(sem erros de classificação!)",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(caminho_png, dpi=150)
        plt.close()
        return

    # Amostragem reprodutível quando há mais imagens do que o limite.
    gerador = np.random.default_rng(seed)
    if len(indices) > n:
        indices = np.sort(gerador.choice(indices, size=n, replace=False))

    linhas = math.ceil(len(indices) / colunas)
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 2.4, linhas * 2.7))
    eixos = np.atleast_1d(eixos).ravel()

    for eixo, idx in zip(eixos, indices):
        imagem = cv2.imread(caminhos[idx])
        if imagem is not None:
            eixo.imshow(imagem[:, :, ::-1])  # BGR -> RGB
        real = classes[y_verdadeiro[idx]]
        previsto = classes[y_previsto[idx]]
        acertou = y_verdadeiro[idx] == y_previsto[idx]
        # Verde = acerto, vermelho = erro: leitura imediata na apresentação.
        cor = "green" if acertou else "red"
        marcador = "✓" if acertou else "✗"
        eixo.set_title(f"{marcador} Real: {real}\nPrev: {previsto}", color=cor, fontsize=9)
        eixo.axis("off")

    # Desliga os eixos que sobraram na última linha da grade.
    for eixo in eixos[len(indices):]:
        eixo.axis("off")

    titulo = "Erros de classificação" if apenas_erros else "Exemplos de predições no conjunto de teste"
    plt.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()
