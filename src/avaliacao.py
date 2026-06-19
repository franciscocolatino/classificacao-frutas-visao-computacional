"""
Metricas e visualizacoes.

Calcula as metricas (acuracia, precisao, recall, F1), salva o relatorio
em CSV e gera os graficos (matriz de confusao, F1 por classe, etapas do
pre-processamento, assinatura de Fourier e mapa LBP).

Usa o backend Agg do matplotlib pra funcionar sem interface grafica.
"""

import math
import os

import cv2
import matplotlib

matplotlib.use("Agg")
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
    """Cria a pasta de resultados se nao existir."""
    os.makedirs(pasta, exist_ok=True)


def calcular_metricas(y_verdadeiro, y_previsto):
    """Metricas globais. Uso average weighted por causa do desbalanceamento."""
    return {
        "acuracia": accuracy_score(y_verdadeiro, y_previsto),
        "precisao": precision_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
        "recall": recall_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
        "f1": f1_score(y_verdadeiro, y_previsto, average="weighted", zero_division=0),
    }


def salvar_relatorio_csv(y_verdadeiro, y_previsto, classes, caminho_csv):
    """Salva o classification_report (metricas por classe) em CSV."""
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
    """Matriz de confusao como heatmap."""
    matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=list(range(len(classes))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matriz,
        annot=True,
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
    """Grafico de barras com o F1 de cada classe."""
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

    # escreve o valor em cima de cada barra
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
    """Mostra original, apos filtro e apos segmentacao lado a lado."""
    fig, eixos = plt.subplots(1, 3, figsize=(14, 5))

    # OpenCV usa BGR, matplotlib espera RGB
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
    """Plota a assinatura de forma (distancia ao centroide)."""
    assinatura = extrator_fourier.assinatura_distancia_centroide(dados["mascara"])

    plt.figure(figsize=(9, 4))
    plt.plot(assinatura, color="darkorange", linewidth=2)
    plt.title("Descritor de Forma - Assinatura (distância ao centroide)")
    plt.xlabel("Ponto ao longo do contorno")
    plt.ylabel("Distância normalizada")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()


def plotar_mapa_lbp(dados, caminho_png):
    """Plota o mapa LBP ao lado da imagem em cinza."""
    lbp = extrator_textura.mapa_lbp(dados["cinza"])

    fig, eixos = plt.subplots(1, 2, figsize=(11, 5))
    eixos[0].imshow(dados["cinza"], cmap="gray")
    eixos[0].set_title("Imagem em Cinza")
    eixos[0].axis("off")

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
    """Galeria de imagens de teste com o rotulo real e o previsto.

    Acerto fica verde, erro fica vermelho. Se apenas_erros=True mostra so
    onde o modelo errou.
    """
    caminhos = np.asarray(caminhos)
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_previsto = np.asarray(y_previsto)

    indices = np.arange(len(caminhos))
    if apenas_erros:
        indices = indices[y_verdadeiro != y_previsto]

    # nada pra mostrar (ex.: nenhum erro)
    if len(indices) == 0:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Nenhuma amostra para exibir\n(sem erros de classificação!)",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(caminho_png, dpi=150)
        plt.close()
        return

    # se tem mais imagens que o limite, sorteia de forma reprodutivel
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
        cor = "green" if acertou else "red"
        marcador = "✓" if acertou else "✗"
        eixo.set_title(f"{marcador} Real: {real}\nPrev: {previsto}", color=cor, fontsize=9)
        eixo.axis("off")

    # desliga os eixos que sobraram na ultima linha
    for eixo in eixos[len(indices):]:
        eixo.axis("off")

    titulo = "Erros de classificação" if apenas_erros else "Exemplos de predições no conjunto de teste"
    plt.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150)
    plt.close()
