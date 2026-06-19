"""
Demonstracao sem dataset.

Gera imagens sinteticas de tres formas (circulo, retangulo, elipse) e
roda o mesmo pipeline do main.py. Serve pra testar tudo de ponta a ponta
mesmo sem o dataset baixado.

Uso:
    python demo.py
"""

import logging
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import avaliacao, classificador, preprocessamento  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

CLASSES = ["circulo", "retangulo", "elipse"]
IMAGENS_POR_CLASSE = 40
TAMANHO = 256


def _rng(semente):
    """Gerador aleatorio com semente fixa (reprodutivel)."""
    return np.random.default_rng(semente)


def gerar_imagem_sintetica(forma, gerador):
    """Desenha uma forma escura sobre fundo claro, com variacoes aleatorias."""
    # fundo claro (parecido com foto), com leve variacao
    fundo = int(gerador.integers(215, 240))
    imagem = np.full((TAMANHO, TAMANHO, 3), fundo, dtype=np.uint8)

    cor = tuple(int(c) for c in gerador.integers(20, 80, size=3))

    cx = int(gerador.integers(100, 156))
    cy = int(gerador.integers(100, 156))

    if forma == "circulo":
        raio = int(gerador.integers(50, 80))
        cv2.circle(imagem, (cx, cy), raio, cor, thickness=-1)

    elif forma == "retangulo":
        largura = int(gerador.integers(70, 110))
        altura = int(gerador.integers(70, 110))
        p1 = (cx - largura // 2, cy - altura // 2)
        p2 = (cx + largura // 2, cy + altura // 2)
        cv2.rectangle(imagem, p1, p2, cor, thickness=-1)

    elif forma == "elipse":
        eixo_a = int(gerador.integers(70, 95))
        eixo_b = int(gerador.integers(35, 55))  # achatada, pra diferenciar do circulo
        angulo = int(gerador.integers(0, 180))
        cv2.ellipse(imagem, (cx, cy), (eixo_a, eixo_b), angulo, 0, 360, cor, thickness=-1)

    # ruido pra simular imperfeicao de captura
    ruido = gerador.normal(0, 8, imagem.shape)
    imagem = np.clip(imagem.astype(np.float64) + ruido, 0, 255).astype(np.uint8)
    return imagem


def construir_dataset_sintetico():
    """Gera o dataset sintetico e ja extrai as caracteristicas."""
    X, y = [], []
    exemplo_por_classe = {}

    for idx, forma in enumerate(CLASSES):
        gerador = _rng(semente=1000 + idx)
        log.info("Gerando %d imagens sintéticas da classe '%s'...", IMAGENS_POR_CLASSE, forma)

        for i in range(IMAGENS_POR_CLASSE):
            imagem = gerar_imagem_sintetica(forma, gerador)
            if forma not in exemplo_por_classe:
                exemplo_por_classe[forma] = imagem

            try:
                dados = preprocessamento.preprocessar_array(imagem)
                vetor = classificador.montar_vetor_caracteristicas(dados)
                X.append(vetor)
                y.append(idx)
            except ValueError as erro:
                log.warning("  Imagem %d da classe '%s' ignorada: %s", i, forma, erro)

    return np.array(X), np.array(y), CLASSES, exemplo_por_classe


def main():
    pasta_resultados = "resultados"
    avaliacao.garantir_pasta(pasta_resultados)

    log.info("Demonstração com imagens sintéticas (sem dataset).")

    X, y, classes, exemplos = construir_dataset_sintetico()
    log.info("Total: %d amostras | %d características | %d classes",
             X.shape[0], X.shape[1], len(classes))

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    modelo = classificador.criar_classificador()
    classificador.treinar(modelo, X_treino, y_treino)

    y_previsto = classificador.prever(modelo, X_teste)
    metricas = avaliacao.calcular_metricas(y_teste, y_previsto)
    log.info("Acurácia : %.3f", metricas["acuracia"])
    log.info("Precisão : %.3f", metricas["precisao"])
    log.info("Recall   : %.3f", metricas["recall"])
    log.info("F1-score : %.3f", metricas["f1"])

    avaliacao.salvar_relatorio_csv(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "relatorio_metricas.csv")
    )
    avaliacao.plotar_matriz_confusao(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "matriz_confusao.png")
    )
    avaliacao.plotar_f1_por_classe(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "f1_por_classe.png")
    )

    # visualizacoes usando um exemplo de cada forma
    for forma, imagem in exemplos.items():
        dados = preprocessamento.preprocessar_array(imagem)
        avaliacao.plotar_etapas_preprocessamento(
            dados, os.path.join(pasta_resultados, f"etapas_{forma}.png")
        )
        avaliacao.plotar_assinatura_fourier(
            dados, os.path.join(pasta_resultados, f"fourier_{forma}.png")
        )
        avaliacao.plotar_mapa_lbp(
            dados, os.path.join(pasta_resultados, f"lbp_{forma}.png")
        )

    log.info("Demonstração concluída! Veja os arquivos em '%s/'.", pasta_resultados)


if __name__ == "__main__":
    main()
