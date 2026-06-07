"""
DEMONSTRAÇÃO sem dataset — prova que o pipeline funciona de ponta a ponta.

Gera imagens SINTÉTICAS de três formas geométricas, tratadas como classes
fictícias:
    - "circulo"
    - "retangulo"
    - "elipse"

Cada imagem recebe variação aleatória de posição, tamanho e ruído, para
simular um dataset real. Em seguida roda exatamente o MESMO pipeline do
`main.py`: pré-processamento -> extração (Fourier + GLCM + LBP) ->
classificação (SVM) -> avaliação + visualizações.

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

# Classes sintéticas e quantas imagens gerar por classe.
CLASSES = ["circulo", "retangulo", "elipse"]
IMAGENS_POR_CLASSE = 40
TAMANHO = 256


def _rng(semente):
    """Cria um gerador de números aleatórios reprodutível (semente fixa)."""
    return np.random.default_rng(semente)


def gerar_imagem_sintetica(forma, gerador):
    """Desenha UMA forma geométrica sobre um fundo claro, com variações.

    O fundo é claro e a forma é escura — assim a limiarização de Otsu
    (com inversão) destaca a forma como objeto, igual a uma foto real.

    Args:
        forma (str): 'circulo', 'retangulo' ou 'elipse'.
        gerador (np.random.Generator): fonte de aleatoriedade.

    Returns:
        numpy.ndarray: imagem BGR (TAMANHO x TAMANHO x 3).
    """
    # Fundo claro (cinza ~230) com leve variação para não ser perfeito.
    fundo = int(gerador.integers(215, 240))
    imagem = np.full((TAMANHO, TAMANHO, 3), fundo, dtype=np.uint8)

    # Cor escura da forma (objeto).
    cor = tuple(int(c) for c in gerador.integers(20, 80, size=3))

    # Centro com pequena variação aleatória (invariância à translação).
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
        eixo_b = int(gerador.integers(35, 55))  # achatada -> diferente do círculo
        angulo = int(gerador.integers(0, 180))  # rotação aleatória
        cv2.ellipse(imagem, (cx, cy), (eixo_a, eixo_b), angulo, 0, 360, cor, thickness=-1)

    # Ruído gaussiano para simular imperfeições de captura. Isso também
    # exercita o filtro Gaussiano e a morfologia do pré-processamento.
    ruido = gerador.normal(0, 8, imagem.shape)
    imagem = np.clip(imagem.astype(np.float64) + ruido, 0, 255).astype(np.uint8)
    return imagem


def construir_dataset_sintetico():
    """Gera o dataset sintético inteiro e extrai as características.

    Returns:
        tuple: (X, y, classes, exemplo_por_classe)
            exemplo_por_classe: dict classe -> imagem (para visualização).
    """
    X, y = [], []
    exemplo_por_classe = {}

    for idx, forma in enumerate(CLASSES):
        # Semente diferente por classe -> reprodutível e variado.
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
                # Mesmo tratamento de contorno vazio do pipeline real.
                log.warning("  Imagem %d da classe '%s' ignorada: %s", i, forma, erro)

    return np.array(X), np.array(y), CLASSES, exemplo_por_classe


def main():
    pasta_resultados = "resultados"
    avaliacao.garantir_pasta(pasta_resultados)

    log.info("=" * 60)
    log.info("DEMONSTRAÇÃO — pipeline com imagens sintéticas (sem dataset)")
    log.info("=" * 60)

    log.info("ETAPA 1/4 — Gerando dataset sintético e extraindo características")
    X, y, classes, exemplos = construir_dataset_sintetico()
    log.info("Total: %d amostras | %d características | %d classes",
             X.shape[0], X.shape[1], len(classes))

    log.info("ETAPA 2/4 — Treino/teste e treinamento do SVM")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    modelo = classificador.criar_classificador()
    classificador.treinar(modelo, X_treino, y_treino)

    log.info("ETAPA 3/4 — Avaliação")
    y_previsto = classificador.prever(modelo, X_teste)
    metricas = avaliacao.calcular_metricas(y_teste, y_previsto)
    log.info("Acurácia : %.3f", metricas["acuracia"])
    log.info("Precisão : %.3f", metricas["precisao"])
    log.info("Recall   : %.3f", metricas["recall"])
    log.info("F1-score : %.3f", metricas["f1"])

    log.info("ETAPA 4/4 — Salvando relatórios e visualizações em '%s'", pasta_resultados)
    avaliacao.salvar_relatorio_csv(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "relatorio_metricas.csv")
    )
    avaliacao.plotar_matriz_confusao(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "matriz_confusao.png")
    )
    avaliacao.plotar_f1_por_classe(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "f1_por_classe.png")
    )

    # Visualizações do pipeline usando um exemplo de cada forma.
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
