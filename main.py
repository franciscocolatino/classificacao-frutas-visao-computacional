"""
Pipeline completo de classificacao de frutas (forma + textura).

Le as imagens do dataset, pre-processa, extrai as caracteristicas
(Fourier + GLCM + LBP), treina o SVM e avalia no conjunto de teste,
gerando os relatorios e graficos na pasta de resultados.

Uso:
    python main.py
    python main.py --dataset CAMINHO --resultados PASTA
"""

import argparse
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# permite rodar `python main.py` de dentro da pasta do projeto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import avaliacao, classificador, preprocessamento  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def listar_classes(pasta_dataset):
    """Lista as subpastas do dataset (cada subpasta e uma classe)."""
    if not os.path.isdir(pasta_dataset):
        return []
    classes = [
        nome
        for nome in sorted(os.listdir(pasta_dataset))
        if os.path.isdir(os.path.join(pasta_dataset, nome))
    ]
    return classes


def carregar_dataset(pasta_dataset, n_fourier=15, max_por_classe=None, retornar_caminhos=False):
    """Percorre o dataset e extrai as caracteristicas de cada imagem.

    Devolve (X, y, classes). Se retornar_caminhos=True, devolve tambem a
    lista de caminhos alinhada com X e y (util pra mostrar exemplos depois).
    """
    classes = listar_classes(pasta_dataset)
    if not classes:
        raise FileNotFoundError(
            f"Nenhuma classe (subpasta) encontrada em '{pasta_dataset}'. "
            "Baixe o dataset (veja dataset/LEIA-ME.txt) ou rode `python demo.py`."
        )

    X, y, caminhos = [], [], []
    total_ok, total_falhas = 0, 0

    for idx, classe in enumerate(classes):
        pasta_classe = os.path.join(pasta_dataset, classe)
        arquivos = [
            f for f in sorted(os.listdir(pasta_classe))
            if f.lower().endswith(EXTENSOES_VALIDAS)
        ]
        if max_por_classe:
            arquivos = arquivos[:max_por_classe]

        log.info("Classe '%s' (%d): %d imagens", classe, idx, len(arquivos))

        for arquivo in arquivos:
            caminho = os.path.join(pasta_classe, arquivo)
            try:
                # imagem que nao abre ou mascara vazia: pula sem derrubar tudo
                vetor = classificador.extrair_caracteristicas_de_caminho(
                    caminho, n_fourier=n_fourier
                )
                X.append(vetor)
                y.append(idx)
                caminhos.append(caminho)
                total_ok += 1
            except (FileNotFoundError, ValueError) as erro:
                total_falhas += 1
                log.warning("  Ignorando '%s': %s", arquivo, erro)

    if not X:
        raise RuntimeError("Nenhuma imagem válida foi processada. Verifique o dataset.")

    log.info("Extração concluída: %d ok, %d ignoradas.", total_ok, total_falhas)
    if retornar_caminhos:
        return np.array(X), np.array(y), classes, caminhos
    return np.array(X), np.array(y), classes


def gerar_visualizacoes_amostra(pasta_dataset, classes, pasta_resultados):
    """Gera as figuras do pipeline a partir de uma imagem de exemplo."""
    for classe in classes:
        pasta_classe = os.path.join(pasta_dataset, classe)
        for arquivo in sorted(os.listdir(pasta_classe)):
            if not arquivo.lower().endswith(EXTENSOES_VALIDAS):
                continue
            caminho = os.path.join(pasta_classe, arquivo)
            try:
                dados = preprocessamento.preprocessar(caminho)
                avaliacao.plotar_etapas_preprocessamento(
                    dados, os.path.join(pasta_resultados, "etapas_preprocessamento.png")
                )
                avaliacao.plotar_assinatura_fourier(
                    dados, os.path.join(pasta_resultados, "descritor_fourier.png")
                )
                avaliacao.plotar_mapa_lbp(
                    dados, os.path.join(pasta_resultados, "mapa_lbp.png")
                )
                log.info("Visualizações de amostra geradas a partir de '%s/%s'.", classe, arquivo)
                return
            except (FileNotFoundError, ValueError):
                continue
    log.warning("Não foi possível gerar visualizações de amostra (sem imagem válida).")


def main():
    parser = argparse.ArgumentParser(description="Classificação de Frutas (Forma + Textura).")
    parser.add_argument("--dataset", default="dataset", help="Pasta do dataset.")
    parser.add_argument("--resultados", default="resultados", help="Pasta de saída.")
    parser.add_argument("--n_fourier", type=int, default=15, help="Coeficientes de Fourier.")
    parser.add_argument("--max_por_classe", type=int, default=None, help="Limite de imagens/classe.")
    parser.add_argument("--teste", type=float, default=0.3, help="Proporção do conjunto de teste.")
    args = parser.parse_args()

    pasta_resultados = args.resultados
    avaliacao.garantir_pasta(pasta_resultados)

    log.info("Carregando e processando o dataset...")
    X, y, classes = carregar_dataset(
        args.dataset, n_fourier=args.n_fourier, max_por_classe=args.max_por_classe
    )
    log.info("Total: %d amostras | %d características por amostra | %d classes",
             X.shape[0], X.shape[1], len(classes))

    # stratify mantem a proporcao das classes em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=args.teste, random_state=42, stratify=y
    )
    log.info("Treino: %d amostras | Teste: %d amostras", len(X_treino), len(X_teste))

    log.info("Treinando o classificador (StandardScaler + SVM RBF)...")
    modelo = classificador.criar_classificador()
    classificador.treinar(modelo, X_treino, y_treino)

    y_previsto = classificador.prever(modelo, X_teste)
    metricas = avaliacao.calcular_metricas(y_teste, y_previsto)
    log.info("Acurácia : %.3f", metricas["acuracia"])
    log.info("Precisão : %.3f", metricas["precisao"])
    log.info("Recall   : %.3f", metricas["recall"])
    log.info("F1-score : %.3f", metricas["f1"])

    log.info("Gerando relatórios e visualizações em '%s'...", pasta_resultados)
    avaliacao.salvar_relatorio_csv(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "relatorio_metricas.csv")
    )
    avaliacao.plotar_matriz_confusao(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "matriz_confusao.png")
    )
    avaliacao.plotar_f1_por_classe(
        y_teste, y_previsto, classes, os.path.join(pasta_resultados, "f1_por_classe.png")
    )
    gerar_visualizacoes_amostra(args.dataset, classes, pasta_resultados)

    log.info("Concluído! Confira os arquivos em '%s/'.", pasta_resultados)


if __name__ == "__main__":
    main()
