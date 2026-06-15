"""
Utilitário para PREPARAR um subconjunto do Fruits 360 para o projeto.

O Fruits 360 original traz ~260 classes muito granulares (ex.: "Apple 10",
"Apple 11", várias cerejas quase idênticas) e em uma estrutura aninhada:

    dataset/archive/fruits-360_100x100/fruits-360/Training/<classe>/*.jpg

Para a apresentação, usar TODAS as classes seria inviável (treino lento e
matriz de confusão ilegível) e, como os descritores são handcrafted
(Fourier + GLCM + LBP) + SVM, classes quase idênticas só gerariam confusão.

Este script monta um subconjunto curado e no formato que o `main.py` espera
(`dataset/frutas/<Classe>/*.jpg`):
    - escolhe poucas classes com FORMAS e TEXTURAS bem distintas;
    - junta as variantes numeradas de cada fruta numa única classe
      (ex.: "Banana 1", "Banana 3", "Banana 4" -> "Banana");
    - amostra um número fixo de imagens por classe (balanceamento).

Uso:
    python preparar_dataset.py            # usa os padrões abaixo
    python preparar_dataset.py --por_classe 300 --classes Banana Apple Lemon
"""

import argparse
import os
import re
import shutil

try:
    import kagglehub
except ImportError:  # pragma: no cover - tratado em runtime com mensagem clara
    kagglehub = None

# Caminho padrão onde o Fruits 360 100x100 foi extraído (ajuste se necessário).
ORIGEM_PADRAO = os.path.join(
    "dataset", "archive", "fruits-360_100x100", "fruits-360", "Training"
)
DESTINO_PADRAO = os.path.join("dataset", "frutas")
NOME_DATASET_KAGGLE = "moltean/fruits"

# 8 frutas escolhidas para MAXIMIZAR a diversidade de forma e textura:
#   - Banana    : alongada (forma marcante)
#   - Apple     : redonda, superfície lisa
#   - Orange    : redonda, casca com leve textura
#   - Strawberry: cônica + sementes (textura granulada)
#   - Avocado   : oval/escura
#   - Pineapple : superfície muito rugosa (ótimo para GLCM/LBP)
#   - Raspberry : aglomerado granular (textura forte)
#   - Lemon     : oval, amarela
CLASSES_PADRAO = [
    "Banana", "Apple", "Orange", "Strawberry",
    "Avocado", "Pineapple", "Raspberry", "Lemon",
]

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")


def resolver_origem(origem):
    """Resolve a pasta de origem, baixando o dataset via Kaggle se necessário."""
    if os.path.isdir(origem):
        if os.path.isdir(os.path.join(origem, "Training")):
            return os.path.join(origem, "Training")
        subpastas = [
            nome for nome in os.listdir(origem)
            if os.path.isdir(os.path.join(origem, nome))
        ]
        if subpastas:
            possui_imagens_diretas = any(
                any(arq.lower().endswith(EXTENSOES) for arq in os.listdir(os.path.join(origem, sub)))
                for sub in subpastas
            )
            if possui_imagens_diretas:
                return origem

    if kagglehub is None:
        raise ImportError(
            "kagglehub não está instalado. Execute `pip install -r requirements.txt` "
            "ou instale `kagglehub` para baixar o dataset automaticamente."
        )

    print(f"[INFO] Pasta local não encontrada. Baixando '{NOME_DATASET_KAGGLE}' via kagglehub...")
    caminho_baixado = kagglehub.dataset_download(NOME_DATASET_KAGGLE)

    pasta_training = os.path.join(caminho_baixado, "Training")
    if os.path.isdir(pasta_training):
        return pasta_training

    for raiz, subpastas, _ in os.walk(caminho_baixado):
        if os.path.basename(raiz).lower() == "training" and subpastas:
            return raiz

    if os.path.isdir(caminho_baixado):
        subpastas = [
            nome for nome in os.listdir(caminho_baixado)
            if os.path.isdir(os.path.join(caminho_baixado, nome))
        ]
        if subpastas:
            possui_imagens_diretas = any(
                any(arq.lower().endswith(EXTENSOES) for arq in os.listdir(os.path.join(caminho_baixado, sub)))
                for sub in subpastas
            )
            if possui_imagens_diretas:
                return caminho_baixado

    raise FileNotFoundError(
        "O dataset foi baixado, mas a pasta 'Training' não foi encontrada "
        f"em '{caminho_baixado}'."
    )


def pastas_da_classe(origem, base):
    """Retorna as subpastas que pertencem a uma fruta 'base'.

    Considera a pasta com o nome exato ('Banana') e as variantes numeradas
    ('Banana 1', 'Banana 3'), mas NÃO variantes com outro nome
    ('Banana Red', 'Banana Lady Finger') — assim a classe fica coesa.
    """
    padrao = re.compile(rf"^{re.escape(base)}( \d+)?$")
    return [
        nome for nome in sorted(os.listdir(origem))
        if os.path.isdir(os.path.join(origem, nome)) and padrao.fullmatch(nome)
    ]


def coletar_imagens(origem, subpastas):
    """Lista os caminhos de todas as imagens das subpastas informadas."""
    caminhos = []
    for sub in subpastas:
        pasta = os.path.join(origem, sub)
        for arq in sorted(os.listdir(pasta)):
            if arq.lower().endswith(EXTENSOES):
                caminhos.append(os.path.join(pasta, arq))
    return caminhos


def amostrar(caminhos, n):
    """Amostra ~n imagens de forma uniforme (espaçada) na lista.

    Pegar em passo regular (em vez das n primeiras) cobre TODAS as variantes
    numeradas, dando mais variedade visual à classe.
    """
    if n is None or len(caminhos) <= n:
        return caminhos
    passo = len(caminhos) / n
    return [caminhos[int(i * passo)] for i in range(n)]


def preparar(origem, destino, classes, por_classe):
    """Monta o subconjunto curado copiando as imagens selecionadas."""
    origem = resolver_origem(origem)

    os.makedirs(destino, exist_ok=True)
    print(f"Origem : {origem}")
    print(f"Destino: {destino}\n")

    total_geral = 0
    for base in classes:
        subpastas = pastas_da_classe(origem, base)
        if not subpastas:
            print(f"[AVISO] Nenhuma pasta encontrada para a classe '{base}'. Pulando.")
            continue

        imagens = amostrar(coletar_imagens(origem, subpastas), por_classe)
        pasta_destino = os.path.join(destino, base)
        os.makedirs(pasta_destino, exist_ok=True)

        # Copia renomeando para evitar colisão de nomes entre variantes.
        for i, caminho in enumerate(imagens):
            ext = os.path.splitext(caminho)[1]
            shutil.copy2(caminho, os.path.join(pasta_destino, f"{base}_{i:04d}{ext}"))

        total_geral += len(imagens)
        print(f"{base:<12}: {len(subpastas):>2} variante(s) -> {len(imagens)} imagens copiadas")

    print(f"\nConcluído! {total_geral} imagens em '{destino}'.")
    print("Agora rode:  python main.py --dataset dataset/frutas")


def main():
    parser = argparse.ArgumentParser(description="Prepara subconjunto curado do Fruits 360.")
    parser.add_argument("--origem", default=ORIGEM_PADRAO, help="Pasta Training do Fruits 360.")
    parser.add_argument("--destino", default=DESTINO_PADRAO, help="Pasta de saída do subconjunto.")
    parser.add_argument("--classes", nargs="+", default=CLASSES_PADRAO, help="Frutas a incluir.")
    parser.add_argument("--por_classe", type=int, default=250, help="Imagens por classe.")
    args = parser.parse_args()
    preparar(args.origem, args.destino, args.classes, args.por_classe)


if __name__ == "__main__":
    main()
