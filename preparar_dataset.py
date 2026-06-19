"""
Prepara um subconjunto do Fruits 360 pro projeto.

O Fruits 360 original tem ~260 classes muito parecidas (Apple 10, Apple 11,
varias cerejas) numa estrutura aninhada. Aqui escolho poucas frutas bem
distintas, junto as variantes numeradas (Banana 1, Banana 3 -> Banana) e
copio um numero fixo de imagens por classe pro formato que o main.py espera
(dataset/frutas/<Classe>/*.jpg).

Uso:
    python preparar_dataset.py
    python preparar_dataset.py --por_classe 300 --classes Banana Apple Lemon
"""

import argparse
import os
import re
import shutil

try:
    import kagglehub
except ImportError:
    kagglehub = None

ORIGEM_PADRAO = os.path.join(
    "dataset", "archive", "fruits-360_100x100", "fruits-360", "Training"
)
DESTINO_PADRAO = os.path.join("dataset", "frutas")
NOME_DATASET_KAGGLE = "moltean/fruits"

# frutas escolhidas por terem formas e texturas bem diferentes entre si
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
    """Pega as subpastas de uma fruta: o nome exato e as variantes numeradas
    (Banana, Banana 1, Banana 3), mas nao 'Banana Red' etc."""
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
    """Pega ~n imagens espacadas na lista (em vez das n primeiras), assim
    cobre as varias variantes numeradas."""
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
