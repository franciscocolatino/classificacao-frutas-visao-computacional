# Classificação de Frutas com Descritores de Forma (Fourier) e Textura (GLCM + LBP)

Projeto de Visão Computacional que classifica frutas combinando descritores
clássicos com um classificador SVM:

- Forma: Descritores de Fourier (a partir do contorno do objeto)
- Textura: GLCM (matriz de co-ocorrência) + LBP (padrões binários locais)
- Classificação: `StandardScaler` + `SVM` com kernel RBF (Scikit-Learn)

A ideia é montar um pipeline completo, do pré-processamento da imagem até a
avaliação do modelo, com relatórios e gráficos.

## Estrutura do projeto

```
.
├── dataset/                 # coloque aqui o dataset (veja LEIA-ME.txt)
├── src/
│   ├── preprocessamento.py  # resize, cinza, blur, Otsu, morfologia
│   ├── extrator_fourier.py  # descritor de forma (contorno + FFT)
│   ├── extrator_textura.py  # GLCM + LBP
│   ├── classificador.py     # vetor de características + pipeline SVM
│   └── avaliacao.py         # métricas, relatórios CSV e gráficos
├── main.py                  # pipeline completo com o dataset real
├── demo.py                  # demonstração sem dataset (imagens sintéticas)
├── preparar_dataset.py      # monta subconjunto do Fruits 360
├── apresentacao.ipynb       # notebook que importa de src/
├── requirements.txt
└── README.md
```

As saídas (gráficos e CSV) vão pra pasta `resultados/` (criada automaticamente).

## Como instalar

Requer Python 3.10+. Recomendo um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Como rodar

### Opção A: demonstração rápida (sem baixar nada)

Roda o pipeline inteiro usando imagens sintéticas (círculos, retângulos e
elipses como classes fictícias):

```bash
python demo.py
```

### Opção B: pipeline completo com o dataset real

Depois de deixar o dataset disponível:

```bash
python main.py
# ou:
python main.py --dataset dataset --resultados resultados --max_por_classe 100
```

Parâmetros:

| Parâmetro          | Padrão       | Descrição                                   |
|--------------------|--------------|---------------------------------------------|
| `--dataset`        | `dataset`    | Pasta com uma subpasta por classe           |
| `--resultados`     | `resultados` | Pasta de saída dos gráficos/CSV             |
| `--n_fourier`      | `15`         | Nº de coeficientes de Fourier               |
| `--max_por_classe` | (todas)      | Limita imagens por classe (acelera testes)  |
| `--teste`          | `0.3`        | Proporção do conjunto de teste              |

## Como baixar e preparar o dataset (Fruits 360)

O script de preparação pode baixar o dataset via `kagglehub`:

```bash
python preparar_dataset.py
```

Se a pasta local não existir, ele chama:

```python
import kagglehub
path = kagglehub.dataset_download("moltean/fruits")
```

e monta um subconjunto em `dataset/frutas/` com 8 frutas de formas e texturas
distintas (Banana, Apple, Orange, Strawberry, Avocado, Pineapple, Raspberry,
Lemon), 250 imagens por classe. Depois é só:

```bash
python main.py --dataset dataset/frutas
```

### Por que um subconjunto e não as 260 classes?

O Fruits 360 tem ~260 classes muito granulares (`Apple 10`, `Apple 11`, vários
tipos de cereja). Como os descritores são handcrafted (Fourier + GLCM + LBP) +
SVM, classes quase idênticas só atrapalhariam as métricas e deixariam a matriz
de confusão ilegível. Usar poucas classes distintas é uma escolha metodológica;
pra usar tudo, basta apontar `--dataset .../Training`.

Pra personalizar: `python preparar_dataset.py --classes Banana Apple Lemon --por_classe 300`.
Detalhes do download estão em [`dataset/LEIA-ME.txt`](dataset/LEIA-ME.txt).

### Resultado obtido (8 classes, 250 imgs/classe)

| Métrica   | Valor |
|-----------|-------|
| Acurácia  | 0,970 |
| Precisão  | 0,972 |
| Recall    | 0,970 |
| F1-score  | 0,970 |

Todas as 8 classes ficaram com F1 >= 0,90 (Lemon = 1,00).

## O pipeline passo a passo

### 1. Pré-processamento (`src/preprocessamento.py`)
- Redimensiona para `256x256`.
- Converte para cinza.
- Filtro Gaussiano pra reduzir ruído.
- Limiarização de Otsu pra separar objeto do fundo.
- Morfologia (abertura + fechamento) pra limpar a máscara.

### 2. Descritor de Forma - Fourier (`src/extrator_fourier.py`)
- Acha o maior contorno (`findContours`).
- Representa o contorno como sinal complexo `z = x + i*y`.
- Aplica FFT e pega os primeiros `N` coeficientes.
- Normaliza pra ficar invariante a translação (descarta o DC), rotação/ponto
  inicial (usa o módulo) e escala (divide pelo primeiro coeficiente).

### 3. Descritor de Textura (`src/extrator_textura.py`)
- GLCM: `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`
  (média de 4 direções).
- LBP: histograma normalizado dos padrões binários locais.
- Os dois são concatenados num vetor de textura.

### 4. Vetor de características final (`src/classificador.py`)
```
[ Fourier (forma) | GLCM (textura) | LBP (textura) ]
```

### 5. Classificador (`src/classificador.py`)
- `Pipeline` do Scikit-Learn com `StandardScaler` + `SVM` (kernel RBF).

### 6. Avaliação (`src/avaliacao.py`)
- Métricas: acurácia, precisão, recall e F1-score.
- Relatório CSV com métricas por classe (`relatorio_metricas.csv`).
- Matriz de confusão (heatmap com seaborn).

### 7. Visualizações geradas em `resultados/`

| Arquivo                          | Conteúdo                                            |
|----------------------------------|-----------------------------------------------------|
| `etapas_preprocessamento.png`    | Original, filtro e segmentação lado a lado          |
| `descritor_fourier.png`          | Assinatura de forma (distância ao centroide)        |
| `mapa_lbp.png`                   | Mapa de textura LBP                                 |
| `matriz_confusao.png`            | Matriz de confusão                                  |
| `f1_por_classe.png`              | Gráfico de barras de F1-score por classe            |
| `relatorio_metricas.csv`         | Relatório completo de métricas                      |

No `demo.py` as visualizações de etapas/Fourier/LBP são geradas para cada forma
(`etapas_circulo.png`, `fourier_elipse.png`, etc.).
