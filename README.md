# Classificação de Frutas usando Descritores de Forma (Fourier) e Textura (GLCM + LBP)

Projeto de **Visão Computacional** que classifica frutas combinando dois tipos de
descritores clássicos e um classificador SVM:

- **Forma** → Descritores de Fourier (a partir do contorno do objeto)
- **Textura** → GLCM (matriz de co-ocorrência) + LBP (padrões binários locais)
- **Classificação** → `StandardScaler` + `SVM` com kernel RBF (Scikit-Learn)

O objetivo é mostrar, de forma clara e didática, um **pipeline completo de visão
computacional**: do pré-processamento da imagem até a avaliação do modelo, com
relatórios e visualizações prontos para apresentação.

---

## 📁 Estrutura do projeto

```
.                            # raiz do projeto
├── dataset/                 # coloque aqui o dataset (veja LEIA-ME.txt)
├── src/
│   ├── preprocessamento.py  # redimensionar, cinza, blur, Otsu, morfologia
│   ├── extrator_fourier.py  # descritor de forma (contorno + FFT)
│   ├── extrator_textura.py  # GLCM + LBP
│   ├── classificador.py     # vetor de características + pipeline SVM
│   └── avaliacao.py         # métricas, relatórios CSV e gráficos
├── main.py                  # pipeline completo com o dataset real
├── demo.py                  # demonstração SEM dataset (imagens sintéticas)
├── preparar_dataset.py      # monta subconjunto curado do Fruits 360
├── apresentacao.ipynb       # notebook de apresentação (importa de src/)
├── requirements.txt
└── README.md
```

As saídas (gráficos e CSV) são geradas na pasta `resultados/` (criada automaticamente).

---

## 🔧 Como instalar

Requer **Python 3.10+**. Recomenda-se um ambiente virtual:

```bash
# 1. criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. instalar as dependências (versões fixadas)
pip install -r requirements.txt
```

---

## ▶️ Como rodar

### Opção A — Demonstração rápida (sem baixar nada)

Prova que o pipeline funciona de ponta a ponta usando **imagens sintéticas**
(círculos, retângulos e elipses como classes fictícias):

```bash
python demo.py
```

Ótimo para apresentar em sala mesmo sem internet ou dataset.

### Opção B — Pipeline completo com o dataset real

Depois de baixar o dataset (próxima seção):

```bash
python main.py
# ou personalizando:
python main.py --dataset dataset --resultados resultados --max_por_classe 100
```

Parâmetros úteis:

| Parâmetro          | Padrão       | Descrição                                   |
|--------------------|--------------|---------------------------------------------|
| `--dataset`        | `dataset`    | Pasta com uma subpasta por classe           |
| `--resultados`     | `resultados` | Pasta de saída dos gráficos/CSV             |
| `--n_fourier`      | `15`         | Nº de coeficientes de Fourier               |
| `--max_por_classe` | (todas)      | Limita imagens por classe (acelera testes)  |
| `--teste`          | `0.3`        | Proporção do conjunto de teste              |

---

## 📥 Como baixar e preparar o dataset (Fruits 360)

1. Acesse o **[Fruits 360 no Kaggle](https://www.kaggle.com/datasets/moltean/fruits)**.
2. Baixe e extraia o `.zip` (precisa de conta no Kaggle). Ele vem aninhado, ex.:
   `dataset/archive/fruits-360_100x100/fruits-360/Training/<classe>/*.jpg`.
3. Rode o utilitário que monta um **subconjunto curado** no formato esperado:

```bash
python preparar_dataset.py
```

Isso cria `dataset/frutas/` com 8 frutas de **formas e texturas distintas**
(Banana, Apple, Orange, Strawberry, Avocado, Pineapple, Raspberry, Lemon),
250 imagens por classe. Depois é só:

```bash
python main.py --dataset dataset/frutas
```

### Por que um subconjunto curado (e não as 260 classes)?
- O Fruits 360 tem ~260 classes **muito granulares** (`Apple 10`, `Apple 11`,
  7 tipos de cereja...). Como os descritores são *handcrafted* (Fourier + GLCM + LBP)
  + SVM, classes quase idênticas só gerariam confusão e **piorariam** as métricas.
- Escolher poucas classes **distintas** é prática acadêmica padrão e deixa a
  matriz de confusão legível. O pipeline continua escalável: para usar tudo,
  basta apontar `--dataset .../Training`.

> 💡 Personalize as classes/quantidade: `python preparar_dataset.py --classes Banana Apple Lemon --por_classe 300`.
> Detalhes do download (e Kaggle CLI) em [`dataset/LEIA-ME.txt`](dataset/LEIA-ME.txt).

### Resultado obtido (8 classes, 250 imgs/classe)
| Métrica   | Valor |
|-----------|-------|
| Acurácia  | 0,970 |
| Precisão  | 0,972 |
| Recall    | 0,970 |
| F1-score  | 0,970 |

Todas as 8 classes atingiram **F1 ≥ 0,90** (Lemon = 1,00).

---

## 🧠 O pipeline explicado (passo a passo)

### 1. Pré-processamento (`src/preprocessamento.py`)
- **Redimensiona** para `256x256` → padroniza a escala.
- **Converte para cinza** → forma e textura não dependem de cor.
- **Filtro Gaussiano** → reduz ruído antes de segmentar.
- **Limiarização de Otsu** → separa objeto do fundo automaticamente.
- **Morfologia (abertura + fechamento)** → limpa a máscara (remove ruído e fecha buracos).

### 2. Descritor de Forma — Fourier (`src/extrator_fourier.py`)
- Acha o **maior contorno** (`findContours`).
- Representa o contorno como **sinal complexo** `z = x + i·y`.
- Aplica **FFT** e extrai os primeiros `N` coeficientes.
- **Normalizações** para invariância a:
  - **Translação** → descarta o coeficiente DC (índice 0);
  - **Rotação / ponto inicial** → usa o **módulo** dos coeficientes;
  - **Escala** → divide pelo primeiro coeficiente não nulo.

### 3. Descritor de Textura (`src/extrator_textura.py`)
- **GLCM**: extrai `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`
  (média de 4 direções → aproximadamente invariante à rotação).
- **LBP**: histograma normalizado dos padrões binários locais (robusto à iluminação).
- Os dois são **concatenados** em um único vetor de textura.

### 4. Vetor de características final (`src/classificador.py`)
```
[ Fourier (forma) | GLCM (textura) | LBP (textura) ]
```

### 5. Classificador (`src/classificador.py`)
- `Pipeline` do Scikit-Learn:
  - **StandardScaler** → padroniza escalas (essencial para o SVM);
  - **SVM (kernel RBF)** → fronteiras de decisão não lineares.

### 6. Avaliação (`src/avaliacao.py`)
- **Métricas**: acurácia, precisão, recall e F1-score.
- **Relatório CSV**: métricas por classe (`relatorio_metricas.csv`).
- **Matriz de confusão**: heatmap colorido (seaborn).

### 7. Visualizações geradas em `resultados/`
| Arquivo                          | Conteúdo                                            |
|----------------------------------|-----------------------------------------------------|
| `etapas_preprocessamento.png`    | Original → filtro → segmentação (lado a lado)       |
| `descritor_fourier.png`          | Assinatura de forma (distância ao centroide)        |
| `mapa_lbp.png`                   | Mapa de textura LBP                                 |
| `matriz_confusao.png`            | Matriz de confusão colorida                         |
| `f1_por_classe.png`              | Gráfico de barras de F1-score por classe            |
| `relatorio_metricas.csv`         | Relatório completo de métricas                      |

> No `demo.py`, as visualizações de etapas/Fourier/LBP são geradas para **cada forma**
> (`etapas_circulo.png`, `fourier_elipse.png`, etc.).

---

## ✅ Boas práticas adotadas
- Código modular, com **docstrings** e **comentários em português** explicando o *porquê* de cada técnica.
- **Tratamento de exceções** em pontos críticos (leitura de imagem e contorno vazio): imagens problemáticas são puladas sem derrubar a execução.
- **Logs claros** de progresso em cada etapa do `main.py`.
- Dependências com **versões fixadas** em `requirements.txt`.
- Backend `Agg` do matplotlib → funciona em servidores sem interface gráfica.
