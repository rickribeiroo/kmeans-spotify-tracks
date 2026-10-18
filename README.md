# Segmentação de Músicas do Spotify com K-Means

Atividade diagnóstica de aprendizado não-supervisionado (Clustering). O objetivo é agrupar faixas musicais com base em quatro atributos sonoros (`danceability`, `energy`, `valence` e `tempo`) e determinar a quantidade ideal de clusters ($k$) por meio do **Coeficiente de Silhueta**.

---

## Dataset Original

- **Fonte:** [Spotify Tracks Dataset no Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Arquivo local:** `dataset.csv` (não incluído no repositório: baixe no link acima e coloque na raiz do projeto)
- **Atributos utilizados:**
  - `danceability`: Adequação da faixa para dançar (0.0 a 1.0).
  - `energy`: Intensidade e atividade perceptível da faixa (0.0 a 1.0).
  - `valence`: Positividade e humor transmitido pela música (0.0 = triste/tenso, 1.0 = alegre/eufórico).
  - `tempo`: Velocidade estimada da música em batimentos por minuto (BPM).

---

## Requisitos

- **Python:** 3.10 ou superior (testado com Python 3.12)
- **Bibliotecas:** listadas em `requirements.txt`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

---

## Como Rodar

1. **Criar e ativar o ambiente virtual (opcional se já tiver configurado):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Executar o pré-processamento:**
   Remove duplicatas de `track_id`, trata registros inconsistentes e padroniza as features com `StandardScaler` (gera `dataset_processed.csv` e `scaler.joblib`).
   ```powershell
   python pre_processing.py
   ```

3. **Executar o treinamento visual:**
   Abre a visualização em tempo real da curva de silhueta para $k \in [2, 10]$, executa a animação passo a passo da convergência dos centroides do K-Means e gera o relatório dos grupos.
   ```powershell
   python train.py
   ```

---

## Resultado: Quantidade Ideal de Clusters

A varredura pelo Coeficiente de Silhueta determinou **$k = 3$** como a quantidade ideal (maior valor de silhueta média):

| $k$ | Coeficiente de Silhueta | Diagnóstico |
| :---: | :---: | :--- |
| 2 | 0.2493 | Separação binária simples |
| **3** | **0.2577** | **Pico Global (Quantidade Ideal)** |
| 4 | 0.2370 | Queda na coesão dos grupos |
| 5 a 10 | $\le 0.2257$ | Fragmentação e sobreposição dos clusters |

### Perfis Encontrados ($k = 3$)

- **Cluster 0 (43,9%): Dançante e Alto Astral**
  - Músicas com alta dançabilidade (0.69) e alta valência (0.68). Predomínio de pop latino, salsa, house e afrobeat.
- **Cluster 1 (30,7%): Rápido, Intenso e Agressivo**
  - Músicas de ritmo acelerado (144 BPM), altíssima energia (0.81) e baixa valência (0.35). Predomínio de rock pesado, metal e grindcore.
- **Cluster 2 (25,4%): Calmo, Acústico e Melancólico**
  - Músicas lentas (104 BPM), baixa energia (0.33) e tom introspectivo (valência 0.26). Predomínio de música clássica, ambient e baladas acústicas.
