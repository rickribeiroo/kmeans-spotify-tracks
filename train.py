"""
Módulo de Treinamento Visual de Agrupamento K-Means (Spotify Tracks).

Executa:
1. Carregamento dos dados pré-processados (dataset_processed.csv)
2. Determinação visual de k em tempo real (Coeficiente de Silhueta e Método do Cotovelo)
3. Treinamento visual passo a passo do K-Means (animação da convergência dos centroides)
4. Interpretação dos clusters encontrados (perfis musicais, gêneros e faixas populares)
5. Geração e salvamento dos gráficos explicativos na pasta plots/
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pre_processing import FEATURES, SCALED_FEATURES, preprocess_spotify_dataset

RANDOM_STATE = 42
OUTPUT_DIR = Path('plots')


def get_or_create_processed_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega dataset_processed.csv se existir, ou executa o pré-processamento."""
    processed_file = Path('dataset_processed.csv')
    if not processed_file.exists():
        print("Arquivo 'dataset_processed.csv' não encontrado. Executando pre_processing.py...")
        df, X_scaled, _ = preprocess_spotify_dataset()
    else:
        print(f"Carregando dados pré-processados de: {processed_file.resolve()} ...")
        df = pd.read_csv(processed_file)
        X_scaled = df[SCALED_FEATURES].values
        print(f"Total de faixas carregadas: {len(df):,}")

    return df, X_scaled


def visual_evaluate_k(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    interactive: bool = True
) -> tuple[int, list[int], list[float], list[float]]:
    """
    Avalia valores de k de k_min a k_max com exibição visual em tempo real.
    Gera a curva de Coeficiente de Silhueta e de Inércia (Cotovelo).
    """
    print("\n" + "=" * 65)
    print("ETAPA 1: DETERMINAÇÃO VISUAL DO NÚMERO DE CLUSTERS (k)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    k_values = list(range(k_min, k_max + 1))
    silhouette_scores = []
    inertias = []

    # Configuração da janela gráfica em tempo real
    if interactive:
        plt.ion()
    
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    color_sil = '#1f77b4'
    color_ine = '#ff7f0e'

    ax1.set_xlabel("Número de Clusters (k)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coeficiente de Silhueta (quanto maior, melhor)", color=color_sil, fontsize=11)
    ax2.set_ylabel("Inércia / WCSS (quanto menor, mais compacto)", color=color_ine, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color_sil)
    ax2.tick_params(axis='y', labelcolor=color_ine)
    ax1.set_xlim(k_min - 0.5, k_max + 0.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    print(f"{'k':<4} | {'Silhueta Média':<16} | {'Inércia (WCSS)':<16} | Status")
    print("-" * 55)

    evaluated_k = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(
            X_scaled, labels, sample_size=10000, random_state=RANDOM_STATE
        )
        
        evaluated_k.append(k)
        silhouette_scores.append(sil)
        inertias.append(km.inertia_)

        print(f"{k:<4} | {sil:<16.4f} | {km.inertia_:<16.1f} | Calculado")

        # Atualização visual interativa do gráfico
        ax1.cla()
        ax2.cla()

        ax1.set_title(f"Avaliando k = {k}/{k_max} | Coeficiente de Silhueta vs Inércia", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Número de Clusters (k)", fontsize=11)
        ax1.set_ylabel("Coeficiente de Silhueta", color=color_sil, fontsize=11)
        ax2.set_ylabel("Inércia (WCSS)", color=color_ine, fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xlim(k_min - 0.5, k_max + 0.5)

        ax1.plot(evaluated_k, silhouette_scores, marker='o', color=color_sil, linewidth=2.5, label='Silhueta')
        ax2.plot(evaluated_k, inertias, marker='s', linestyle='--', color=color_ine, linewidth=2, label='Inércia')
        ax1.tick_params(axis='y', labelcolor=color_sil)
        ax2.tick_params(axis='y', labelcolor=color_ine)

        if interactive:
            plt.pause(0.15)

    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_idx]
    best_score = silhouette_scores[best_idx]

    # Destaca o melhor k no gráfico final
    ax1.scatter([best_k], [best_score], color='#d9534f', s=180, zorder=6, label=f'Melhor k={best_k}')
    ax1.annotate(
        f"Melhor k = {best_k}\n(Silhueta: {best_score:.4f})",
        xy=(best_k, best_score),
        xytext=(best_k + 0.4, best_score - 0.008),
        arrowprops=dict(arrowstyle="->", color='#d9534f', lw=2),
        fontweight='bold',
        color='#d9534f',
        fontsize=10
    )
    ax1.set_title(f"Resultado Final: k={best_k} selecionado pelo Coeficiente de Silhueta", fontsize=12, fontweight='bold')

    output_path = OUTPUT_DIR / 'visual_silhouette_elbow.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print("-" * 55)
    print(f"-> Melhor valor determinado: k = {best_k} (Silhueta = {best_score:.4f})")
    print(f"-> Gráfico salvo em: {output_path}")

    if interactive:
        plt.pause(1.0)
        plt.ioff()
        plt.close(fig)

    return best_k, k_values, silhouette_scores, inertias


def visual_train_kmeans(
    X_scaled: np.ndarray,
    df: pd.DataFrame,
    k: int,
    max_steps: int = 15,
    sample_size: int = 4000,
    interactive: bool = True
) -> tuple[KMeans, np.ndarray]:
    """
    Treina o K-Means final exibindo visualmente a convergência das iterações dos centroides.
    """
    print("\n" + "=" * 65)
    print(f"ETAPA 2: PROCESSO VISUAL DE TREINAMENTO DO K-MEANS (k = {k})")
    print("=" * 65)

    # Amostra de pontos para renderização fluida sem travamento de GUI
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_indices]

    # Features para visualização 2D (Danceability vs Energy)
    f1_idx, f2_idx = 0, 1  # 0: danceability, 1: energy

    if interactive:
        plt.ion()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", k)

    # Histórico de posições dos centroides para traçar o caminho percorrido
    centroid_history = {c: [] for c in range(k)}

    print(f"{'Iteração':<10} | {'Deslocamento dos Centroides':<30} | Status")
    print("-" * 55)

    final_kmeans = None
    final_labels = None

    for step in range(1, max_steps + 1):
        # Treina até a iteração 'step' com a mesma semente inicial
        kmeans_step = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            init='k-means++',
            n_init=1,
            max_iter=step
        )
        labels_sample = kmeans_step.fit_predict(X_sample)
        centroids = kmeans_step.cluster_centers_

        for c in range(k):
            centroid_history[c].append((centroids[c, f1_idx], centroids[c, f2_idx]))

        # Calcula o deslocamento em relação ao passo anterior
        if step > 1:
            prev_centroids = np.array([centroid_history[c][-2] for c in range(k)])
            curr_centroids = np.array([centroid_history[c][-1] for c in range(k)])
            shift = np.linalg.norm(curr_centroids - prev_centroids)
        else:
            shift = 1.0

        status = "Convergido!" if shift < 1e-4 else "Ajustando centroides..."
        print(f"{step:<10} | {shift:<30.5f} | {status}")

        # Atualização visual da dispersão e dos centroides
        ax.cla()
        ax.set_title(
            f"Treinamento do K-Means (Iteração {step}/{max_steps}) - k={k}\nVariação dos centroides: {shift:.5f}",
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("Danceability (Padronizado)", fontsize=11)
        ax.set_ylabel("Energy (Padronizado)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Plota os pontos atribuídos aos clusters atuais
        for c in range(k):
            mask = (labels_sample == c)
            ax.scatter(
                X_sample[mask, f1_idx],
                X_sample[mask, f2_idx],
                color=palette[c],
                alpha=0.4,
                s=25,
                label=f'Cluster {c}'
            )

            # Plota a trilha percorrida pelo centroide c
            history = np.array(centroid_history[c])
            ax.plot(
                history[:, 0],
                history[:, 1],
                color='black',
                linestyle=':',
                linewidth=1.8,
                alpha=0.7
            )

            # Posição atual do centroide
            ax.scatter(
                centroids[c, f1_idx],
                centroids[c, f2_idx],
                color=palette[c],
                edgecolor='black',
                s=250,
                marker='X',
                linewidth=2,
                zorder=10
            )

        ax.legend(loc='upper left')

        if interactive:
            plt.pause(0.3)

        final_kmeans = kmeans_step

        if shift < 1e-4 and step > 2:
            break

    # Treina o modelo final com todos os dados da base
    final_kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)

    output_path = OUTPUT_DIR / 'visual_kmeans_convergence.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n-> Gráfico da convergência do treinamento salvo em: {output_path}")

    if interactive:
        plt.pause(1.0)
        plt.ioff()
        plt.close(fig)

    return final_kmeans, final_labels


def profile_and_save_results(df: pd.DataFrame, cluster_labels: np.ndarray, k: int) -> pd.DataFrame:
    """Interpreta os clusters encontrados, exibe perfis musicais e salva o resumo em CSV."""
    print("\n" + "=" * 65)
    print(f"ETAPA 3: PERFILAMENTO DOS {k} CLUSTERS ENCONTRADOS")
    print("=" * 65)

    df_labeled = df.copy()
    df_labeled['cluster'] = cluster_labels

    summary = df_labeled.groupby('cluster')[FEATURES].mean().round(3)
    summary['count'] = df_labeled.groupby('cluster').size()
    summary['percent'] = (summary['count'] / len(df_labeled) * 100).round(1)

    print("\nMédias dos Atributos Originais por Cluster:")
    print(summary[['count', 'percent'] + FEATURES].to_string())

    # Rótulos automáticos interpretativos
    cluster_names = {}
    for c in range(k):
        d = summary.loc[c, 'danceability']
        e = summary.loc[c, 'energy']
        v = summary.loc[c, 'valence']

        traits = []
        if e > 0.65:
            traits.append("Alta Energia")
        elif e < 0.40:
            traits.append("Acústico/Calmo")
        else:
            traits.append("Energia Moderada")

        if d > 0.65:
            traits.append("Muito Dançante")
        elif d < 0.45:
            traits.append("Pouco Dançante")

        if v > 0.60:
            traits.append("Alto Astral")
        elif v < 0.35:
            traits.append("Melancólico")

        cluster_names[c] = " | ".join(traits)

    df_labeled['cluster_name'] = df_labeled['cluster'].map(cluster_names)

    print("\nPerfis Interpretativos:")
    for c, name in cluster_names.items():
        print(f" - Cluster {c}: {name} ({summary.loc[c, 'percent']}% da base)")

    print("\nTop 3 Gêneros por Cluster:")
    for c in range(k):
        top_genres = (
            df_labeled[df_labeled['cluster'] == c]['track_genre']
            .value_counts(normalize=True)
            .head(3)
            .apply(lambda p: f"{p*100:.1f}%")
            .to_dict()
        )
        print(f" - Cluster {c}: {top_genres}")

    # Salva resumo em CSV
    summary.to_csv('cluster_summary.csv')
    print("\n-> Resumo numérico salvo com sucesso em: cluster_summary.csv")

    return df_labeled


def visual_final_dashboard(df_labeled: pd.DataFrame, k: int):
    """Gera visualização final comparando dispersões e centroides médios."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", k)

    sample = df_labeled.sample(n=min(5000, len(df_labeled)), random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Dançabilidade vs Valência
    sns.scatterplot(
        data=sample,
        x='danceability',
        y='valence',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        ax=axes[0]
    )
    axes[0].set_title("Dançabilidade vs Valência (Humor Musical)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Danceability")
    axes[0].set_ylabel("Valence")

    # Gráfico 2: Tempo vs Energia
    sns.scatterplot(
        data=sample,
        x='tempo',
        y='energy',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        ax=axes[1]
    )
    axes[1].set_title("Tempo (BPM) vs Energia", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tempo (BPM)")
    axes[1].set_ylabel("Energy")

    output_path = OUTPUT_DIR / 'visual_final_clusters.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Painel final dos clusters salvo em: {output_path}")


def main():
    # 1. Carrega dados pré-processados
    df, X_scaled = get_or_create_processed_data()

    # 2. Avaliação visual de k em tempo real
    best_k, k_values, silhouette_scores, inertias = visual_evaluate_k(
        X_scaled, k_min=2, k_max=10, interactive=True
    )

    # 3. Treinamento visual passo a passo do K-Means
    kmeans_model, cluster_labels = visual_train_kmeans(
        X_scaled, df, k=best_k, max_steps=12, interactive=True
    )

    # 4. Perfilamento e interpretação dos clusters
    df_labeled = profile_and_save_results(df, cluster_labels, best_k)

    # 5. Painel visual final
    visual_final_dashboard(df_labeled, best_k)

    print("\n" + "=" * 65)
    print("PROCESSO DE TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 65)


if __name__ == '__main__':
    main()
