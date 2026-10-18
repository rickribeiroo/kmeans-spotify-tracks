"""
Segmentação de Músicas no Spotify usando K-Means e Coeficiente de Silhueta.

Atividade diagnóstica de Agrupamento (Clustering).
Features utilizadas: danceability, energy, valence, tempo.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ['danceability', 'energy', 'valence', 'tempo']
RANDOM_STATE = 42
OUTPUT_DIR = Path('plots')


def load_and_preprocess_data(filepath: str = 'dataset.csv') -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler]:
    """Carrega o dataset, remove duplicatas e nulos, e normaliza as features."""
    print("=" * 60)
    print("1. CARREGAMENTO E PRÉ-PROCESSAMENTO")
    print("=" * 60)
    
    df_raw = pd.read_csv(filepath)
    print(f"Total de registros brutos: {len(df_raw):,}")

    # Desduplicação por track_id conforme alinhado
    df = df_raw.drop_duplicates(subset='track_id').copy()
    print(f"Total após remover duplicatas (track_id único): {len(df):,}")

    # Limpeza de inconsistências (ex: tempo zerado ou nulos)
    df = df.dropna(subset=FEATURES)
    df = df[df['tempo'] > 0].copy()
    print(f"Total após remover tempos zerados/nulos: {len(df):,}")

    X = df[FEATURES].copy()

    # Padronização (Z-Score) essencial devido à diferença de escala do 'tempo' (BPM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X, X_scaled, scaler


def evaluate_k_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, list[int], list[float], list[float]]:
    """Testa valores de k de k_min até k_max, calculando Inércia e Coeficiente de Silhueta."""
    print("\n" + "=" * 60)
    print("2. AVALIAÇÃO DO NÚMERO DE CLUSTERS (k) COM SILHUETA E INÉRCIA")
    print("=" * 60)

    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouette_scores = []

    print(f"{'k':<4} | {'Silhueta Média':<16} | {'Inércia (Elbow)':<16}")
    print("-" * 42)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        
        # Amostragem para cálculo rápido e estatisticamente robusto da silhueta
        sil_score = silhouette_score(
            X_scaled, labels, sample_size=10000, random_state=RANDOM_STATE
        )
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(sil_score)
        print(f"{k:<4} | {sil_score:<16.4f} | {kmeans.inertia_:<16.1f}")

    best_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_idx]
    best_score = silhouette_scores[best_idx]

    print("-" * 42)
    print(f"-> Melhor k determinado pelo Coeficiente de Silhueta: k = {best_k} (score: {best_score:.4f})")

    return best_k, k_values, silhouette_scores, inertias


def plot_metrics(k_values: list[int], silhouette_scores: list[float], inertias: list[float], best_k: int):
    """Gera gráfico comparativo de Silhouette Score e Inércia (Elbow)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_sil = '#2b5c8f'
    ax1.set_title("Determinação de k: Coeficiente de Silhueta vs Método do Cotovelo", fontsize=13, pad=12, fontweight='bold')
    ax1.set_xlabel("Número de Clusters (k)", fontsize=11)
    ax1.set_ylabel("Coeficiente de Silhueta (quanto maior, melhor)", color=color_sil, fontsize=11)
    line1 = ax1.plot(k_values, silhouette_scores, marker='o', color=color_sil, linewidth=2.5, label='Silhueta')
    ax1.tick_params(axis='y', labelcolor=color_sil)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Destaque no melhor k
    best_score = silhouette_scores[k_values.index(best_k)]
    ax1.scatter([best_k], [best_score], color='#d9534f', s=160, zorder=5, label=f'Melhor k={best_k} (Silhueta={best_score:.4f})')
    ax1.annotate(f"Máximo: k={best_k}\n({best_score:.3f})",
                 xy=(best_k, best_score),
                 xytext=(best_k + 0.3, best_score - 0.005),
                 arrowprops=dict(arrowstyle="->", color='#d9534f', lw=1.5),
                 fontweight='bold', color='#d9534f')

    # Eixo secundário para inércia
    ax2 = ax1.twinx()
    color_ine = '#e67e22'
    ax2.set_ylabel("Inércia / WCSS (quanto menor, mais compacto)", color=color_ine, fontsize=11)
    line2 = ax2.plot(k_values, inertias, marker='s', linestyle='--', color=color_ine, linewidth=2, label='Inércia')
    ax2.tick_params(axis='y', labelcolor=color_ine)

    # Legenda combinada
    lines = line1 + line2 + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d9534f', markersize=9)]
    labels = ['Silhueta', 'Inércia', f'Melhor k={best_k}']
    ax1.legend(lines, labels, loc='center right')

    output_path = OUTPUT_DIR / 'silhouette_elbow_curve.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico salvo em: {output_path}")


def train_final_model(X_scaled: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """Treina o modelo K-Means final com o número ótimo de clusters."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels


def interpret_and_profile_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, best_k: int) -> pd.DataFrame:
    """Calcula médias, identifica características dominantes, gêneros e exemplos de faixas."""
    print("\n" + "=" * 60)
    print(f"3. PERFILAMENTO E INTERPRETAÇÃO DOS {best_k} CLUSTERS")
    print("=" * 60)

    df_labeled = df.copy()
    df_labeled['cluster'] = cluster_labels

    # Estatísticas na escala original
    summary = df_labeled.groupby('cluster')[FEATURES].mean().round(3)
    summary['count'] = df_labeled.groupby('cluster').size()
    summary['percent'] = (summary['count'] / len(df_labeled) * 100).round(1)

    print("\nMédias das variáveis por cluster (escala original):")
    print(summary[['count', 'percent'] + FEATURES].to_string())

    # Sugestão de personas/rótulos automáticos com base nos centroides
    cluster_names = {}
    for c in range(best_k):
        d = summary.loc[c, 'danceability']
        e = summary.loc[c, 'energy']
        v = summary.loc[c, 'valence']
        t = summary.loc[c, 'tempo']

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
            traits.append("Alto Astral/Feliz")
        elif v < 0.35:
            traits.append("Melancólico/Triste")

        cluster_names[c] = " | ".join(traits)

    df_labeled['cluster_name'] = df_labeled['cluster'].map(cluster_names)

    print("\nPerfis Interpretativos:")
    for c, name in cluster_names.items():
        print(f" - Cluster {c}: {name} ({summary.loc[c, 'percent']}% das faixas)")

    # Gêneros mais frequentes por cluster
    print("\nTop 4 Gêneros mais presentes por cluster:")
    for c in range(best_k):
        top_genres = (
            df_labeled[df_labeled['cluster'] == c]['track_genre']
            .value_counts(normalize=True)
            .head(4)
            .apply(lambda p: f"{p*100:.1f}%")
            .to_dict()
        )
        print(f" - Cluster {c}: {top_genres}")

    # Amostra de faixas populares
    print("\nExemplos de faixas populares em cada cluster:")
    for c in range(best_k):
        samples = (
            df_labeled[df_labeled['cluster'] == c]
            .sort_values(by='popularity', ascending=False)
            [['artists', 'track_name', 'track_genre', 'popularity']]
            .head(3)
        )
        print(f"\n[Cluster {c}] {cluster_names[c]}:")
        for _, row in samples.iterrows():
            print(f"   * {row['artists']} - {row['track_name']} (Gênero: {row['track_genre']}, Pop: {row['popularity']})")

    # Salva resumo em CSV
    summary.to_csv('cluster_summary.csv')
    print("\nResumo numérico salvo em 'cluster_summary.csv'.")

    return df_labeled


def plot_cluster_visualizations(df_labeled: pd.DataFrame, best_k: int):
    """Gera gráficos de dispersão das variáveis e perfil médio dos clusters."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("tab10", best_k)

    # Amostra para visualização sem poluir o scatter plot
    sample_df = df_labeled.sample(n=min(5000, len(df_labeled)), random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Danceability vs Valence
    sns.scatterplot(
        data=sample_df,
        x='danceability',
        y='valence',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        ax=axes[0]
    )
    axes[0].set_title("Clusters: Dançabilidade vs Valência (Humor)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Danceability (0 = menos, 1 = mais)")
    axes[0].set_ylabel("Valence (0 = triste, 1 = feliz)")

    # Gráfico 2: Energy vs Tempo
    sns.scatterplot(
        data=sample_df,
        x='tempo',
        y='energy',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        ax=axes[1]
    )
    axes[1].set_title("Clusters: Tempo (BPM) vs Energia", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tempo (BPM)")
    axes[1].set_ylabel("Energy (0 = calmo, 1 = enérgico)")

    output_path = OUTPUT_DIR / 'clusters_scatter.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de dispersão salvo em: {output_path}")

    # Gráfico 3: Perfil médio de cada cluster em barras
    summary_norm = df_labeled.groupby('cluster')[FEATURES].mean()
    # Min-max scaling apenas para visualização comparativa conjunta das 4 métricas
    summary_scaled = (summary_norm - summary_norm.min()) / (summary_norm.max() - summary_norm.min() + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 5))
    summary_norm.T.plot(kind='bar', ax=ax, colormap='tab10', width=0.7)
    ax.set_title("Comparação das Médias Originais dos Atributos por Cluster", fontsize=12, fontweight='bold')
    ax.set_ylabel("Valor Médio")
    ax.set_xlabel("Atributos Musicais")
    plt.xticks(rotation=0)
    plt.legend(title='Cluster', labels=[f"Cluster {i}" for i in range(best_k)])
    
    output_path_bar = OUTPUT_DIR / 'clusters_feature_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path_bar, dpi=300)
    plt.close()
    print(f"Gráfico de comparação de atributos salvo em: {output_path_bar}")


def main():
    # 1. Carregar e Pré-processar
    df, X, X_scaled, scaler = load_and_preprocess_data('dataset.csv')

    # 2. Avaliar faixa de k usando Coeficiente de Silhueta
    best_k, k_values, silhouette_scores, inertias = evaluate_k_range(X_scaled, k_min=2, k_max=10)

    # 3. Gerar curvas de métricas (Silhueta + Cotovelo)
    plot_metrics(k_values, silhouette_scores, inertias, best_k)

    # 4. Treinar modelo final com o k ótimo
    kmeans, cluster_labels = train_final_model(X_scaled, best_k)

    # 5. Interpretar perfis dos clusters
    df_labeled = interpret_and_profile_clusters(df, cluster_labels, best_k)

    # 6. Gerar gráficos de visualização
    plot_cluster_visualizations(df_labeled, best_k)

    print("\n" + "=" * 60)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == '__main__':
    main()
