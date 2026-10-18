"""
Módulo de Pré-processamento de Dados para Agrupamento K-Means (Spotify Tracks).

Executa:
1. Carregamento da base bruta (dataset.csv)
2. Remoção de faixas duplicadas por track_id
3. Tratamento de valores nulos e inconsistências (tempo <= 0)
4. Padronização (StandardScaler) das 4 features sonoras
5. Exportação do dataset limpo e escalonado (dataset_processed.csv) e do scaler (scaler.joblib)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ['danceability', 'energy', 'valence', 'tempo']
SCALED_FEATURES = [f'{col}_scaled' for col in FEATURES]


def preprocess_spotify_dataset(
    input_csv: str = 'dataset.csv',
    output_csv: str = 'dataset_processed.csv',
    scaler_output: str = 'scaler.joblib'
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Carrega, limpa e padroniza as features do dataset do Spotify.

    Retorna:
        df: DataFrame limpo contendo colunas originais e colunas padronizadas.
        X_scaled: matriz numpy (N, 4) com as 4 features padronizadas.
        scaler: objeto StandardScaler ajustado.
    """
    print("=" * 65)
    print("ETAPA DE PRÉ-PROCESSAMENTO DE DADOS")
    print("=" * 65)

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path.resolve()}")

    print(f"[1/4] Lendo base de dados bruta de: {input_csv} ...")
    df_raw = pd.read_csv(input_path)
    total_raw = len(df_raw)
    print(f"      Total de registros carregados: {total_raw:,}")

    print("[2/4] Removendo duplicatas pelo 'track_id'...")
    df = df_raw.drop_duplicates(subset='track_id').copy()
    duplicates_removed = total_raw - len(df)
    print(f"      Duplicatas removidas: {duplicates_removed:,} (Restam {len(df):,} faixas únicas)")

    print("[3/4] Tratando nulos e inconsistências de ritmo (tempo <= 0)...")
    initial_valid = len(df)
    df = df.dropna(subset=FEATURES)
    df = df[df['tempo'] > 0].copy()
    invalid_removed = initial_valid - len(df)
    print(f"      Registros inconsistentes removidos: {invalid_removed:,}")
    print(f"      Total final de faixas válidas: {len(df):,}")

    print("[4/4] Aplicando Padronização Z-Score (StandardScaler)...")
    scaler = StandardScaler()
    X = df[FEATURES].values
    X_scaled = scaler.fit_transform(X)

    # Anexa colunas escalonadas ao dataframe para facilitar consultas e análises
    for i, col in enumerate(SCALED_FEATURES):
        df[col] = X_scaled[:, i]

    # Salva arquivos de saída
    df.to_csv(output_csv, index=False)
    joblib.dump(scaler, scaler_output)
    print(f"\n-> Dataset processado salvo com sucesso em: {output_csv}")
    print(f"-> Objeto Scaler salvo em: {scaler_output}")

    # Exibe resumo estatístico das variáveis antes e depois
    print("\nEstatísticas das Features Selecionadas:")
    stats_orig = df[FEATURES].describe().T[['mean', 'std', 'min', 'max']].round(3)
    stats_orig.columns = ['Média (Orig)', 'Desv.Padrão', 'Mínimo', 'Máximo']
    print(stats_orig.to_string())

    print("\n" + "=" * 65)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 65)

    return df, X_scaled, scaler


if __name__ == '__main__':
    preprocess_spotify_dataset()
