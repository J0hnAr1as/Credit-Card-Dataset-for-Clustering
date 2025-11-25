import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

def train():
    print("--- Iniciando Entrenamiento ---")
    
    # 1. Cargar Dataset
    if not os.path.exists("CC_GENERAL.csv"):
        print("❌ ERROR: No se encuentra el archivo CC_GENERAL.csv en la carpeta.")
        return

    df = pd.read_csv("CC_GENERAL.csv")
    
    # 2. Limpieza básica
    df.drop('CUST_ID', axis=1, inplace=True)
    df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace=True)
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean(), inplace=True)

    # 3. Escalado
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # 4. K-Means (4 Clusters)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans.fit(df_scaled)
    labels = kmeans.labels_

    # 5. PCA para visualización (Reducción a 2D)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_scaled)
    
    # Guardar datos base para el gráfico de fondo
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels.astype(str)
    pca_df.to_csv('pca_labeled_data.csv', index=False)

    # 6. Guardar Modelos
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca_model.pkl')

    print("✅ ¡Éxito! Modelos guardados: kmeans_model.pkl, scaler.pkl, pca_model.pkl")

if __name__ == "__main__":
    train()