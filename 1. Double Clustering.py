#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load data
path_2023 = "2023 DATOS.xlsx"
path_2024 = "2024_DATOS_CLEAN.xlsx"
df_2023 = pd.read_excel(path_2023, dtype=str)
df_2024 = pd.read_excel(path_2024, dtype=str)

# Keep only clients present in both datasets
common_clients = set(df_2023["COD"]).intersection(df_2024["COD"])
df_2023_common = df_2023[df_2023["COD"].isin(common_clients)].copy()
df_2024_common = df_2024[df_2024["COD"].isin(common_clients)].copy()

#First Clustering: Engagement-based
df1 = df_2023_common.copy()

# Select and convert relevant columns
order_cols = [col for col in df1.columns if "ORDERS" in col]
demand_cols = [col for col in df1.columns if "FACTURACION" in col]
df1[order_cols + demand_cols] = df1[order_cols + demand_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Create aggregated features
df1['NUM_PRODUCTS_ORDERED'] = (df1[order_cols] > 0).sum(axis=1)
df1['TOTAL_DEMAND'] = df1[demand_cols].sum(axis=1)
df1['N_ORDERS'] = df1[order_cols].max(axis=1)

# Scale features
X1 = df1[['NUM_PRODUCTS_ORDERED', 'TOTAL_DEMAND', 'N_ORDERS']]
X1_scaled = StandardScaler().fit_transform(X1)

# Elbow and silhouette method
inertia = []
silhouette = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X1_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X1_scaled, labels))

# Plot elbow method
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bo-', label='Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.legend()
plt.show()

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette, 'go-', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores')
plt.grid(True)
plt.legend()
plt.show()

# Final clustering
k_final = 2
kmeans_final = KMeans(n_clusters=k_final, random_state=42)
df1['CLUSTER_NEW_VARS'] = kmeans_final.fit_predict(X1_scaled)

# Evaluation output
final_silhouette = silhouette_score(X1_scaled, df1['CLUSTER_NEW_VARS'])
print(f"Final Silhouette Score: {final_silhouette:.4f}")
print(f"Cluster Distribution:\n{df1['CLUSTER_NEW_VARS'].value_counts()}")

############################################################################

# Second Clustering: Product Mix
df2 = df_2023_common.copy()

# Convert all relevant columns
for col in df2.columns:
    if any(metric in col for metric in ['FACTURACION', 'MARGIN', 'ORDERS']):
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)

# Define product category mapping
category_mapping = {
    'Beer': ['1/3 5E RET+NR', '1/5 RET 5E', 'TOSTADA + SIN CAJA', 'MAESTRA EN BOTELLA',
             'ALHAMBRA', 'CAJA INVEB+WAST+KONING', 'MIXTA 1/5+1/3', 'RADLER CAJA'],
    'Soft Drinks': ['AGUA CON GAS', 'AGUA CRISTAL', 'AGUA PET', 'BATIDOS/CACOLAT', 'CAJE DE LECHE',
                    'COCA COLA', 'NESTEA+AQUARIUS', 'RED BULL', 'CAJA MOSTO', 'CO2',
                    'KAS-GASEOSA', 'TONICA'],
    'Wine': ['BOX VINO', 'CHAMPAGNE BOTELLA', 'COSECHERO/DON FRUTOS', 'ESPUMOSOS', 'FRIZZANTES + 5.5',
             'TIERRA CYL CAJA', 'VERMUT BOTELLA', 'VINO DO TORO CAJA', 'VINOS D.O. CIGALES CAJAS',
             'VINOS DO RIBERA BOTELLA', 'VINOS DO RIBERA CAJA', 'VINOS DO RUEDA CAJAS'],
    'Spirits': ['BOTELLA ALCOHOLES'],
    'Food': ['COCINA/ALIMENTACION', 'DESAYUNO (CAFE,ZUMO ETC)', 'NO RETORNABLE (ALIMENTACION)', 'SIN GLUTEN'],
    'Other': ['PRODUCTOS LIMPIEZA']
}

# Aggregate metrics by category
for category in category_mapping:
    for metric in ['FACTURACION', 'ORDERS']:
        df2[f'{category}_{metric}'] = 0

for category, products in category_mapping.items():
    for product in products:
        for metric in ['FACTURACION', 'ORDERS']:
            col_name = f"{product} - {metric}"
            if col_name in df2.columns:
                df2[f'{category}_{metric}'] += df2[col_name]

# Keep only relevant columns
cols_to_keep = ['COD'] + [f"{cat}_{m}" for cat in category_mapping for m in ['FACTURACION', 'ORDERS']]
df2 = df2[cols_to_keep]

# Normalize by total per row
sales_cols = [col for col in df2.columns if col.endswith('_FACTURACION')]
orders_cols = [col for col in df2.columns if col.endswith('_ORDERS')]

sales_relative = df2[sales_cols].div(df2[sales_cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
orders_relative = df2[orders_cols].div(df2[orders_cols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# Combine and scale features
X2 = pd.concat([sales_relative, orders_relative], axis=1)
X2_scaled = StandardScaler().fit_transform(X2)

# Elbow and silhouette method
inertia = []
silhouette = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X2_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X2_scaled, labels))

# Plot elbow method
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bo-', label='Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method (Category Data)')
plt.grid(True)
plt.legend()
plt.show()

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette, 'go-', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores (Category Data)')
plt.grid(True)
plt.legend()
plt.show()

# Final clustering
k_final = 5
kmeans2 = KMeans(n_clusters=k_final, random_state=42)
df2['CLUSTER_RELATIVE_CATEGORIES'] = kmeans2.fit_predict(X2_scaled)

# Evaluation output
final_silhouette = silhouette_score(X2_scaled, df2['CLUSTER_RELATIVE_CATEGORIES'])
print(f"Final Silhouette Score (Category Data): {final_silhouette:.4f}")
print(f"Cluster Distribution:\n{df2['CLUSTER_RELATIVE_CATEGORIES'].value_counts()}")

############################################################################
# Merge both clustering results

merged = df1[['COD', 'CLUSTER_NEW_VARS']].merge(df2[['COD', 'CLUSTER_RELATIVE_CATEGORIES']], on='COD')
df_raw = df_2023.copy()
df_final = df_raw.merge(merged, on='COD')

# Save the dataset
output_path = "/Users/saragarciadefuentes/Documents/TESIS/DATOS/2023_DATOS_CLUSTERED.xlsx"
df_final.to_excel(output_path, index=False)
print(f"\nFile saved with cluster labels:\n{output_path}")
