#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random

from scipy.stats import shapiro, wilcoxon
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Load data
data_2023 = pd.read_excel("2023_DATOS_CLUSTERED.xlsx")
data_2024 = pd.read_excel("2024_DATOS_CLEAN.xlsx")


# Clean and standardize column names  to enable the matching between datasets
def standardize_columns(df):
    df.columns = (
        df.columns
          .str.replace(" - ", " ", regex=False)
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
          .str.upper()
    )

standardize_columns(data_2023)
standardize_columns(data_2024)



# Map product order columns between years 
def extract_order_columns(df):
    return [col for col in df.columns if "ORDERS" in col]

orders_2023 = extract_order_columns(data_2023)
orders_2024 = extract_order_columns(data_2024)

def map_product_to_order(col_list):
    mapping = {}
    suffix = " ORDERS"
    for col in col_list:
        if col.endswith(suffix):
            product = col[:-len(suffix)].strip()
            mapping[product] = col
    return mapping

product_map_2023 = map_product_to_order(orders_2023)
product_map_2024 = map_product_to_order(orders_2024)
common_products = sorted(set(product_map_2023) & set(product_map_2024))



# Build yearly DataFrames 
def build_yearly_df(data, product_map, products, year):
    cols = ["COD"] + [product_map[p] for p in products]
    df = data[cols].copy()
    renamed = {"COD": "COD"}
    renamed.update({product_map[p]: f"{p}_{year}" for p in products})
    df.rename(columns=renamed, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

df_23 = build_yearly_df(data_2023, product_map_2023, common_products, year=2023)
df_24 = build_yearly_df(data_2024, product_map_2024, common_products, year=2024)



# Identify customers who bought new products in 2024 
merged = df_23.merge(df_24, on="COD", how="inner")

new_buy_mask = {
    prod: (merged[f"{prod}_2024"] > 0) & (merged[f"{prod}_2023"] == 0)
    for prod in common_products
}

new_customers = merged[pd.DataFrame(new_buy_mask).any(axis=1)].copy()
new_customers["NUM_NEW_PRODUCTS_2024"] = pd.DataFrame(new_buy_mask).sum(axis=1)
new_customers.to_excel("filtered_customers_new_2024.xlsx", index=False)

# ¡Prepare KNN training data 
cluster_col1 = "CLUSTER_NEW_VARS"
cluster_col2 = "CLUSTER_RELATIVE_CATEGORIES"
for col in (cluster_col1, cluster_col2):
    if col not in data_2023.columns:
        raise KeyError(f"Missing cluster column: {col}")

pivot = data_2023.set_index("COD")[[product_map_2023[p] for p in common_products]].fillna(0)
pivot.columns = common_products
sparse_matrix = csr_matrix(pivot.values)

knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(sparse_matrix)

# Recommendation functions
def recommend_random(customer_id, df, products, n=3):
    """Suggest random products the customer hasn't bought yet."""
    row = df[df["COD"] == customer_id]
    if row.empty:
        return []
    purchased = [row[f"{p} ORDERS"].iloc[0] > 0 for p in products]
    candidates = [p for p, bought in zip(products, purchased) if not bought]
    return random.sample(candidates, min(n, len(candidates)))

def recommend_by_cluster(customer_id, df, products, cluster_col, n=3):
    """Suggest top products commonly bought within the same cluster."""
    row = df[df["COD"] == customer_id]
    if row.empty:
        return []
    label = row[cluster_col].iloc[0]
    group = df[df[cluster_col] == label]
    totals = group[[f"{p} ORDERS" for p in products]].sum()
    ranked = totals.sort_values(ascending=False).index.str.replace(" ORDERS", "")
    already = [row[f"{p} ORDERS"].iloc[0] > 0 for p in products]
    return [p for p in ranked if p in products and p not in np.array(products)[already]][:n]

def recommend_combined(customer_id, df, products, col1, col2, n=3):
    """Suggest products based on combined cluster labels; fallback to single labels if needed."""
    row = df[df["COD"] == customer_id]
    if row.empty:
        return []
    l1, l2 = row[col1].iloc[0], row[col2].iloc[0]
    subset = df[(df[col1] == l1) & (df[col2] == l2)]
    if len(subset) < 3:
        subset = df[(df[col1] == l1) | (df[col2] == l2)]
    totals = subset[[f"{p} ORDERS" for p in products]].sum()
    ranked = totals.sort_values(ascending=False).index.str.replace(" ORDERS", "")
    already = [row[f"{p} ORDERS"].iloc[0] > 0 for p in products]
    return [p for p in ranked if p in products and p not in np.array(products)[already]][:n]

def recommend_knn(customer_id, pivot_df, model, sparse_mat, products, n=3):
    """Suggest products based on similar customers (KNN)."""
    if customer_id not in pivot_df.index:
        return []
    idx = pivot_df.index.get_loc(customer_id)
    distances, indices = model.kneighbors(sparse_mat[idx], n_neighbors=n + 1)
    neighbors = pivot_df.iloc[indices.flatten()[1:]]
    avg_purchase = neighbors.mean().sort_values(ascending=False)
    owned = pivot_df.loc[customer_id] > 0
    return avg_purchase[~owned].index[:n].tolist()



# Evaluate recommendation quality
methods = {
    "Random": lambda cid: recommend_random(cid, data_2023, common_products),
    "Cluster_New_Vars": lambda cid: recommend_by_cluster(cid, data_2023, common_products, cluster_col1),
    "Cluster_Rel_Cat": lambda cid: recommend_by_cluster(cid, data_2023, common_products, cluster_col2),
    "Combined_Clusters": lambda cid: recommend_combined(cid, data_2023, common_products, cluster_col1, cluster_col2),
    "KNN": lambda cid: recommend_knn(cid, pivot, knn_model, sparse_matrix, common_products),
}

precision_scores = {}
for name, rec_fn in methods.items():
    scores = []
    for cid in new_customers["COD"]:
        recs = rec_fn(cid)
        if not recs:
            continue
        actual = new_customers[new_customers["COD"] == cid].iloc[0]
        hits = sum((actual[f"{p}_2024"] > 0) and (actual[f"{p}_2023"] == 0) for p in recs)
        scores.append(hits / len(recs))
    precision_scores[name] = scores





# Summarize and test results 
summary = []
for name, scores in precision_scores.items():
    summary.append((name, np.mean(scores) if scores else 0.0))

results_df = pd.DataFrame(summary, columns=["Model", "Average Precision"])
print(results_df)




# Check statistical significance versus Random model
baseline = "Random"
comparisons = [m for m in methods if m != baseline]
differences = {
    m: np.array(precision_scores[baseline]) - np.array(precision_scores[m])
    for m in comparisons
}

print("\nShapiro-Wilk test for normality of differences:")
for m, diffs in differences.items():
    stat, pval = shapiro(diffs)
    print(f"{baseline} vs {m}: p = {pval:.4f}")

print("\nWilcoxon signed-rank test results:")
for m, diffs in differences.items():
    stat, pval = wilcoxon(diffs, zero_method="wilcox")
    print(f"{baseline} vs {m}: W = {stat:.4f}, p = {pval:.4f}")




plt.figure(figsize=(8, 5))
for m, diffs in differences.items():
    sorted_diffs = np.sort(diffs)
    cdf = np.linspace(0, 1, len(sorted_diffs))
    plt.plot(sorted_diffs, cdf, label=f"{baseline} − {m}")
plt.legend()
plt.title("CDF of Precision Differences (Random minus Other Models)")
plt.xlabel("Precision Difference")
plt.ylabel("CDF")
plt.tight_layout()
plt.show()

# Save evaluation output 
results_df.to_excel("model_evaluation_results.xlsx", index=False)
