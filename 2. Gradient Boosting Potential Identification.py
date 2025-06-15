#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df_2023 = pd.read_excel("2023_DATOS_CLUSTERED.xlsx")
df_2024 = pd.read_excel("2024_DATOS_CLEAN.xlsx")

# Prepare Demand variables
df_2023['TOTAL_VENTAS_2023'] = df_2023.filter(regex='FACTURACION').apply(pd.to_numeric, errors='coerce').sum(axis=1)
df_2024['TOTAL_VENTAS_2024'] = df_2024.filter(regex='VENTAS').apply(pd.to_numeric, errors='coerce').sum(axis=1)

df = pd.merge(
    df_2024,
    df_2023[['COD', 'TOTAL_VENTAS_2023', 'CLUSTER_NEW_VARS', 'CLUSTER_RELATIVE_CATEGORIES']],
    on='COD', how='inner'
)
## CLUSTER_NEW_VARS: clustering resulting from Engagement-Based clustering 
## CLUSTER_RELATIVE_CATEGORIES: clustering resulting from Product Mix clustering 


df['LOG_VENTAS_2023'] = np.log1p(df['TOTAL_VENTAS_2023'])
df['LOG_VENTAS_2024'] = np.log1p(df['TOTAL_VENTAS_2024'])

# Feature and target definition
X = pd.concat([
    df[['LOG_VENTAS_2023']],
    pd.get_dummies(df[['TIPOLOGIA', 'TAMAÑO', 'GRIFO', 'ZONA', 'CLUSTER_NEW_VARS', 'CLUSTER_RELATIVE_CATEGORIES']], drop_first=True)
], axis=1).fillna(0)
y = df['LOG_VENTAS_2024']

#  Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)



#  Quantile Model Function 
def train_quantile_model(alpha, X_train, y_train):
    model = GradientBoostingRegressor(
        loss='quantile', alpha=alpha, n_estimators=200,
        learning_rate=0.01, max_depth=4, subsample=0.9, random_state=42)
    model.fit(X_train, y_train)
    return model



#  Train Quantile Models 
model_p5 = train_quantile_model(0.05, X_train, y_train)
model_p50 = train_quantile_model(0.5, X_train, y_train)
model_p95 = train_quantile_model(0.95, X_train, y_train)

#  Predictions and Interval Construction 
df['PRED_P05'] = np.expm1(model_p5.predict(X))
df['PRED_P50'] = np.expm1(model_p50.predict(X))
df['PRED_P95'] = np.expm1(model_p95.predict(X))
df['PRED_INTERVAL_WIDTH'] = df['PRED_P95'] - df['PRED_P05']
df['PREDICTED_2024_DEMAND'] = df['PRED_P50']

#  Model Evaluation 
y_test_pred_p5 = np.expm1(model_p5.predict(X_test))
y_test_pred_p50 = np.expm1(model_p50.predict(X_test))
y_test_pred_p95 = np.expm1(model_p95.predict(X_test))
y_true_test = np.expm1(y_test)

coverage = np.mean((y_true_test >= y_test_pred_p5) & (y_true_test <= y_test_pred_p95))
rmse = np.sqrt(mean_squared_error(y_true_test, y_test_pred_p50))
mae = mean_absolute_error(y_true_test, y_test_pred_p50)

print(f"Interval Coverage (P5–P95): {coverage:.2%}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


#########################
#### Further Analysis
#########################

#  Residuals Analysis 
df['RESIDUALS_LOG'] = df['LOG_VENTAS_2024'] - np.log1p(df['PREDICTED_2024_DEMAND'])

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['PREDICTED_2024_DEMAND'], y=df['RESIDUALS_LOG'])
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Demand 2024")
plt.ylabel("Log Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['RESIDUALS_LOG'], bins=30, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Log Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()



#  Outlier  Flags 
df['RESIDUAL_OUTLIER'] = ~df['RESIDUALS_LOG'].between(
    df['RESIDUALS_LOG'].quantile(0.01), df['RESIDUALS_LOG'].quantile(0.99))
df['YOY_CHANGE'] = (df['TOTAL_VENTAS_2024'] - df['TOTAL_VENTAS_2023']) / df['TOTAL_VENTAS_2023']
df['VOLATILE_SALES'] = df['YOY_CHANGE'].abs() > 1.5

df['CLUSTER_MEAN'] = df.groupby('CLUSTER_NEW_VARS')['TOTAL_VENTAS_2024'].transform('mean')
df['CLUSTER_STD'] = df.groupby('CLUSTER_NEW_VARS')['TOTAL_VENTAS_2024'].transform('std')
df['Z_SCORE_CLUSTER'] = (df['TOTAL_VENTAS_2024'] - df['CLUSTER_MEAN']) / df['CLUSTER_STD']
df['ATYPICAL_IN_CLUSTER'] = df['Z_SCORE_CLUSTER'].abs() > 2

df['IRREGULAR_BEHAVIOR'] = (
    df['RESIDUAL_OUTLIER'] | df['VOLATILE_SALES'] | df['ATYPICAL_IN_CLUSTER']
)



#########################
#### Potential Classification
#########################

# Growth Potential Classification
conditions = [
    df['TOTAL_VENTAS_2024'] > df['PRED_P50'] * 1.1,
    df['TOTAL_VENTAS_2024'] < df['PRED_P50'] * 0.9
]
choices = ['Growth Potential', 'No Growth Potential']
df['GROWTH_POTENTIAL_TRIMMED'] = np.select(conditions, choices, default='Normal')



#  Exclude Extremes Based on Residuals 
p05 = df['RESIDUALS_LOG'].quantile(0.05)
p95 = df['RESIDUALS_LOG'].quantile(0.95)
df['EXCLUDED_FROM_TRIMMED'] = (df['RESIDUALS_LOG'] < p05) | (df['RESIDUALS_LOG'] > p95)

#  Quartile and Decile Creation 
df['QUARTILE'] = pd.qcut(df['TOTAL_VENTAS_2024'], 4, labels=[1, 2, 3, 4])
df['DECILE'] = pd.qcut(df['TOTAL_VENTAS_2024'], 10, labels=False) + 1

#  Error Metrics by Group 
def compute_error_metrics(group):
    actual = group['TOTAL_VENTAS_2024'].replace(0, np.nan)
    predicted = group['PREDICTED_2024_DEMAND']
    valid = actual.notna() & predicted.notna()
    return pd.Series({
        'Count': valid.sum(),
        'RMSE': np.sqrt(mean_squared_error(actual[valid], predicted[valid])),
        'MAE': mean_absolute_error(actual[valid], predicted[valid]),
        'Bias': (predicted[valid] - actual[valid]).mean(),
        'MAPE (%)': np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid])) * 100
    })

quartile_perf_full = df.groupby('QUARTILE').apply(compute_error_metrics).reset_index()
quartile_perf_full['Version'] = 'Full'

#  Trimmed Performance (exclude 5th–95th Residuals) 
low, high = df['RESIDUALS_LOG'].quantile([0.05, 0.95])
df_trimmed = df[(df['RESIDUALS_LOG'] >= low) & (df['RESIDUALS_LOG'] <= high)]
quartile_perf_trimmed = df_trimmed.groupby('QUARTILE').apply(compute_error_metrics).reset_index()
quartile_perf_trimmed['Version'] = 'Trimmed'

quartile_comparison = pd.concat([quartile_perf_full, quartile_perf_trimmed], ignore_index=True)
print("\nError Metrics by Quartile (Full vs Trimmed):\n", quartile_comparison)



#########################
#### Output
#########################

# Export final results
if 'ESTABLECIMIENTO' not in df.columns:
    df['ESTABLECIMIENTO'] = 'Unknown'

output_df = df[['COD', 'ESTABLECIMIENTO', 'TOTAL_VENTAS_2024', 'PREDICTED_2024_DEMAND',
                'GROWTH_POTENTIAL_TRIMMED', 'EXCLUDED_FROM_TRIMMED']].copy()
output_df['Demand Gap'] = output_df['TOTAL_VENTAS_2024'] - output_df['PREDICTED_2024_DEMAND']

output_df.rename(columns={
    'COD': 'Business Code',
    'ESTABLECIMIENTO': 'Business Name',
    'TOTAL_VENTAS_2024': 'Actual Demand 2024',
    'PREDICTED_2024_DEMAND': 'Predicted Demand 2024',
    'GROWTH_POTENTIAL_TRIMMED': 'Potential Classification',
    'EXCLUDED_FROM_TRIMMED': 'Flagged as Outlier'
}, inplace=True)

output_df = output_df.sort_values(by='Demand Gap')
output_df.to_excel("classification_results_ranked.xlsx", index=False)
print("File exported: classification_results_ranked.xlsx")
