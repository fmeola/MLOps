import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.inspection import permutation_importance
from sklearn.datasets import fetch_california_housing
import shap
from scipy.stats import ks_2samp, wasserstein_distance, energy_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

# Configuración visual
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
colors = sns.color_palette("mako", 10)

# Establecer semilla para reproducibilidad
np.random.seed(42)

print("="*50)
print("MONITOREO DE RENDIMIENTO DEL MODELO CON ANÁLISIS SHAP")
print("="*50)

# Crear un directorio para las gráficas
import os
if not os.path.exists('model_monitoring'):
    os.makedirs('model_monitoring')

# Cargar el dataset
print("\nCargando datos...")
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['target'] = housing.target

# Dividir en train/test
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Simular datos con drift para comparación
print("Preparando datos de referencia y actuales...")
current_data = test_data.copy()
# Introducir drift artificial en algunas características
current_data['MedInc'] = current_data['MedInc'] * 1.2
current_data['AveRooms'] = current_data['AveRooms'] + 0.5

# Preparar conjuntos
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']
X_current = current_data.drop('target', axis=1)
y_current = current_data['target']

# Entrenar modelo
print("Entrenando modelo...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generar predicciones
y_pred_test = model.predict(X_test)
y_pred_current = model.predict(X_current)

# Crear DataFrame de resultados
results_test = pd.DataFrame({
    'actual': y_test, 
    'predicted': y_pred_test, 
    'residual': y_test - y_pred_test,
    'abs_error': np.abs(y_test - y_pred_test),
    'pct_error': np.abs((y_test - y_pred_test) / y_test) * 100
})

results_current = pd.DataFrame({
    'actual': y_current,
    'predicted': y_pred_current,
    'residual': y_current - y_pred_current,
    'abs_error': np.abs(y_current - y_pred_current),
    'pct_error': np.abs((y_current - y_pred_current) / y_current) * 100
})

# 1. EVALUACIÓN DEL RENDIMIENTO DEL MODELO
print("\n" + "-"*50)
print("1. MÉTRICAS DE RENDIMIENTO DEL MODELO")
print("-"*50)

# Calcular métricas básicas
metrics = {
    'MSE': [mean_squared_error(y_test, y_pred_test), mean_squared_error(y_current, y_pred_current)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_test)), np.sqrt(mean_squared_error(y_current, y_pred_current))],
    'MAE': [mean_absolute_error(y_test, y_pred_test), mean_absolute_error(y_current, y_pred_current)],
    'MAPE (%)': [mean_absolute_percentage_error(y_test, y_pred_test) * 100, mean_absolute_percentage_error(y_current, y_pred_current) * 100],
    'R²': [r2_score(y_test, y_pred_test), r2_score(y_current, y_pred_current)],
    'Var. Explicada': [explained_variance_score(y_test, y_pred_test), explained_variance_score(y_current, y_pred_current)]
}

# Crear tabla comparativa de métricas
metrics_df = pd.DataFrame(metrics, index=['Datos de Referencia', 'Datos Actuales'])
print(metrics_df.round(4))

# Calcular diferencias
diff_df = pd.DataFrame({
    'Métrica': metrics_df.columns,
    'Referencia': metrics_df.iloc[0].values,
    'Actual': metrics_df.iloc[1].values,
    'Diferencia': metrics_df.iloc[1].values - metrics_df.iloc[0].values,
    'Cambio (%)': (metrics_df.iloc[1].values - metrics_df.iloc[0].values) / metrics_df.iloc[0].values * 100
})
print("\nCambios en las métricas:")
print(diff_df.round(4))

# 2. ANÁLISIS DE DRIFT AVANZADO EN VARIABLES
print("\n" + "-"*50)
print("2. ANÁLISIS AVANZADO DE DRIFT EN VARIABLES")
print("-"*50)

# Calcular estadísticas para cada columna
drift_analysis = []

for col in X_test.columns:
    # Estadísticas básicas
    ref_mean = X_test[col].mean()
    ref_std = X_test[col].std()
    curr_mean = X_current[col].mean()
    curr_std = X_current[col].std()
    
    # Calcular métricas de cambio
    mean_change_pct = (curr_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else np.inf
    std_change_pct = (curr_std - ref_std) / ref_std * 100 if ref_std != 0 else np.inf
    
    # Test de Kolmogorov-Smirnov (medida estadística de drift)
    ks_stat, ks_pval = ks_2samp(X_test[col], X_current[col])
    
    # Distancia de Wasserstein (Earth Mover's Distance)
    w_distance = wasserstein_distance(X_test[col], X_current[col])
    # Normalizar por el rango para hacerla comparable entre variables
    w_distance_norm = w_distance / (X_test[col].max() - X_test[col].min())
    
    # Distancia Energy (más sensible que KS para distribuciones multidimensionales)
    e_distance = energy_distance(X_test[col], X_current[col])
    # Normalizar por el rango
    e_distance_norm = e_distance / (X_test[col].max() - X_test[col].min())
    
    # Calcular distancia de Jensen-Shannon
    min_val = min(X_test[col].min(), X_current[col].min())
    max_val = max(X_test[col].max(), X_current[col].max())
    x_range = np.linspace(min_val, max_val, 1000)
    
    # KDE para datos de referencia y actuales
    ref_kde = gaussian_kde(X_test[col])
    ref_pdf = ref_kde(x_range)
    ref_pdf = ref_pdf / np.sum(ref_pdf)
    
    curr_kde = gaussian_kde(X_current[col])
    curr_pdf = curr_kde(x_range)
    curr_pdf = curr_pdf / np.sum(curr_pdf)
    
    js_distance = jensenshannon(ref_pdf, curr_pdf)
    
    # Determinar si hay drift significativo basado en múltiples criterios
    # 1. KS p-value < 0.05 significa distribuciones estadísticamente diferentes
    # 2. JS distance > 0.1 suele indicar drift significativo
    # 3. Wasserstein norm > 0.1 indica cambio sustancial
    drift_score = (
        (1 if ks_pval < 0.05 else 0) + 
        (1 if js_distance > 0.1 else 0) + 
        (1 if w_distance_norm > 0.1 else 0)
    )
    
    # Clasificación del drift
    if drift_score == 0:
        drift_level = "NO"
    elif drift_score == 1:
        drift_level = "LEVE"
    elif drift_score == 2:
        drift_level = "MODERADO"
    else:
        drift_level = "SEVERO"
    
    drift_analysis.append({
        'Feature': col,
        'Ref_Mean': ref_mean,
        'Curr_Mean': curr_mean,
        'Mean_Change(%)': mean_change_pct,
        'Ref_Std': ref_std,
        'Curr_Std': curr_std,
        'Std_Change(%)': std_change_pct,
        'KS_Stat': ks_stat,
        'KS_PValue': ks_pval,
        'Wasserstein': w_distance_norm,
        'Energy': e_distance_norm,
        'JS_Distance': js_distance,
        'Drift_Score': drift_score,
        'Drift_Level': drift_level
    })

# Crear DataFrame con resultados de drift
drift_df = pd.DataFrame(drift_analysis)
drift_df = drift_df.sort_values('Drift_Score', ascending=False)

print("\nAnálisis de drift en variables (ordenado por severidad del drift):")
print(drift_df[['Feature', 'Mean_Change(%)', 'KS_PValue', 'Wasserstein', 'JS_Distance', 'Drift_Level']].round(4))

# Visualizar cambios en las distribuciones para las variables con mayor drift
top_drift_features = drift_df[drift_df['Drift_Level'] != "NO"]['Feature'].tolist()[:3]  # Hasta 3 con drift

for feature in top_drift_features:
    plt.figure(figsize=(12, 6))
    
    # Histogramas con KDE
    sns.histplot(X_test[feature], color=colors[0], label='Referencia', kde=True, alpha=0.6)
    sns.histplot(X_current[feature], color=colors[1], label='Actual', kde=True, alpha=0.6)
    
    # Añadir estadísticas al gráfico
    feature_stats = drift_df[drift_df['Feature'] == feature].iloc[0]
    plt.title(f'Drift en {feature} (Nivel: {feature_stats["Drift_Level"]})')
    plt.text(0.02, 0.95, 
             f'KS p-valor: {feature_stats["KS_PValue"]:.4f}\n'
             f'Jensen-Shannon: {feature_stats["JS_Distance"]:.4f}\n'
             f'Wasserstein: {feature_stats["Wasserstein"]:.4f}',
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'model_monitoring/drift_{feature}.png')
    plt.close()

# Visualizar todas las métricas de drift para cada variable
plt.figure(figsize=(14, 8))
metrics_to_plot = ['KS_Stat', 'Wasserstein', 'JS_Distance']
drift_plot_data = drift_df.melt(
    id_vars=['Feature', 'Drift_Level'], 
    value_vars=metrics_to_plot,
    var_name='Métrica', 
    value_name='Valor'
)
sns.barplot(data=drift_plot_data, x='Feature', y='Valor', hue='Métrica')
plt.xticks(rotation=45, ha='right')
plt.title('Comparación de métricas de drift por variable')
plt.tight_layout()
plt.savefig('model_monitoring/drift_metrics_comparison.png')
plt.close()

# 3. ANÁLISIS GLOBAL DE VALORES SHAP
print("\n" + "-"*50)
print("3. ANÁLISIS GLOBAL DE VALORES SHAP")
print("-"*50)

# Crear el explicador SHAP
print("Calculando valores SHAP...")
explainer = shap.TreeExplainer(model)

# Calcular valores SHAP para el conjunto de referencia
shap_values_test = explainer(X_test)

# 3.1 Gráfico de resumen global (SHAP Summary Plot)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_test.values, X_test, plot_type="bar", show=False)
plt.title('Importancia Global de Variables (SHAP)')
plt.tight_layout()
plt.savefig('model_monitoring/shap_global_importance.png')
plt.close()

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_test.values, X_test, show=False)
plt.title('Impacto y Distribución de Variables (SHAP)')
plt.tight_layout()
plt.savefig('model_monitoring/shap_global_summary.png')
plt.close()

# 3.2 Gráficos de dependencia para las 3 variables más importantes
top_features = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': np.abs(shap_values_test.values).mean(0)
}).sort_values('Importance', ascending=False)['Feature'].head(3).tolist()

for feature in top_features:
    plt.figure(figsize=(12, 6))
    feature_idx = list(X_test.columns).index(feature)
    shap.dependence_plot(feature_idx, shap_values_test.values, X_test, show=False)
    plt.title(f'Gráfico de Dependencia SHAP para {feature}')
    plt.tight_layout()
    plt.savefig(f'model_monitoring/shap_dependence_{feature}.png')
    plt.close()

# 3.3 Comparar valores SHAP entre datos de referencia y actuales
print("\nComparando valores SHAP entre datos de referencia y actuales...")
shap_values_current = explainer(X_current)

# Calcular la media absoluta de los valores SHAP para cada característica
shap_comparison = pd.DataFrame({
    'Feature': X_test.columns,
    'SHAP_Ref': np.abs(shap_values_test.values).mean(0),
    'SHAP_Current': np.abs(shap_values_current.values).mean(0)
})

shap_comparison['SHAP_Change'] = shap_comparison['SHAP_Current'] - shap_comparison['SHAP_Ref']
shap_comparison['SHAP_Change_Pct'] = (shap_comparison['SHAP_Change'] / shap_comparison['SHAP_Ref']) * 100
shap_comparison = shap_comparison.sort_values('SHAP_Change_Pct', ascending=False)

print("\nCambios en la importancia de variables (SHAP):")
print(shap_comparison.round(4))

# Visualizar cambios en importancia SHAP
plt.figure(figsize=(12, 8))
shap_comp_plot = pd.melt(
    shap_comparison, 
    id_vars=['Feature'], 
    value_vars=['SHAP_Ref', 'SHAP_Current'], 
    var_name='Dataset', 
    value_name='SHAP Value'
)
sns.barplot(data=shap_comp_plot, x='Feature', y='SHAP Value', hue='Dataset')
plt.xticks(rotation=45, ha='right')
plt.title('Comparación de valores SHAP entre datos de referencia y actuales')
plt.tight_layout()
plt.savefig('model_monitoring/shap_importance_comparison.png')
plt.close()

# 4. ANÁLISIS DE UN CASO PARTICULAR CON SHAP
print("\n" + "-"*50)
print("4. ANÁLISIS DE UN CASO PARTICULAR CON SHAP")
print("-"*50)

# Seleccionar un caso interesante (por ejemplo, uno con error alto)
high_error_idx = results_current['abs_error'].idxmax()
print(f"\nAnalizando el caso #{high_error_idx} (con error máximo):")

# Obtener los datos del caso
instance = X_current.iloc[[high_error_idx]]
instance_pred = y_pred_current[high_error_idx]
instance_actual = y_current.iloc[high_error_idx]
instance_error = instance_actual - instance_pred

# Mostrar detalles del caso
print(f"\nDetalles del caso #{high_error_idx}:")
for col, val in instance.iloc[0].items():
    print(f"- {col}: {val:.4f}")
print(f"- Valor real: {instance_actual:.4f}")
print(f"- Predicción: {instance_pred:.4f}")
print(f"- Error: {instance_error:.4f} ({(instance_error/instance_actual)*100:.2f}%)")

# Calcular valores SHAP para este caso específico
instance_shap = explainer(instance)

# 4.1 Gráfico de fuerza (Force Plot)
plt.figure(figsize=(20, 3))
shap.plots.force(instance_shap[0], matplotlib=True, show=False)
plt.title(f'Análisis de factores para el caso #{high_error_idx}')
plt.tight_layout()
plt.savefig('model_monitoring/shap_force_plot_instance.png')
plt.close()

# 4.2 Gráfico de cascada (Waterfall Plot)
plt.figure(figsize=(12, 8))
shap.plots.waterfall(instance_shap[0], max_display=len(X_test.columns), show=False)
plt.title(f'Contribución de cada variable para el caso #{high_error_idx}')
plt.tight_layout()
plt.savefig('model_monitoring/shap_waterfall_instance.png')
plt.close()

# 4.3 Información detallada sobre las contribuciones de cada variable
instance_contribution = pd.DataFrame({
    'Feature': X_test.columns,
    'Value': instance.values[0],
    'SHAP_Value': instance_shap.values[0],
    'Abs_SHAP': np.abs(instance_shap.values[0])
}).sort_values('Abs_SHAP', ascending=False)

print("\nContribución de variables para este caso específico:")
for i, row in instance_contribution.iterrows():
    direction = "aumentó" if row['SHAP_Value'] > 0 else "disminuyó"
    print(f"- {row['Feature']} = {row['Value']:.4f} {direction} la predicción en {abs(row['SHAP_Value']):.4f}")

# 5. INTEGRACIÓN DE DRIFT Y SHAP
print("\n" + "-"*50)
print("5. INTEGRACIÓN DE ANÁLISIS DE DRIFT Y SHAP")
print("-"*50)

# Combinar resultados de drift y cambios en SHAP
integrated_analysis = pd.merge(
    drift_df[['Feature', 'Drift_Level', 'KS_PValue', 'JS_Distance']], 
    shap_comparison[['Feature', 'SHAP_Change_Pct']], 
    on='Feature'
)

# Ordenar por importancia del cambio
integrated_analysis = integrated_analysis.sort_values('SHAP_Change_Pct', ascending=False)

print("\nRelación entre drift y cambios en importancia de variables:")
print(integrated_analysis.round(4))

# Visualizar la relación entre drift y cambios en SHAP
plt.figure(figsize=(12, 8))
for level in ['SEVERO', 'MODERADO', 'LEVE', 'NO']:
    level_data = integrated_analysis[integrated_analysis['Drift_Level'] == level]
    if len(level_data) > 0:
        plt.scatter(
            level_data['JS_Distance'], 
            level_data['SHAP_Change_Pct'],
            label=f'Drift {level}',
            s=100,
            alpha=0.7
        )
        
        # Añadir etiquetas a los puntos
        for _, row in level_data.iterrows():
            plt.annotate(
                row['Feature'], 
                (row['JS_Distance'], row['SHAP_Change_Pct']),
                xytext=(5, 5),
                textcoords='offset points'
            )

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Umbral de drift (JS)')
plt.xlabel('Distancia Jensen-Shannon (Drift)')
plt.ylabel('Cambio en importancia SHAP (%)')
plt.title('Relación entre Drift y Cambios en Importancia de Variables')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_monitoring/drift_shap_relationship.png')
plt.close()

# 6. INFORME FINAL
print("\n" + "="*50)
print("6. RESUMEN DEL MONITOREO")
print("="*50)

# Ver qué variables tienen drift significativo
drift_vars = drift_df[drift_df['Drift_Level'] != 'NO']['Feature'].tolist()

print(f"\nFecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total de instancias analizadas: {len(X_test)}")

if len(drift_vars) > 0:
    print(f"\nVariables con drift detectado ({len(drift_vars)}):")
    for feature in drift_vars:
        level = drift_df[drift_df['Feature'] == feature]['Drift_Level'].values[0]
        print(f"- {feature}: Drift {level}")
else:
    print("\nNo se detectó drift significativo en ninguna variable.")

# Resumen de degradación
r2_change = metrics['R²'][1] - metrics['R²'][0]
rmse_change = metrics['RMSE'][1] - metrics['RMSE'][0]

print("\nCambios en el rendimiento del modelo:")
if r2_change < 0:
    print(f"- R² disminuyó en {abs(r2_change):.4f} ({(r2_change/metrics['R²'][0])*100:.2f}%)")
else:
    print(f"- R² aumentó en {r2_change:.4f} ({(r2_change/metrics['R²'][0])*100:.2f}%)")

if rmse_change > 0:
    print(f"- RMSE empeoró en {rmse_change:.4f} ({(rmse_change/metrics['RMSE'][0])*100:.2f}%)")
else:
    print(f"- RMSE mejoró en {abs(rmse_change):.4f} ({(rmse_change/metrics['RMSE'][0])*100:.2f}%)")

# Variables con mayor cambio en importancia SHAP
top_shap_changes = shap_comparison.head(3)['Feature'].tolist()
print("\nVariables con mayor cambio en importancia:")
for feature in top_shap_changes:
    change = shap_comparison[shap_comparison['Feature'] == feature]['SHAP_Change_Pct'].values[0]
    print(f"- {feature}: {change:.2f}% de cambio en importancia SHAP")

# Análisis de caso particular
print(f"\nAnálisis de caso particular (#{high_error_idx}):")
top_impact = instance_contribution.head(3)['Feature'].tolist()
print(f"Las variables con mayor impacto en la predicción fueron: {', '.join(top_impact)}")

# Recomendaciones
print("\nRECOMENDACIONES:")
severe_drift = any(drift_df['Drift_Level'] == 'SEVERO')
moderate_drift = any(drift_df['Drift_Level'] == 'MODERADO')

if severe_drift and r2_change < -0.05:
    print("- El modelo muestra signos de degradación significativa con drift severo.")
    print("- Se recomienda REENTRENAR el modelo con datos más recientes.")
    print(f"- Variables críticas a monitorear: {', '.join(drift_df[drift_df['Drift_Level'] == 'SEVERO']['Feature'].tolist())}")
elif moderate_drift or severe_drift:
    print("- Se detectó drift en algunas variables, con impacto moderado en rendimiento.")
    print("- Se recomienda realizar VALIDACIÓN ADICIONAL y considerar reentrenamiento.")
    print(f"- Variables a monitorear: {', '.join(drift_vars[:3])}")
elif len(drift_vars) > 0:
    print("- Se detectó drift leve en algunas variables, con impacto limitado en rendimiento.")
    print("- Monitorear el modelo en los próximos días.")
else:
    print("- El modelo mantiene un rendimiento estable sin drift significativo.")
    print("- Continuar con el monitoreo rutinario.")

print("\nGráficos generados en el directorio 'model_monitoring/'")