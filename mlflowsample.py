import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def entrenar_modelo_iris():
    # Configurar MLflow
    mlflow.set_experiment("clasificacion_iris_detallada")
    
    # Cargar datos
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Iniciar run de MLflow
    with mlflow.start_run():
        # Crear modelo de Regresión Logística
        modelo = LogisticRegression(max_iter=200, multi_class='ovr')
        modelo.fit(X_train_scaled, y_train)
        
        # Predecir
        predicciones = modelo.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, predicciones)
        
        # Matriz de Confusión
        cm = confusion_matrix(y_test, predicciones)
        
        # Visualizar Matriz de Confusión
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris.target_names, 
                    yticklabels=iris.target_names)
        plt.title('Matriz de Confusión - Clasificación Iris')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig('confusion_matrix.png')
        
        # Registrar métricas y parámetros en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("modelo", "Regresión Logística")
        mlflow.log_param("max_iter", 200)
        
        # Registrar la imagen de la matriz de confusión
        mlflow.log_artifact('confusion_matrix.png')
        
        # Guardar modelo
        mlflow.sklearn.log_model(modelo, "modelo_iris")
        
        # Imprimir resultados
        print("Accuracy:", accuracy)
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, predicciones, target_names=iris.target_names))

        import subprocess
import sys
import os
import subprocess

import webbrowser
import threading
import subprocess

def abrir_mlflow_ui():
    # Abre MLflow UI en un hilo separado
    def iniciar_mlflow():
        subprocess.run(["mlflow", "ui"])
    
    # Inicia MLflow en segundo plano
    thread = threading.Thread(target=iniciar_mlflow)
    thread.start()
    
    # Abre navegador después de un pequeño retraso
    import time
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == "__main__":
    entrenar_modelo_iris()
    
    # Lanzar UI de MLflow
    abrir_mlflow_ui()
