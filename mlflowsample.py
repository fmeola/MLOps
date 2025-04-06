#MLFlow sample
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
import webbrowser
import threading
import time
import subprocess


def entrenar_modelo_iris():
    mlflow.set_experiment("clasificacion_iris_detallada")

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        modelo = LogisticRegression(max_iter=200, multi_class='ovr')
        modelo.fit(X_train_scaled, y_train)

        predicciones = modelo.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predicciones)
        cm = confusion_matrix(y_test, predicciones)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names
        )
        plt.title('Matriz de Confusión - Clasificación Iris')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("modelo", "Regresión Logística")
        mlflow.log_param("max_iter", 200)
        mlflow.log_artifact('confusion_matrix.png')
        mlflow.sklearn.log_model(modelo, "modelo_iris")

        print("Accuracy:", accuracy)
        print("\nReporte de Clasificación:")
        print(
            classification_report(
                y_test, predicciones, target_names=iris.target_names
            )
        )


def abrir_mlflow_ui():
    def iniciar_mlflow():
        subprocess.run(["mlflow", "ui"])

    thread = threading.Thread(target=iniciar_mlflow)
    thread.start()
    time.sleep(2)
    webbrowser.open('http://localhost:5000')


if __name__ == "__main__":
    entrenar_modelo_iris()
    abrir_mlflow_ui()
