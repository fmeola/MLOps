
import uvicorn
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Crear la aplicación FastAPI
app = FastAPI(title="Modelo ML API", description="API para servir predicciones del modelo de MLflow")

# Acá va la configuración de CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O limitá a ["http://localhost:8000"] si querés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo al iniciar la aplicación
# Reemplaza "models:/nombre_modelo/version" con tu ruta real
model = mlflow.pyfunc.load_model("models:/ridge_model_v1/8")

# Definir el esquema de la solicitud
class PredictionRequest(BaseModel):
    features: List[List[float]]

# Definir el esquema de la respuesta
class PredictionResponse(BaseModel):
    predictions: List

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convertir las características a un array numpy
        features = np.array(request.features)
        # Realizar la predicción
        predictions = model.predict(features).tolist()
        # Devolver las predicciones
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

