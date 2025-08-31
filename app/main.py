from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

bundle = joblib.load("model/model.pkl")
model = bundle["model"]
FEATURES = bundle["feature_names"]

PRED_COUNTER = Counter("predictions_total", "Nombre de prédictions")
PRED_LATENCY = Summary("prediction_seconds", "Temps de prédiction (s)")

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI(title="Housing API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features")
def features():
    return {"features": FEATURES}

@PRED_LATENCY.time()
@app.post("/predict")
def predict(item: HousingInput):
    PRED_COUNTER.inc()
    X = [[getattr(item, f) for f in FEATURES]]
    y_hat = model.predict(X)[0]
    return {"prediction": float(y_hat)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
