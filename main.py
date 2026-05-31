from fastapi import FastAPI
from app.schemas import PredictionInput, PredictionOutput
from app.model_loader import load_model
import numpy as np

app = FastAPI(title="Linear Regression API")

model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/")
def root():
    return {"message": "Linear Regression API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"prediction": float(prediction)}