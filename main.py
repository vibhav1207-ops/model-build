from fastapi import FastAPI, HTTPException
from .schema import PredictionInput
from .model_loader import load_model

app = FastAPI(title="Linear Regression API")

model = None

@app.on_event("startup")
def load():
    global model
    model = load_model()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    if len(data.features) != 10:
        raise HTTPException(
            status_code=400,
            detail="Input must contain exactly 10 features"
        )

    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}