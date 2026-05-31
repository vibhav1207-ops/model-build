import joblib

MODEL_PATH = "saved_model/model.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model