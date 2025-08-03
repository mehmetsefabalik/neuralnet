from fastapi import FastAPI, HTTPException, Query
import numpy as np
from tensorflow import keras
import os

# Load the trained model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'xor_model.keras')
model = keras.models.load_model(MODEL_PATH)

app = FastAPI()

@app.get("/predict")
def predict(
    x1: float = Query(..., ge=0.0, le=1.0, description="First input in [0, 1]"),
    x2: float = Query(..., ge=0.0, le=1.0, description="Second input in [0, 1]")
):
    x = np.array([[x1, x2]])
    prediction = model.predict(x)
    return {"prediction": float(prediction[0][0])}
