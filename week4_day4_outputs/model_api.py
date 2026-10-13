
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "api_model.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_names.txt")

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, encoding="utf-8") as file:
    FEATURE_NAMES = [line.strip() for line in file if line.strip()]

app = FastAPI(title="Breast Cancer Prediction API")

class PredictionInput(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature names and their numeric values"
    )

@app.get("/")
def home():
    return {"message": "Model API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PredictionInput):
    if set(data.features.keys()) != set(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail="Provide all required feature names, with no extras."
        )

    try:
        row = pd.DataFrame(
            [[data.features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        return {
            "prediction": prediction,
            "predicted_class": (
                "malignant" if prediction == 0 else "benign"
            ),
            "probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            }
        }
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))
