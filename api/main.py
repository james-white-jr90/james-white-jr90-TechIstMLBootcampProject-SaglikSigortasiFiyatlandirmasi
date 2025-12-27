from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# FastAPI uygulaması
app = FastAPI(
    title="Medical Cost Prediction API",
    description="Yaş, cinsiyet, BMI ve sigara bilgisine göre yıllık sağlık masrafı tahmini yapan API.",
    version="1.0.0"
)

# Modeli yükle
model_path = "../models/best_model.pkl"
best_model = joblib.load(model_path)

# İstek şeması (request body)
class CustomerFeatures(BaseModel):
    age: int
    sex: str       # "male" veya "female"
    bmi: float
    smoker: str    # "yes" veya "no"

# Yanıt şeması (response)
class PredictionResponse(BaseModel):
    predicted_cost: float


@app.get("/")
def root():
    return {"message": "Medical Cost Prediction API çalışıyor."}


@app.post("/predict", response_model=PredictionResponse)
def predict_cost(features: CustomerFeatures):
    # Gelen veriyi DataFrame'e çevir
    input_df = pd.DataFrame([{
        "age": features.age,
        "sex": features.sex,
        "bmi": features.bmi,
        "smoker": features.smoker
    }])

    # Tahmin
    pred = best_model.predict(input_df)[0]

    return PredictionResponse(predicted_cost=float(round(pred, 2)))
