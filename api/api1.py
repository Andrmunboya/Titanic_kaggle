import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Titanic Pipeline API")

# Загружаем сразу готовый конвейер (и препроцессинг, и модель внутри)
pipeline = joblib.load("models/titanic_pipeline.pkl")


class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    SibSp: int
    Parch: int


@app.post("/predict")
def predict_survival(passenger: PassengerData):
    # Данные прилетают в сыром виде, например: "Sex": "male"
    df_input = pd.DataFrame([passenger.dict()])

    # Передаем сырой DataFrame напрямую в пайплайн!
    pred = pipeline.predict(df_input)
    prob = pipeline.predict_proba(df_input)

    result = "ВЫЖИЛ" if pred[0] == 1 else "ПОГИБ"
    confidence = prob[0][1] if pred[0] == 1 else prob[0][0]

    return {
        "prediction": result,
        "confidence_percentage": round(confidence * 100, 2),
    }
