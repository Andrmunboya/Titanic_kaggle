# Упаковываем модель в API для веб сервиса

# 1. Устанавливаем FastAPI и Uvicorn (веб-сервер)
!pip install fastapi uvicorn -q

import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# ==========================================
# 1: СОХРАНЯЕМ НАШУ ОБУЧЕННУЮ МОДЕЛЬ НА ДИСК
# ==========================================
model_cat.save_model('catboost_model.bin')

# ==========================================
# 2: НАСТРАИВАЕМ ЛОГИРОВАНИЕ (LOGGING)
# ==========================================
# Создаем файл app.log, куда будем записывать данные сервера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='w', encoding='utf-8'),
        logging.StreamHandler() # Чтобы логи дублировались в консоль Colab
    ]
)
logger = logging.getLogger("Titanic_API")
logger.info("--- СИСТЕМА ЛОГИРОВАНИЯ УСПЕШНО ЗАПУЩЕНА ---")

# ==========================================
# 3: СОЗДАЕМ ВЕБ-ПРИЛОЖЕНИЕ FASTAPI
# ==========================================
app = FastAPI(title="Titanic Survival Predictor API")

# Описываем формат данных, который мы ждем от веб-интерфейса (схема JSON)
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    SibSp: int
    Parch: int

# Главная страница нашего веб-сервиса
@app.get("/")
def home():
    return {"message": "Привет! API Титаника работает. Отправляй POST запрос на /predict"}

# Эндпоинт (ссылка), которая принимает данные и делает предсказание
@app.post("/predict")
def predict_survival(passenger: PassengerData):
    logger.info(f"Получен веб-запрос на предсказание. Данные: {passenger.dict()}")

    # 1. Превращаем пришедший JSON в DataFrame для модели
    df_input = pd.DataFrame([passenger.dict()])

    # 2. Делаем предсказание нашей сохраненной моделью
    pred = model_cat.predict(df_input)
    prob = model_cat.predict_proba(df_input)

    result = "ВЫЖИЛ" if pred[0] == 1 else "ПОГИБ"
    confidence = prob[0][1] if pred[0] == 1 else prob[0][0]

    # 3. Пишем результат в ЛОГИ
    logger.info(f"Результат прогноза: Пассажир {result} с уверенностью {confidence*100:.2f}%")

    return {
        "prediction": result,
        "confidence_percentage": round(confidence * 100, 2)
    }

# ==========================================
# 4: ТЕСТИРУЕМ НАШЕ API
# ==========================================
from fastapi.testclient import TestClient
client = TestClient(app)

logger.info("Запуск имитации реального веб-запроса от клиента...")

# Имитируем, что сайт отправил нам данные Дяди Стэна (3 класс, мужчина, 22 года, дешевый билет)
test_passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "Fare": 7.25,
    "SibSp": 0,
    "Parch": 0
}

response = client.post("/predict", json=test_passenger)

print("\n--- ОТВЕТ ВЕБ-СЕРВЕРА (JSON) ---")
print(response.json())
