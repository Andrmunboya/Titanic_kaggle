import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================
# Читай напрямую из своей папки data/
data = pd.read_csv("../data/train.csv")

# Выделяем фичи и таргет, как в твоих прошлых скриптах
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]
X = data[features]
y = data["Survived"]

# Разбиваем на train/val (80/20), фиксируем random_state=42
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ==========================================
# 2. ПРОЕКТИРОВАНИЕ COLUMNTRANSFORMER
# ==========================================
# Нам нужно разделить обработку числовых и категориальных колонок

# Для чисел: заполняем пропуски медианой (работает внутри пайплайна изолированно!)
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

# Для текста (Sex): заполняем пропуски (если будут) и кодируем в One-Hot векторы
categorical_features = ["Sex"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ]
)

# Собираем процессоры воедино. Колонка Pclass останется без изменений (passthrough)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)

# ==========================================
# 3. СБОРКА МОНОЛИТНОГО ПАЙПЛАЙНА С МОДЕЛЬЮ
# ==========================================
# Объединяем препроцессинг и твою лучшую модель RandomForest в один конвейер
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                criterion="entropy",
                max_depth=9,
                random_state=42,
            ),
        ),
    ]
)

# ==========================================
# 4. ОБУЧЕНИЕ И ВАЛИДАЦИЯ ОДНОЙ КОМАНДОЙ
# ==========================================
print("--- ЗАПУСК ОБУЧЕНИЯ ПАЙПЛАЙНА ---")
# Входные данные идут "сырыми" — со строками 'male'/'female' и пропусками в Age!
pipeline.fit(X_train, y_train)

# Делаем предсказание на валидации
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Точность пайплайна на валидации: {accuracy * 100:.2f}%")

# ==========================================
# 5. СОХРАНЕНИЕ КОНВЕЙЕРА НА ДИСК
# ==========================================
# Вместо сохранения только весов модели, мы сохраняем ВЕСЬ пайплайн целиком!
# Создай у себя папку models/ если её нет
import os

os.makedirs("../models", exist_ok=True)
joblib.dump(pipeline, "../models/titanic_pipeline.pkl")
print("Пайплайн успешно сохранен в 'models/titanic_pipeline.pkl'!")
