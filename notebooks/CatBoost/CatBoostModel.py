import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1.Подключаем данные
data_train = pd.read_csv('train.csv')

# Выбираем признаки
features = ['Pclass','Sex','Age','Fare','SibSp','Parch']

# Убираем пропуски
data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())

X = data_train[features]
y = data_train['Survived']

# 1.1 Пилим данные
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Создаем модель CatBoost и настраиваем
# Указываем какие столбцы текстовые
cat_features_indices = ['Sex']

model_cat = CatBoostClassifier(
    iterations=350,
    learning_rate=0.05,
    depth=5,
    cat_features=cat_features_indices,
    random_seed=42,
    verbose=50
)
# 2.1 Обучение модели
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))

# 2.2 Проверка точности
y_pred = model_cat.predict(X_val)
cat_accuracy = accuracy_score(y_val, y_pred)

print(f"--- ТОЧНОСТЬ МОДЕЛИ:---{cat_accuracy*100:.2f}%")

# 3. Строим график важности признаков
importance = model_cat.get_feature_importance()
plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=features, palette='magma')
plt.title('Усредненный структурный сплит (Важность признаков в CatBoost)')
plt.xlabel('Сила признаков')
plt.show()
