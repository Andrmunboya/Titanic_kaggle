import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Подключаем данные
data = pd.read_csv('train.csv')
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']]

# Фильтруем данные
data['Sex'] = data['Sex'].map({'male':1, 'female':0})
data['Age'] = data['Age'].fillna(data['Age'].median())

X = data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
y = data['Survived']

# Пилим данные 80 для обучения 20 для проверки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=42)

# Обучаем моделль на тренировочных данных 
model_evalForest = RandomForestClassifier(n_estimators = 100, criterion='entropy', max_depth=9, random_state=42)
model_evalForest.fit(X_train, y_train)

# Делаем прогноз
y_pred = model_evalForest.predict(X_val)

# Высчитываем точность
accuracy = accuracy_score(y_val, y_pred)

print("---РЕЗУЛЬТАТ ОЦЕНКИ МОДЕЛИ---")
print(f"Точность модели:{accuracy * 100:.2f}%")
print("МАТРИЦА ОШИБОК")

# Строим матрицу ошибок
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize = (6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Предсказанно Погиб', 'Предсказанно Выжил'],
            yticklabels=['Реально Погиб','Реально Выжил'])
plt.ylabel('Реальность')
plt.xlabel('Прогноз модели')
plt.title('Матрица ошибок')
plt.show()
