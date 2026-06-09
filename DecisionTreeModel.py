import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Загружаем данные
data = pd.read_csv('train.csv')

# Признаки
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
data = data[features + ['Survived']]

# Превращаем пол в числа
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

X = data[features]
y = data['Survived']

# 2. Делим данные
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Обучаем модель
model_rich = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model_rich.fit(X_train, y_train)

# 4. Проверяем точность
y_pred = model_rich.predict(X_val)
new_accuracy = accuracy_score(y_val, y_pred)

print(f"--- ТОЧНОСТЬ МОДЕЛИ: {new_accuracy * 100:.2f}% ---")

# 5. Рисуем дерево
plt.figure(figsize=(20, 12))
plot_tree(model_rich, feature_names=X.columns, class_names=['Погиб', 'Выжил'], filled=True, rounded=True, fontsize=10)
plt.show()
