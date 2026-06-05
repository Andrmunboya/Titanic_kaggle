# 1. Загружаем тестовые данные
test_data = pd.read_csv('test.csv')

# 2. Готовим признаки (точно так же, как для train)
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})

# Заполняем пропуски, чтобы код не упал (в тесте есть пустые Age и Fare)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Выделяем матрицу признаков
X_test = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

# 3. Делаем предсказание обученной моделью
test_predictions = model_rich.predict(X_test)

# 4. Формируем финальный файл для Kaggle
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

# 5. Выгружаем в файл
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно создан!")
