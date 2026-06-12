# Подключаем данные
test_data = pd.read_csv('test.csv')

# Заменяем пол на цифры
test_data['Sex'] = test_data['Sex'].map({'male':1, 'female':0})
X_test = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

# Делаем прогноз
test_prediction = model_evalForest.predict(X_test)

# Формируем таблицу для kaggle
submission = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : test_prediction
})

# Выгружаем таблицу с ответами
submission.to_csv('submission.csv', index=False)
print('Файл: submission.csv успешно создан!')
