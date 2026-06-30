# 1. Берем официальный легковесный образ Python
FROM python:3.10-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файл зависимостей внутрь контейнера
COPY api/requirements.txt .

# 4. Устанавливаем библиотеки без сохранения кэша (чтобы образ весил меньше)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем файлы твоего API и предобученный пайплайн
COPY api/ /app/api/
COPY models/titanic_pipeline.pkl /app/models/titanic_pipeline.pkl

# 6. Открываем порт 8000 для внешнего мира
EXPOSE 8000

# 7. Команда для запуска сервера при старте контейнера
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
