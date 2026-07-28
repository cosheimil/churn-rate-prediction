<div align="center">
  <h1>End-to-End ML Pipeline для предсказания оттока клиентов с MLOps и деплойментом</h1>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20Optuna-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/MLOps-DVC%20%26%20MLflow-lightblue?style=flat-square"/>
  <img src="https://img.shields.io/badge/API-FastAPI%20%7C%20SHAP-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/%D0%A2%D0%B5%D1%81%D1%82%D1%8B-23%2F23%20%D0%BF%D1%80%D0%BE%D0%B9%D0%B4%D0%B5%D0%BD%D0%BE-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Docker-%D0%93%D0%BE%D1%82%D0%BE%D0%B2-brightgreen?style=flat-square"/>
</p>

---

## 🧠 Бизнес-задача

Телеком-компании теряют миллионы долларов ежегодно из-за оттока клиентов. Этот проект строит систему машинного обучения, которая предсказывает, какие клиенты с высокой вероятностью уйдут, что позволяет запускать проактивные кампании по удержанию.

---

## 🎯 Цель

Построить модель бинарной классификации, предсказывающую отток клиента (Churn — Yes/No), на основе его демографических данных, потребляемых услуг и платёжной истории.

---

## 🛠 Технологический стек

| Область | Технологии |
|--------|-----------|
| **Язык** | Python 3.11+ |
| **ML-модели** | LightGBM, XGBoost, Logistic Regression, Random Forest |
| **Подбор гиперпараметров** | Optuna (50 trials) |
| **Трекинг экспериментов** | MLflow |
| **Оркестрация пайплайна** | DVC (`dvc repro`) |
| **Объяснимость (XAI)** | SHAP |
| **API-сервер** | FastAPI + Uvicorn |
| **Фронтенд** | Vanilla HTML/CSS/JS (тёмная тема) |
| **Мониторинг** | Streamlit дашборд |
| **Контейнеризация** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Тестирование** | pytest (23 теста) |

---

## 📊 Датасет

IBM Telco Customer Churn (синтетический) — **7 043 записи, 20 признаков**:

- **Демография**: gender, SeniorCitizen, Partner, Dependents
- **Аккаунт**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Услуги**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Платежи**: MonthlyCharges, TotalCharges
- **Целевая переменная**: Churn (Yes/No)

---

## 📁 Структура проекта

```
├── data/
│   ├── raw/telco_churn.csv          # Входной датасет
│   └── generate_dataset.py          # Генератор синтетических данных
├── training_pipeline/
│   ├── components/
│   │   ├── data_ingestion.py        # Этап 1: Загрузка сырых данных
│   │   ├── data_preprocessing.py    # Этап 2: Разбиение, масштабирование, кодирование
│   │   └── model_training.py        # Этап 3: Обучение LightGBM + MLflow
│   ├── pipelines/pipeline.py        # Оркестратор
│   ├── models/                      # Артефакты обученной модели
│   └── metrics/                     # Метрики обучения в JSON
├── experiments/
│   ├── base_model.py                # Бенчмарк 4 моделей
│   └── tuning.py                    # Optuna-тюнинг гиперпараметров
├── utils/
│   ├── features.py                  # Определения колонок-признаков
│   └── utils.py                     # PreprocessorManager, VIF, хелперы
├── inference/
│   ├── inference.py                 # Скрипт батчевого инференса
│   ├── new_data.csv                 # Пример входных данных
│   └── predictions.csv              # Результаты предсказаний
├── deployment_package/
│   ├── src/lead_scoring_service/
│   │   ├── api.py                   # FastAPI (predict, explain, stats)
│   │   └── pyfunc_model.py          # Обёртка MLflow PythonModel
│   └── scripts/register_mlflow_model.py
├── frontend/index.html              # UI для предсказания оттока
├── dashboard/monitoring.py          # Streamlit дашборд мониторинга
├── docker/
│   ├── nginx.conf                   # Конфигурация nginx для фронтенда
│   └── Dockerfile.frontend          # Контейнер фронтенда
├── tests/                           # pytest (23 теста)
├── .github/workflows/ci.yml         # GitHub Actions CI/CD
├── Dockerfile                       # Контейнер API
├── docker-compose.yml               # Оркестрация всего стека
├── dvc.yaml                         # DVC-пайплайн
└── requirements.txt                 # Зависимости
```

---

## 🚀 Быстрый старт

### 1. Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Генерация датасета и обучение

```bash
python data/generate_dataset.py
python training_pipeline/pipelines/pipeline.py
```

Или через DVC:

```bash
dvc repro
```

### 3. Запуск инференса

```bash
python inference/inference.py
```

### 4. Запуск API

```bash
uvicorn lead_scoring_service.api:app --reload --app-dir deployment_package/src
```

Эндпоинты API:
- `GET /health` — Проверка работоспособности
- `GET /stats` — Статистика модели
- `GET /model/info` — Информация о признаках
- `POST /predict` — Предсказание оттока
- `POST /predict/explain` — Предсказание + SHAP-объяснение (вклад каждого признака)

### 5. Открытие фронтенда

Открой `frontend/index.html` в браузере.

### 6. Запуск дашборда

```bash
streamlit run dashboard/monitoring.py
```

### 7. Docker

```bash
docker-compose up --build
```

- API: `http://localhost:8000`
- Фронтенд: `http://localhost:8080`

### 8. Запуск тестов

```bash
python -m pytest tests/ -v
```

---

## 🔥 Ключевые фичи

- **Бенчмарк 4 моделей**: LightGBM vs XGBoost vs Logistic Regression vs Random Forest — всё логируется в MLflow
- **Optuna-тюнинг**: 50-итерационный поиск гиперпараметров с максимизацией macro F1
- **DVC-пайплайн**: Воспроизводимый 3-этапный пайплайн с отслеживанием зависимостей
- **SHAP-объяснимость**: Анализ вклада каждого признака в конкретное предсказание
- **Streamlit дашборд**: Интерактивный мониторинг модели с метриками, предсказаниями и важностью признаков
- **Docker Compose**: Продакшен-готовые контейнеры (API + фронтенд через nginx)
- **GitHub Actions CI/CD**: Автоматическое тестирование и сборка Docker-образа при пуше
- **23 pytest-теста**: Покрытие препроцессинга, пайплайна, инференса и утилит

---

## 📈 Метрики модели

| Метрика | Churn=No | Churn=Yes |
|--------|----------|-----------|
| F1 Score | 0.645 | 0.630 |
| Precision | 0.650 | 0.626 |
| Recall | 0.641 | 0.635 |
