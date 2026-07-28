# Lead Scoring — пакет для деплоймента

Эта папка содержит изолированный деплоймент-пакет и не изменяет существующий код проекта.

## Что включает пакет

- Кастомную MLflow `pyfunc`-модель, которая содержит:
  - списки признаков
  - `processor.pkl`
  - файл обученной модели
  - логику препроцессинга для инференса
- FastAPI-приложение, которое отдаёт предсказания из зарегистрированной в MLflow модели
- Стандартный Python-пакет (`pyproject.toml`), который можно собрать в wheel

## Структура папки

- `src/lead_scoring_service/pyfunc_model.py`: Кастомная MLflow-обёртка модели
- `src/lead_scoring_service/api.py`: FastAPI-приложение
- `scripts/register_mlflow_model.py`: Логирует и регистрирует упакованную модель в MLflow

## Регистрация модели в MLflow

Из корня проекта:

```bash
python deployment_package/scripts/register_mlflow_model.py
```

Опциональные переменные окружения:

- `MODEL_FILE_PATH` (по умолчанию: `training_pipeline/models/lightgbm_best_model.pkl`)
- `PROCESSOR_FILE_PATH` (по умолчанию: `utils/processor.pkl`)
- `TRACKING_PATH` (по умолчанию: `experiments/mlruns`)
- `REGISTERED_MODEL_NAME` (по умолчанию: `LeadScoringService`)

## Запуск FastAPI

```bash
uvicorn lead_scoring_service.api:app --reload --app-dir deployment_package/src
```

Опциональные переменные окружения для API:

- `MLFLOW_TRACKING_URI` (по умолчанию указывает на локальный `experiments/mlruns`)
- `MODEL_URI` (по умолчанию: `models:/LeadScoringService/latest`)

## Сборка wheel-пакета

```bash
cd deployment_package
pip install build
python -m build
```

Wheel будет сгенерирован в `deployment_package/dist`.
