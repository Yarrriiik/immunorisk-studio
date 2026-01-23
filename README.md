# Immunorisk Studio

Immunorisk Studio — проект для обучения моделей по медицинским когортам и последующего инференса по единому контракту артефактов (модель + метаданные + метрики).

В репозитории уже есть пайплайн обучения, инференс-обвязка и утилиты для подготовки шаблонов входных данных.

## Структура репозитория

- `data_raw/` — исходные данные (как есть).
- `data_parquet/` — датасеты по когортам в формате Parquet (`<cohort>.parquet`).
- `src/cohorts/targets.py` — конфиги когорт: task/target/drop_cols/cat_candidates.
- `src/pipeline/train_all.py` — обучение “одной кнопкой” по всем когортам из `COHORT_TARGETS`.
- `src/pipeline/feature_cleaning.py` — фильтрация/санитайз признаков + отчётность по причинам дропа.
- `src/predict.py` — инференс: загрузка артефактов, выравнивание фич, predict для regression/binary/multiclass.
- `src/predict_cli.py` — CLI для прогона инференса без UI.
- `scripts/make_templates.py` — генерация шаблонов входов и подсказок (help) для каждой когорты.
- `artifacts/<cohort>/` — выход обучения: `model.cbm`, `meta.json`, `metrics.json`.
- `reports/` — отчёты `train_summary.csv` и `features_<cohort>.csv`.

> Примечание: `main.py` сейчас является заглушкой/тестовым файлом и не участвует в пайплайне.

## Установка окружения

Рекомендуем Python 3.10+.

В репозитории сейчас нет заполненного `requirements.txt`/`pyproject.toml`, поэтому зависимости ставятся вручную (см. импорты в `train_all.py`/`predict.py`).

Пример установки через venv:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -U pip
pip install numpy pandas scikit-learn catboost
```

Если планируется чтение Excel и стабильная работа с Parquet, обычно также нужны `openpyxl` и `pyarrow` (или другой engine для `pandas.read_parquet`).

## Обучение моделей

1) Подготовьте Parquet-файлы в `data_parquet/` в формате:
- `data_parquet/sepsis.parquet`
- `data_parquet/peritonitis.parquet`
- `data_parquet/ihd.parquet`
- `data_parquet/il2_postcovid.parquet`

2) Запустите обучение:

```bash
python src/pipeline/train_all.py
```

После выполнения появятся:
- `artifacts/<cohort>/model.cbm`, `meta.json`, `metrics.json`.
- `reports/train_summary.csv` (сводка обучения) и `reports/features_<cohort>.csv` (отчёт по фичам).

## Шаблоны входных данных (для демо и Streamlit)

Сгенерируйте шаблоны:

```bash
python scripts/make_templates.py
```

Скрипт создаст в `templates/`:
- `<cohort>_template.csv` — “пустой” шаблон с 1 строкой (чтобы файл не был пустым).
- `<cohort>_example.csv` — пример-заглушку для быстрого end-to-end прогона.
- `<cohort>_help.md` — подсказку с топ‑20 самых заполненных признаков (на базе `reports/features_<cohort>.csv`).

## Инференс (CLI, без UI)

Пример прогона для IHD:

```bash
python -m src.predict_cli --cohort ihd --input templates/ihd_example.csv
```

CLI загружает артефакты из `artifacts/<cohort>/`, выравнивает вход под `meta.json["features"]` и печатает результат предсказания.

Для binary-задач дополнительно используется порог `best_thr`, сохранённый в `metrics.json`.

## Инференс (API для приложения)

Основная точка входа для UI:

- `predict_df(cohort, df_raw) -> dict` в `src/predict.py`.
- `missing_columns_for_cohort(cohort, df_raw) -> list[str]` — чтобы подсветить недостающие поля в форме/файле.

Контракт результата:
- Regression: `{ "pred": [...] }`.
- Binary: `{ "proba": [...], "pred": [...], "best_thr": ... }`.
- Multiclass: `{ "pred": [...], "top3": [...], "proba": ... }`.

## Как добавить новую когорту

1) Добавить `data_parquet/<new_cohort>.parquet`.
2) Добавить конфиг в `src/cohorts/targets.py` (тип задачи, `target_col`, `drop_cols`, `cat_candidates`, `min_rows`).
3) Запустить `python src/pipeline/train_all.py` и убедиться, что артефакты/отчёты появились.
4) Сгенерировать новые шаблоны `python scripts/make_templates.py`.

## Статус UI (Streamlit)

Streamlit MVP пока не зафиксирован отдельным модулем.

План: UI будет тонкой обвязкой над `src/predict.py` + `templates/` (выбор когорты → загрузка/ввод данных → predict → вывод результата и подсказок).
