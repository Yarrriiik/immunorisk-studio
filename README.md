# Immunorisk Studio

Immunorisk Studio — учебная система для обучения моделей по медицинским когортам и последующего инференса через единый контракт артефактов: `model.cbm + meta.json + metrics.json`.

Проект уже включает:

- ML pipeline для подготовки и обучения;
- Streamlit-приложение для демонстрации и ручного ввода;
- CLI-инференс;
- шаблоны и demo-кейсы для защиты/показа проекта.

## Что показывать на демонстрации

Если нужно быстро показать проект на защите, оптимальный сценарий такой:

1. Запустить Streamlit: `python -m streamlit run immunorisk_app.py`.
2. Войти или зарегистрировать тестового пользователя.
3. Выбрать когорту в sidebar.
4. Открыть `Библиотека примеров` и загрузить `Минимальный` или `Расширенный` кейс.
5. Показать:
   - заполнение минимального профиля;
   - дополнительные параметры;
   - автосохранение и ручной черновик;
   - экспорт текущего ввода в `JSON/CSV`;
   - финальный прогноз.

Для быстрого показа без ручного ввода используйте готовые файлы из `demo_cases/`.

## Структура репозитория

- `data_raw/` — исходные данные (как есть).
- `data_parquet/` — датасеты по когортам в формате Parquet (`<cohort>.parquet`).
- `demo_cases/` — готовые учебные JSON-кейсы для демонстрации UI и инференса.
- `src/cohorts/targets.py` — конфиги когорт: task/target/drop_cols/cat_candidates.
- `src/pipeline/train_all.py` — обучение “одной кнопкой” по всем когортам из `COHORT_TARGETS`.
- `src/pipeline/feature_cleaning.py` — фильтрация/санитайз признаков + отчётность по причинам дропа.
- `src/predict.py` — инференс: загрузка артефактов, выравнивание фич, predict для regression/binary/multiclass.
- `src/predict_cli.py` — CLI для прогона инференса без UI.
- `scripts/make_templates.py` — генерация шаблонов входов и подсказок (help) для каждой когорты.
- `artifacts/<cohort>/` — выход обучения: `model.cbm`, `meta.json`, `metrics.json`.
- `reports/` — отчёты `train_summary.csv` и `features_<cohort>.csv`.
- `drafts/` — локальные ручные черновики и автосохранения Streamlit-приложения. Каталог не должен коммититься.

> Примечание: `main.py` сейчас является заглушкой/тестовым файлом и не участвует в пайплайне.

## Установка окружения

Рекомендуем Python 3.10+.

В проекте есть минимальный `requirements.txt` для локального запуска.

Пример установки через venv:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

Если планируется чтение Excel и стабильная работа с Parquet, обычно также нужны `openpyxl` и `pyarrow` (или другой engine для `pandas.read_parquet`).

## Быстрый запуск приложения

```bash
python -m streamlit run immunorisk_app.py
```

Если нужен администраторский сброс пароля, задайте секрет перед запуском.

PowerShell:

```powershell
$env:IMMUNORISK_ADMIN_RESET_CODE="demo-reset-code"
python -m streamlit run immunorisk_app.py
```

## Локальные данные и безопасность

- Пользовательская база хранится локально в `users_db.local.json` и не должна коммититься в репозиторий.
- Для администраторского сброса пароля задайте переменную окружения `IMMUNORISK_ADMIN_RESET_CODE`.
- При необходимости путь к локальной базе пользователей можно переопределить через `IMMUNORISK_USERS_DB`.
- Автосохранения и ручные черновики хранятся локально в `drafts/`.

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

## Demo-кейсы

В `demo_cases/` лежат готовые JSON-файлы:

- `sepsis_minimal.json`, `sepsis_extended.json`
- `peritonitis_minimal.json`, `peritonitis_extended.json`
- `ihd_minimal.json`, `ihd_extended.json`
- `il2_postcovid_minimal.json`, `il2_postcovid_extended.json`

Эти кейсы подходят для:

- загрузки в Streamlit через режим `Файл`;
- вставки в режим `JSON`;
- демонстрации минимального и расширенного ручного профиля.

## Инференс (CLI, без UI)

Пример прогона для IHD:

```bash
python -m src.predict_cli --cohort ihd --input templates/ihd_example.csv
```

CLI загружает артефакты из `artifacts/<cohort>/`, выравнивает вход под `meta.json["features"]` и печатает результат предсказания.

Для binary-задач дополнительно используется порог `best_thr`, сохранённый в `metrics.json`.

Если нужно быстро проверить уже готовые артефакты, можно взять поля из `demo_cases/*.json` и перенести их в CSV или воспользоваться `templates/*_example.csv`.

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

Текущий UI уже поддерживает:

- аутентификацию пользователей;
- выбор когорты;
- режимы ввода `Форма / Текст / JSON / Файл`;
- минимальные и расширенные примеры;
- библиотеку примеров пациентов;
- ручные черновики и автосохранение;
- экспорт текущего ввода в `JSON/CSV`;
- историю анализов;
- вывод прогноза и базовых метрик модели.

Это всё ещё учебный MVP, а не production-медицинская система, но для курсового проекта приложение уже готово к демонстрации end-to-end.
