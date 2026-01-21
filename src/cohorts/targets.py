# src/cohorts/targets.py
# Конфиги таргетов/задач по когортам. Остальные когорты можно включить позже,

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskType = Literal["regression", "classification", "multiclass"]


@dataclass(frozen=True)
class CohortTarget:
    cohort: str
    task: TaskType
    target_col: str
    # какие колонки точно не должны попадать в признаки
    drop_cols: tuple[str, ...] = ()
    # кандидаты категориальных колонок (оставим только те, что реально есть)
    cat_candidates: tuple[str, ...] = ("sex", "Пол")
    # минимальное число строк после dropna по таргету
    min_rows: int = 30


COHORT_TARGETS: dict[str, CohortTarget] = {
    "sepsis": CohortTarget(
        cohort="sepsis",
        task="regression",
        target_col="Шкала SOFA",
        drop_cols=(
            "patient_id",
            "visit_dt",
            "Дата анализа",
            "Дата анализа.1",
            "Диагноз",
            "Статус",
            "Шкала SAPSII",
            "Шкала SAPSII.1",
        ),
        min_rows=30,
    ),

    # IMPORTANT: в peritonitis у SOFA много пропусков, но для учебного проекта
    # обучаемся на том, что есть (после dropna по таргету).
    "peritonitis": CohortTarget(
        cohort="peritonitis",
        task="regression",
        target_col="SOFA",
        drop_cols=(
            "patient_id",
            "visit_dt",
            "Дата",
            "Диагноз",
            "Исход",
            "SAPS II",
        ),
        min_rows=30,
    ),

    "ihd": CohortTarget(
        cohort="ihd",
        task="classification",
        target_col="2 точка (резистентность 0-нет, 1-да)",
        drop_cols=(
            "patient_id",
            "visit_dt",
            "Диагноз",
            "Терапия (б-блокаторы)",
            "Исходы во время госпитализации",
            "Исходы отдалённые",
            "Общие исходы",
            "П/о осложнения в общем",
        ),
        min_rows=50,
    ),

    # По выводу inspect_targets это 4 класса: ПостКовид_0..3
    "il2_postcovid": CohortTarget(
        cohort="il2_postcovid",
        task="multiclass",
        target_col="Постковид",
        drop_cols=(
            "patient_id",
            "visit_dt",
            "Диагноз",
        ),
        min_rows=30,
    ),
}
