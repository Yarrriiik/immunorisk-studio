"""
Dynamic Form Generator - Creates cohort-specific input forms.
"""

from typing import Any, Dict, Set

import streamlit as st

from ml_service import get_model_cohort_name
from src.predict import load_artifacts


def _spec(name: str, label: str | None = None) -> dict[str, str]:
    return {"name": name, "label": label or name}


COHORT_FIELD_PROFILES: dict[str, dict[str, Any]] = {
    "Сепсис": {
        "title": "Минимальный профиль для сепсиса",
        "description": "Базовый набор для быстрого запуска прогноза по клинико-лабораторным данным.",
        "recommended_extra_features": [
            "WBC Лейкоциты",
            "HGB Гемоглобин",
            "HCT Гематокрит",
            "Глюкоза",
            "Общий белок",
            "Фибриноген",
            "МНО",
            "АЧТВ",
        ],
        "sections": [
            {
                "title": "Основная информация",
                "fields": [_spec("patient_id", "ID пациента"), _spec("age", "Возраст (лет)"), _spec("sex", "Пол")],
            },
            {
                "title": "Шкалы и воспаление",
                "fields": [
                    _spec("sofa", "Шкала SOFA"),
                    _spec("glasgow", "Шкала Глазго"),
                    _spec("leukocytes", "Лейкоциты"),
                    _spec("neutrophils", "Нейтрофилы"),
                    _spec("lymphocytes", "Лимфоциты"),
                    _spec("crp", "С-реактивный белок (СРБ)"),
                ],
            },
            {
                "title": "Органная дисфункция",
                "fields": [
                    _spec("platelets", "Тромбоциты"),
                    _spec("creatinine", "Креатинин"),
                    _spec("urea", "Мочевина"),
                    _spec("bilirubin", "Билирубин общий"),
                    _spec("potassium", "Калий"),
                    _spec("sodium", "Натрий"),
                ],
            },
        ],
    },
    "Перитонит": {
        "title": "Минимальный профиль для перитонита",
        "description": "Упор на хирургический анамнез, тяжесть процесса и базовые лабораторные показатели.",
        "recommended_extra_features": [
            "Продолжительность лечения в ЦРБ",
            "продолжительность пребывания в РО (сут.) ЦРБ",
            "продолж-ть ИВЛ (Ч.) ЦРБ",
            "продол-ть пребыв. в ХО (сут.) ЦРБ",
            "Возбудитель",
            "ИД течение",
            "ИД этиология",
            "пал/яд %",
            "пал/яд",
            "сегм/ яд %",
            "сегм/ яд",
            "Нейтрофилы, абс",
            "баз. %",
            "баз.",
            "эоз. %",
            "эоз.",
            "мон. %",
            "мон.",
            "лимф. %",
            "лимф.",
            "IgA",
            "IgG",
            "IgM",
            "IgE",
            "Hu IL-6",
            "Hu IL-8",
            "Hu IL-10",
            "Hu TNF-a",
            "CD3+",
            "CD3+CD4+",
            "CD3+CD8+",
            "CD19+",
            "CD19+CD27+",
            "CD19+CD5+",
            "CD19+CD23+",
            "CD16/56+",
            "MON=HLA-DR+",
        ],
        "sections": [
            {
                "title": "Основная информация",
                "fields": [_spec("patient_id", "ID пациента"), _spec("sex", "Пол")],
            },
            {
                "title": "Хирургический анамнез",
                "fields": [
                    _spec("Количество санаций на момент исследования"),
                    _spec("Продолжительность заболевания до госпитализации"),
                    _spec("Продолжительность пребывания в РО (сут.)"),
                    _spec("Продолжительность пребывания в ХО (сут.)"),
                    _spec("Количество операций"),
                    _spec("Причина перитонита"),
                ],
            },
            {
                "title": "Тяжесть и лаборатория",
                "fields": [
                    _spec("SIRS"),
                    _spec("MIP"),
                    _spec("ИБП"),
                    _spec("Степень тяжести"),
                    _spec("Лейкоциты"),
                    _spec("СОЭ"),
                ],
            },
        ],
    },
    "ИБС": {
        "title": "Минимальный профиль для ИБС",
        "description": "Собран учебный набор дооперационных и базовых клинических признаков без признаков из будущих временных точек.",
        "recommended_extra_features": [
            "Агр. Чувствит Арах. (0.5 мМ)",
            "Агр. Чувствит Арах.+АСК",
            "Гемостаз АЧТВ",
            "Гемостаз МНО",
            "Гемостаз Фибриноген",
            "ЛПНП",
            "ЛПВП",
            "ТГ",
        ],
        "sections": [
            {
                "title": "Основная информация",
                "fields": [_spec("patient_id", "ID пациента"), _spec("age", "Возраст (лет)"), _spec("sex", "Пол")],
            },
            {
                "title": "Лаборатория и гемодинамика",
                "fields": [
                    _spec("ОАК лейкоциты"),
                    _spec("ОАК эритроциты"),
                    _spec("гемоглобин"),
                    _spec("тромбоциты"),
                    _spec("БАК креатинин"),
                    _spec("глюкоза"),
                    _spec("ХС общий"),
                    _spec("ЧСС уд./мин"),
                    _spec("ФВ до операции"),
                ],
            },
            {
                "title": "Операция и коморбидность",
                "fields": [
                    _spec("АКШ кол-во шунтов,общее"),
                    _spec("кол-во вен шунт (в)"),
                    _spec("кол-во артер шунт (а)"),
                    _spec("кол-во мамм (м)"),
                    _spec("Время ИК, мин"),
                    _spec("На работающем сердце"),
                    _spec("Курение"),
                    _spec("Артериальная гипертония"),
                    _spec("СД"),
                ],
            },
        ],
    },
    "IL-2 постковид": {
        "title": "Минимальный профиль для IL-2 постковида",
        "description": "Форма выделяет наиболее интерпретируемую часть иммунологической панели для ручного ввода.",
        "recommended_extra_features": [
            "Моноциты, %",
            "Моноциты, абс",
            "CD3+CD45R0-",
            "CD3+CD45R0+",
            "CD3+CD25+",
            "CD3+CD45R+CD62L-",
            "CD3+CD45R+CD62L+",
            "CD3+CD45R0-CD62L-",
            "CD3+CD45R-CD62L+",
            "CD3+CD4+CD25+",
            "CD3+CD4+CD45R0+CD62L-",
            "CD3+CD4+CD45R0+CD62L+",
            "CD3+CD4+CD45R0-CD62L-",
            "CD3+CD4+CD45R0-CD62L+",
            "T-reg CD45R0+CD62L-",
            "T-reg CD45R0+CD62L+",
            "T-reg CD45R0-CD62L-",
            "T-reg CD45R0-CD62L+",
            "CD19+CD5+",
            "CD19+CD5-",
            "CD19+CD27+",
            "CD19+CD27-",
            "CD19+CD23+",
            "CD19+CD5+CD23+",
            "CD19+CD27+CD23+",
            "NK 56+16+",
            "NK 56+16-",
            "NK 56dim16+",
            "NK 56dim16-",
            "NK 62L+57- exp 57",
            "NK 62L+57+ exp 57",
            "T-regCD62L",
            "B-cells CD5+27+",
        ],
        "sections": [
            {
                "title": "Основная информация",
                "fields": [_spec("patient_id", "ID пациента"), _spec("age", "Возраст (лет)"), _spec("sex", "Пол")],
            },
            {
                "title": "Общий анализ и индексы",
                "fields": [
                    _spec("Общие лейкоциты"),
                    _spec("Гранулоциты"),
                    _spec("Гранулоциты абс."),
                    _spec("Лимфоциты, %"),
                    _spec("Лимфоциты, абс"),
                    _spec("ЛИИ"),
                ],
            },
            {
                "title": "Иммунологический профиль",
                "fields": [
                    _spec("CD3+"),
                    _spec("Th"),
                    _spec("CD3+CD4-(Tcyt)"),
                    _spec("T-reg"),
                    _spec("B-cells"),
                    _spec("NK"),
                    _spec("NK_abs"),
                ],
            },
        ],
    },
}


def _is_age_feature(feature: str) -> bool:
    feature_lower = feature.lower()
    return "возраст" in feature_lower or feature_lower in {"age", "возраст", "лет"}


def _is_sex_feature(feature: str) -> bool:
    feature_lower = feature.lower()
    return feature_lower in {"sex", "пол", "пол.1"}


def get_cohort_features(cohort: str) -> Dict[str, Any]:
    model_cohort = get_model_cohort_name(cohort)
    if not model_cohort:
        return {"features": [], "cat_features": set()}

    try:
        artifacts = load_artifacts(model_cohort)
        return {
            "features": artifacts.features,
            "cat_features": set(artifacts.cat_features),
        }
    except Exception:
        return {"features": [], "cat_features": set()}


def get_minimal_profile(cohort: str) -> dict[str, Any]:
    profile = COHORT_FIELD_PROFILES.get(cohort)
    if profile:
        return profile

    return {
        "title": f"Минимальный профиль для {cohort}",
        "description": "Для этой когорты пока используется универсальная форма.",
        "sections": [{"title": "Основная информация", "fields": [_spec("patient_id", "ID пациента"), _spec("age", "Возраст (лет)"), _spec("sex", "Пол")]}],
    }


def get_input_examples(cohort: str, variant: str = "minimal") -> dict[str, str]:
    examples = {
        "Сепсис": {
            "minimal": {
                "text": """patient_id = P-001
age = 62
sex = Мужской
sofa = 7
glasgow = 14
leukocytes = 14.2
neutrophils = 82
lymphocytes = 8
crp = 124
platelets = 180
creatinine = 110
urea = 9.2
bilirubin = 18
potassium = 4.3
sodium = 138""",
                "json": """{
  "patient_id": "P-001",
  "age": 62,
  "sex": "Мужской",
  "sofa": 7,
  "glasgow": 14,
  "leukocytes": 14.2,
  "neutrophils": 82,
  "lymphocytes": 8,
  "crp": 124,
  "platelets": 180,
  "creatinine": 110,
  "urea": 9.2,
  "bilirubin": 18,
  "potassium": 4.3,
  "sodium": 138
}""",
            },
            "extended": {
                "text": """patient_id = P-001
age = 62
sex = Мужской
sofa = 7
glasgow = 14
leukocytes = 14.2
neutrophils = 82
lymphocytes = 8
crp = 124
platelets = 180
creatinine = 110
urea = 9.2
bilirubin = 18
potassium = 4.3
sodium = 138
WBC Лейкоциты = 14.2
HGB Гемоглобин = 118
HCT Гематокрит = 35.0
Глюкоза = 8.4
Общий белок = 61
Фибриноген = 5.6
МНО = 1.3
АЧТВ = 38""",
                "json": """{
  "patient_id": "P-001",
  "age": 62,
  "sex": "Мужской",
  "sofa": 7,
  "glasgow": 14,
  "leukocytes": 14.2,
  "neutrophils": 82,
  "lymphocytes": 8,
  "crp": 124,
  "platelets": 180,
  "creatinine": 110,
  "urea": 9.2,
  "bilirubin": 18,
  "potassium": 4.3,
  "sodium": 138,
  "WBC Лейкоциты": 14.2,
  "HGB Гемоглобин": 118,
  "HCT Гематокрит": 35.0,
  "Глюкоза": 8.4,
  "Общий белок": 61,
  "Фибриноген": 5.6,
  "МНО": 1.3,
  "АЧТВ": 38
}""",
            },
        },
        "Перитонит": {
            "minimal": {
                "text": """patient_id = P-002
sex = Мужской
Количество санаций на момент исследования = 2
Продолжительность заболевания до госпитализации = 3
Продолжительность пребывания в РО (сут.) = 2
Продолжительность пребывания в ХО (сут.) = 5
Количество операций = 1
Причина перитонита = 1
SIRS = 3
MIP = 2
ИБП = 1
Степень тяжести = 2
Лейкоциты = 12.1
СОЭ = 25
Возбудитель = 1
Hu IL-6 = 18.5""",
                "json": """{
  "patient_id": "P-002",
  "sex": "Мужской",
  "Количество санаций на момент исследования": 2,
  "Продолжительность заболевания до госпитализации": 3,
  "Продолжительность пребывания в РО (сут.)": 2,
  "Продолжительность пребывания в ХО (сут.)": 5,
  "Количество операций": 1,
  "Причина перитонита": 1,
  "SIRS": 3,
  "MIP": 2,
  "ИБП": 1,
  "Степень тяжести": 2,
  "Лейкоциты": 12.1,
  "СОЭ": 25,
  "Возбудитель": 1,
  "Hu IL-6": 18.5
}""",
            },
            "extended": {
                "text": """patient_id = P-002
sex = Мужской
Количество санаций на момент исследования = 2
Продолжительность заболевания до госпитализации = 3
Продолжительность пребывания в РО (сут.) = 2
Продолжительность пребывания в ХО (сут.) = 5
Количество операций = 1
Причина перитонита = 1
SIRS = 3
MIP = 2
ИБП = 1
Степень тяжести = 2
Лейкоциты = 12.1
СОЭ = 25
Продолжительность лечения в ЦРБ = 4
Возбудитель = 1
ИД течение = 2
ИД этиология = 1
пал/яд % = 9
сегм/ яд % = 72
Нейтрофилы, абс = 8.7
лимф. % = 14
IgA = 2.1
IgG = 11.8
IgM = 1.4
Hu IL-6 = 18.5
Hu IL-8 = 22.0
Hu IL-10 = 7.4
Hu TNF-a = 12.3
CD3+ = 64
CD3+CD4+ = 38
CD3+CD8+ = 24
CD19+ = 11
CD19+CD27+ = 4
CD16/56+ = 13
MON=HLA-DR+ = 76""",
                "json": """{
  "patient_id": "P-002",
  "sex": "Мужской",
  "Количество санаций на момент исследования": 2,
  "Продолжительность заболевания до госпитализации": 3,
  "Продолжительность пребывания в РО (сут.)": 2,
  "Продолжительность пребывания в ХО (сут.)": 5,
  "Количество операций": 1,
  "Причина перитонита": 1,
  "SIRS": 3,
  "MIP": 2,
  "ИБП": 1,
  "Степень тяжести": 2,
  "Лейкоциты": 12.1,
  "СОЭ": 25,
  "Продолжительность лечения в ЦРБ": 4,
  "Возбудитель": 1,
  "ИД течение": 2,
  "ИД этиология": 1,
  "пал/яд %": 9,
  "сегм/ яд %": 72,
  "Нейтрофилы, абс": 8.7,
  "лимф. %": 14,
  "IgA": 2.1,
  "IgG": 11.8,
  "IgM": 1.4,
  "Hu IL-6": 18.5,
  "Hu IL-8": 22.0,
  "Hu IL-10": 7.4,
  "Hu TNF-a": 12.3,
  "CD3+": 64,
  "CD3+CD4+": 38,
  "CD3+CD8+": 24,
  "CD19+": 11,
  "CD19+CD27+": 4,
  "CD16/56+": 13,
  "MON=HLA-DR+": 76
}""",
            },
        },
        "ИБС": {
            "minimal": {
                "text": """patient_id = P-003
age = 67
sex = Мужской
ОАК лейкоциты = 6.5
ОАК эритроциты = 4.6
гемоглобин = 135
тромбоциты = 210
БАК креатинин = 90
глюкоза = 5.8
ХС общий = 4.7
ЧСС уд./мин = 72
ФВ до операции = 56
АКШ кол-во шунтов,общее = 3
кол-во вен шунт (в) = 2
кол-во артер шунт (а) = 1
Время ИК, мин = 82""",
                "json": """{
  "patient_id": "P-003",
  "age": 67,
  "sex": "Мужской",
  "ОАК лейкоциты": 6.5,
  "ОАК эритроциты": 4.6,
  "гемоглобин": 135,
  "тромбоциты": 210,
  "БАК креатинин": 90,
  "глюкоза": 5.8,
  "ХС общий": 4.7,
  "ЧСС уд./мин": 72,
  "ФВ до операции": 56,
  "АКШ кол-во шунтов,общее": 3,
  "кол-во вен шунт (в)": 2,
  "кол-во артер шунт (а)": 1,
  "Время ИК, мин": 82
}""",
            },
            "extended": {
                "text": """patient_id = P-003
age = 67
sex = Мужской
ОАК лейкоциты = 6.5
ОАК эритроциты = 4.6
гемоглобин = 135
тромбоциты = 210
БАК креатинин = 90
глюкоза = 5.8
ХС общий = 4.7
ЧСС уд./мин = 72
ФВ до операции = 56
АКШ кол-во шунтов,общее = 3
кол-во вен шунт (в) = 2
кол-во артер шунт (а) = 1
Время ИК, мин = 82
Агр. Чувствит Арах. (0.5 мМ) = 54
Агр. Чувствит Арах.+АСК = 21
Гемостаз АЧТВ = 31
Гемостаз МНО = 1.0
Гемостаз Фибриноген = 3.4
ЛПНП = 2.8
ЛПВП = 1.0
ТГ = 1.6""",
                "json": """{
  "patient_id": "P-003",
  "age": 67,
  "sex": "Мужской",
  "ОАК лейкоциты": 6.5,
  "ОАК эритроциты": 4.6,
  "гемоглобин": 135,
  "тромбоциты": 210,
  "БАК креатинин": 90,
  "глюкоза": 5.8,
  "ХС общий": 4.7,
  "ЧСС уд./мин": 72,
  "ФВ до операции": 56,
  "АКШ кол-во шунтов,общее": 3,
  "кол-во вен шунт (в)": 2,
  "кол-во артер шунт (а)": 1,
  "Время ИК, мин": 82,
  "Агр. Чувствит Арах. (0.5 мМ)": 54,
  "Агр. Чувствит Арах.+АСК": 21,
  "Гемостаз АЧТВ": 31,
  "Гемостаз МНО": 1.0,
  "Гемостаз Фибриноген": 3.4,
  "ЛПНП": 2.8,
  "ЛПВП": 1.0,
  "ТГ": 1.6
}""",
            },
        },
        "IL-2 постковид": {
            "minimal": {
                "text": """patient_id = P-004
age = 45
sex = Женский
Общие лейкоциты = 5.8
Гранулоциты = 60
Гранулоциты абс. = 3.1
Лимфоциты, % = 30
Лимфоциты, абс = 1.6
ЛИИ = 1.2
CD3+ = 68
Th = 42
CD3+CD4-(Tcyt) = 24
T-reg = 6
B-cells = 11
NK = 14
NK_abs = 0.31""",
                "json": """{
  "patient_id": "P-004",
  "age": 45,
  "sex": "Женский",
  "Общие лейкоциты": 5.8,
  "Гранулоциты": 60,
  "Гранулоциты абс.": 3.1,
  "Лимфоциты, %": 30,
  "Лимфоциты, абс": 1.6,
  "ЛИИ": 1.2,
  "CD3+": 68,
  "Th": 42,
  "CD3+CD4-(Tcyt)": 24,
  "T-reg": 6,
  "B-cells": 11,
  "NK": 14,
  "NK_abs": 0.31
}""",
            },
            "extended": {
                "text": """patient_id = P-004
age = 45
sex = Женский
Общие лейкоциты = 5.8
Гранулоциты = 60
Гранулоциты абс. = 3.1
Лимфоциты, % = 30
Лимфоциты, абс = 1.6
ЛИИ = 1.2
CD3+ = 68
Th = 42
CD3+CD4-(Tcyt) = 24
T-reg = 6
B-cells = 11
NK = 14
NK_abs = 0.31
Моноциты, % = 8
Моноциты, абс = 0.42
CD3+CD45R0- = 31
CD3+CD45R0+ = 37
CD3+CD25+ = 9
CD3+CD45R+CD62L- = 18
CD3+CD45R+CD62L+ = 22
CD3+CD4+CD25+ = 11
T-reg CD45R0+CD62L+ = 4
CD19+CD27+ = 5
CD19+CD23+ = 9
NK 56+16+ = 12
NK 56dim16+ = 10
T-regCD62L = 6
B-cells CD5+27+ = 2""",
                "json": """{
  "patient_id": "P-004",
  "age": 45,
  "sex": "Женский",
  "Общие лейкоциты": 5.8,
  "Гранулоциты": 60,
  "Гранулоциты абс.": 3.1,
  "Лимфоциты, %": 30,
  "Лимфоциты, абс": 1.6,
  "ЛИИ": 1.2,
  "CD3+": 68,
  "Th": 42,
  "CD3+CD4-(Tcyt)": 24,
  "T-reg": 6,
  "B-cells": 11,
  "NK": 14,
  "NK_abs": 0.31,
  "Моноциты, %": 8,
  "Моноциты, абс": 0.42,
  "CD3+CD45R0-": 31,
  "CD3+CD45R0+": 37,
  "CD3+CD25+": 9,
  "CD3+CD45R+CD62L-": 18,
  "CD3+CD45R+CD62L+": 22,
  "CD3+CD4+CD25+": 11,
  "T-reg CD45R0+CD62L+": 4,
  "CD19+CD27+": 5,
  "CD19+CD23+": 9,
  "NK 56+16+": 12,
  "NK 56dim16+": 10,
  "T-regCD62L": 6,
  "B-cells CD5+27+": 2
}""",
            },
        },
    }

    default_examples = {
        "minimal": {
            "text": "patient_id = P-001\nage = 50\nsex = Мужской",
            "json": '{\n  "patient_id": "P-001",\n  "age": 50,\n  "sex": "Мужской"\n}',
        },
        "extended": {
            "text": "patient_id = P-001\nage = 50\nsex = Мужской",
            "json": '{\n  "patient_id": "P-001",\n  "age": 50,\n  "sex": "Мужской"\n}',
        },
    }

    cohort_examples = examples.get(cohort, default_examples)
    return cohort_examples.get(variant, cohort_examples["minimal"])


def get_minimal_profile_status(cohort: str, patient_data: Dict[str, Any]) -> dict[str, Any]:
    profile = get_minimal_profile(cohort)
    field_specs = [field for section in profile["sections"] for field in section["fields"] if field["name"] != "patient_id"]
    filled = [field["label"] for field in field_specs if patient_data.get(field["name"]) not in (None, "")]
    missing = [field["label"] for field in field_specs if patient_data.get(field["name"]) in (None, "")]
    total = len(field_specs)

    return {
        "title": profile["title"],
        "description": profile["description"],
        "total_fields": total,
        "filled_fields": filled,
        "missing_fields": missing,
        "completion": len(filled) / total if total else 0.0,
    }


def _number_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _clear_field(patient_data: Dict[str, Any], name: str) -> None:
    patient_data.pop(name, None)


def _clear_age_values(patient_data: Dict[str, Any], all_features: list[str]) -> None:
    patient_data.pop("age", None)
    for feature in all_features:
        if _is_age_feature(feature):
            patient_data.pop(feature, None)


def _clear_sex_values(patient_data: Dict[str, Any], all_features: list[str]) -> None:
    patient_data.pop("sex", None)
    for feature in all_features:
        if _is_sex_feature(feature):
            patient_data.pop(feature, None)


def _set_age_value(patient_data: Dict[str, Any], all_features: list[str], value: int) -> None:
    patient_data["age"] = value
    for feature in all_features:
        if _is_age_feature(feature):
            patient_data[feature] = value


def _set_sex_value(patient_data: Dict[str, Any], all_features: list[str], value: str, cat_features: Set[str]) -> None:
    patient_data["sex"] = value
    for feature in all_features:
        if _is_sex_feature(feature):
            patient_data[feature] = value if feature not in cat_features else value


def _field_badge(label: str, *, filled: bool, required: bool) -> None:
    status_text = "Заполнено" if filled else ("Нужно заполнить" if required else "Доп. поле")
    bg_color = "#e8f5e9" if filled else ("#fff3cd" if required else "#eef5ff")
    text_color = "#1f6f43" if filled else ("#8a6d3b" if required else "#2a5c8a")
    border_color = "#b7dfc3" if filled else ("#f2d58a" if required else "#b8d4f0")
    st.markdown(
        f"<div style='margin:0.1rem 0 0.35rem 0;padding:0.28rem 0.55rem;border-radius:8px;border:1px solid {border_color};background:{bg_color};color:{text_color};font-size:0.8rem;font-weight:600;'>{status_text}: {label}</div>",
        unsafe_allow_html=True,
    )


def _render_generic_input(name: str, label: str, patient_data: Dict[str, Any], all_features: list[str], cat_features: Set[str], *, key_prefix: str) -> None:
    _field_badge(label, filled=patient_data.get(name) not in (None, ""), required=True)
    if name == "patient_id":
        patient_id_value = st.text_input(
            label,
            value=_display_value(patient_data.get("patient_id")),
            placeholder="P-001",
            key=f"{key_prefix}_{name}",
        )
        if patient_id_value.strip():
            patient_data["patient_id"] = patient_id_value.strip()
        else:
            _clear_field(patient_data, "patient_id")
        return

    if name == "age":
        age_value = st.text_input(
            label,
            value=_display_value(patient_data.get("age")),
            placeholder="62",
            key=f"{key_prefix}_{name}",
        )
        if age_value.strip():
            try:
                _set_age_value(patient_data, all_features, int(float(age_value.strip().replace(',', '.'))))
            except ValueError:
                pass
        else:
            _clear_age_values(patient_data, all_features)
        return

    if name == "sex":
        current_sex = patient_data.get("sex", "")
        sex_options = ["", "Мужской", "Женский"]
        sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
        sex_value = st.selectbox(label, sex_options, index=sex_index, key=f"{key_prefix}_{name}")
        if sex_value:
            _set_sex_value(patient_data, all_features, sex_value, cat_features)
        else:
            _clear_sex_values(patient_data, all_features)
        return

    _render_feature_input(name, label, patient_data, cat_features, key_prefix=key_prefix, show_badge=False)


def _render_feature_input(name: str, label: str, patient_data: Dict[str, Any], cat_features: Set[str], *, key_prefix: str, show_badge: bool = True) -> None:
    if show_badge:
        _field_badge(label, filled=patient_data.get(name) not in (None, ""), required=False)
    feature_lower = label.lower()
    current_value = patient_data.get(name)

    if name in cat_features:
        if _is_sex_feature(name):
            sex_options = ["", "Мужской", "Женский"]
            current_sex = patient_data.get("sex", current_value or "")
            sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
            sex_value = st.selectbox(label, sex_options, index=sex_index, key=f"{key_prefix}_{name}")
            if sex_value:
                patient_data[name] = sex_value
            else:
                _clear_field(patient_data, name)
        else:
            text_value = st.text_input(label, value=_display_value(current_value), key=f"{key_prefix}_{name}")
            if text_value.strip():
                patient_data[name] = text_value.strip()
            else:
                _clear_field(patient_data, name)
        return

    if "sofa" in feature_lower or "глазго" in feature_lower or "glasgow" in feature_lower:
        options = [""] + list(range(0, 25))
        current_option = int(current_value) if current_value not in (None, "") else ""
        selected_value = st.selectbox(label, options, index=options.index(current_option) if current_option in options else 0, key=f"{key_prefix}_{name}")
        if selected_value == "":
            _clear_field(patient_data, name)
        else:
            patient_data[name] = int(selected_value)
    else:
        text_value = st.text_input(
            label,
            value=_display_value(current_value),
            placeholder="Введите значение",
            key=f"{key_prefix}_{name}",
        )
        if text_value.strip():
            try:
                patient_data[name] = float(text_value.strip().replace(',', '.'))
            except ValueError:
                patient_data[name] = text_value.strip()
        else:
            _clear_field(patient_data, name)


def _covered_model_features(field_name: str, all_features: list[str]) -> set[str]:
    if field_name == "age":
        return {feature for feature in all_features if _is_age_feature(feature)}
    if field_name == "sex":
        return {feature for feature in all_features if _is_sex_feature(feature)}
    if field_name == "patient_id":
        return set()
    return {field_name} if field_name in all_features else set()


def generate_dynamic_form(cohort: str, patient_data: Dict[str, Any], form_nonce: int = 0) -> Dict[str, Any]:
    features_info = get_cohort_features(cohort)
    all_features = features_info["features"]
    cat_features = features_info["cat_features"]

    if not all_features:
        st.warning("Не удалось загрузить параметры модели для этой когорты")
        return patient_data

    profile = get_minimal_profile(cohort)

    st.markdown(f"### {profile['title']}")
    st.caption(profile["description"])

    shown_features: set[str] = set()

    for section_idx, section in enumerate(profile["sections"]):
        st.markdown(f"### {section['title']}")
        cols = st.columns(2)
        for idx, field in enumerate(section["fields"]):
            with cols[idx % 2]:
                _render_generic_input(field["name"], field["label"], patient_data, all_features, cat_features, key_prefix=f"form_{form_nonce}_section_{section_idx}")
                shown_features.update(_covered_model_features(field["name"], all_features))

    selectable_features = [feature for feature in all_features if feature not in shown_features]
    if selectable_features:
        with st.expander("Дополнительные параметры"):
            st.info(f"Всего признаков в модели: {len(all_features)}. Основная форма показывает только минимально полезный профиль.")
            recommended_extra = [
                feature for feature in profile.get("recommended_extra_features", []) if feature in selectable_features
            ]
            selected_extra_features = st.multiselect(
                "Добавьте дополнительные признаки в форму",
                options=selectable_features,
                default=recommended_extra if patient_data else [],
                help="Если у вас есть расширенные данные пациента, можно вручную добавить дополнительные признаки из модели.",
                key=f"form_{form_nonce}_extra_features",
            )

            if selected_extra_features:
                extra_cols = st.columns(2)
                for idx, feature in enumerate(selected_extra_features):
                    with extra_cols[idx % 2]:
                        _render_feature_input(feature, feature, patient_data, cat_features, key_prefix=f"form_{form_nonce}_extra")

    return patient_data
