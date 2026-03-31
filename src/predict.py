from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.pipeline.feature_cleaning import make_unique_columns, sanitize_for_catboost  # то же, что в train 


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class Artifacts:
    cohort: str
    task: str
    target_col: str
    features: list[str]
    cat_features: list[str]
    model: Any
    metrics: dict[str, Any]


def load_artifacts(cohort: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Artifacts:
    art_dir = artifacts_dir / cohort  # как в train_all.py 
    with open(art_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(art_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    task = meta["task"]
    model_path = art_dir / "model.cbm"  # как в train_all.py 

    if task == "regression":
        model = CatBoostRegressor()
    else:
        model = CatBoostClassifier()

    model.load_model(str(model_path))
    return Artifacts(
        cohort=meta["cohort"],
        task=task,
        target_col=meta["target_col"],
        features=list(meta["features"]),
        cat_features=list(meta["cat_features"]),
        model=model,
        metrics=metrics,
    )


def prepare_features(df_raw: pd.DataFrame, meta: Artifacts) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = make_unique_columns(df.columns)  # как в train 

    # добавить недостающие колонки
    for c in meta.features:
        if c not in df.columns:
            df[c] = np.nan

    X = df[meta.features].copy()
    return sanitize_for_catboost(X, meta.cat_features)


def predict_df(cohort: str, df_raw: pd.DataFrame, artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    art = load_artifacts(cohort, artifacts_dir=artifacts_dir)
    X = prepare_features(df_raw, art)

    if art.task == "regression":
        y_pred = art.model.predict(X)
        return {
            "cohort": cohort,
            "task": art.task,
            "pred": np.asarray(y_pred).reshape(-1).tolist(),
        }

    if art.task == "classification":
        proba = art.model.predict_proba(X)[:, 1]
        proba = np.asarray(proba).reshape(-1)

        thr = float(art.metrics.get("best_thr", 0.5))  # best_thr пишется в metrics.json 
        pred = (proba >= thr).astype(int)

        return {
            "cohort": cohort,
            "task": art.task,
            "best_thr": thr,
            "proba": proba.tolist(),
            "pred": pred.tolist(),
        }

    # multiclass
    proba = art.model.predict_proba(X)
    proba = np.asarray(proba)

    # метки классов CatBoost хранит внутри модели (обычно строки)
    try:
        class_labels = list(art.model.classes_)
    except Exception:
        class_labels = [str(i) for i in range(proba.shape[1])]

    pred_idx = np.argmax(proba, axis=1)
    pred_label = [class_labels[i] for i in pred_idx]

    # top-3
    top3 = []
    for row in proba:
        idx = np.argsort(row)[::-1][:3]
        top3.append([(class_labels[i], float(row[i])) for i in idx])

    return {
        "cohort": cohort,
        "task": art.task,
        "pred": pred_label,
        "proba": proba.tolist(),
        "top3": top3,
    }


def missing_columns_for_cohort(cohort: str, df_raw: pd.DataFrame, artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    art = load_artifacts(cohort, artifacts_dir=artifacts_dir)
    cols = set(make_unique_columns(df_raw.columns))  # как в train 
    return [c for c in art.features if c not in cols]
