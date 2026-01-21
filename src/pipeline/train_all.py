# src/pipeline/train_all.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.cohorts.targets import COHORT_TARGETS, CohortTarget


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PARQUET = PROJECT_ROOT / "data_parquet"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"


def _is_unnamed(col: str) -> bool:
    return str(col).strip().lower().startswith("unnamed")


def _clean_binary_target(s: pd.Series) -> pd.Series:
    # 0/1, да/нет, жив/умер -> 0/1
    if s.dtype.name in ("string", "object"):
        s2 = s.astype("string").str.strip().str.lower()
        repl = {
            "да": "1",
            "нет": "0",
            "умер": "1",
            "жив": "0",
            "смерть": "1",
            "благоприятный": "0",
            "неблагоприятный": "1",
        }
        s2 = s2.replace(repl)
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _clean_multiclass_target(s: pd.Series) -> pd.Series:
    # для CatBoost multiclass лучше строки/категории
    if s.dtype.name not in ("string", "object"):
        return s.astype("string")
    return s.astype("string").str.strip()


def _select_features(df: pd.DataFrame, cfg: CohortTarget) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = [c for c in cfg.cat_candidates if c in df.columns]
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in cat_cols]

    cols = [c for c in (num_cols + cat_cols) if c != cfg.target_col]
    cols = [c for c in cols if not _is_unnamed(c)]
    cols = list(dict.fromkeys(cols))

    X = df[cols].copy()

    # 1) Убираем pandas.NA -> np.nan (CatBoost иначе валится на NAType)
    X = X.replace({pd.NA: np.nan})

    # 2) Категориальные: только строки, без NAType
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA")

    # 3) Числовые nullable-типы (Int64/Float64/boolean) приводим к обычным float,
    # чтобы пропуски стали np.nan, а не <NA>
    for c in X.columns:
        if c in cat_cols:
            continue
        if str(X[c].dtype) in ("Int64", "Int32", "Int16", "Int8",
                               "UInt64", "UInt32", "UInt16", "UInt8",
                               "Float64", "Float32", "boolean"):
            X[c] = X[c].astype("float64")

    return X, cat_cols




def train_one(cfg: CohortTarget, seed: int = 42, test_size: float = 0.2) -> dict[str, Any]:
    pq_path = DATA_PARQUET / f"{cfg.cohort}.parquet"
    if not pq_path.exists():
        return {"cohort": cfg.cohort, "status": "missing_parquet", "path": str(pq_path)}

    df = pd.read_parquet(pq_path)

    if cfg.target_col not in df.columns:
        return {
            "cohort": cfg.cohort,
            "status": "missing_target_col",
            "target_col": cfg.target_col,
        }

    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if cfg.task == "regression":
        y = pd.to_numeric(df[cfg.target_col], errors="coerce")
    elif cfg.task == "classification":
        y = _clean_binary_target(df[cfg.target_col])
    else:
        y = _clean_multiclass_target(df[cfg.target_col])

    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask]

    if len(df) < cfg.min_rows:
        return {
            "cohort": cfg.cohort,
            "status": "too_few_rows",
            "n_rows_after_dropna": int(len(df)),
            "min_rows": cfg.min_rows,
            "task": cfg.task,
            "target_col": cfg.target_col,
        }

    X, cat_cols = _select_features(df, cfg)

    # stratify only for classification-like tasks
    strat = y if cfg.task in ("classification", "multiclass") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    if cfg.task == "regression":
        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=seed,
            verbose=False,
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        pred = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

    elif cfg.task == "classification":
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, average="binary")),
        }

    else:  # multiclass
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        }

    art_dir = ARTIFACTS / cfg.cohort
    art_dir.mkdir(parents=True, exist_ok=True)

    model_path = art_dir / "model.cbm"
    model.save_model(str(model_path))

    meta = {
        "cohort": cfg.cohort,
        "task": cfg.task,
        "target_col": cfg.target_col,
        "features": X.columns.tolist(),
        "cat_features": cat_cols,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "test_size": test_size,
        "seed": seed,
    }

    with open(art_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out = {"status": "ok", **meta, **metrics}
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    results = []
    for cohort, cfg in COHORT_TARGETS.items():
        res = train_one(cfg)
        results.append(res)
        print(f"[{cohort}] {res['status']}")

    pd.DataFrame(results).to_csv(REPORTS / "train_summary.csv", index=False, encoding="utf-8-sig")
    print("Wrote reports/train_summary.csv")


if __name__ == "__main__":
    main()
