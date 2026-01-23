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
from src.pipeline.feature_cleaning import (
    FeaturePolicy,
    apply_feature_policy,
    drop_bad_name_columns,
    make_unique_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PARQUET = PROJECT_ROOT / "data_parquet"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"


#  helpers 

def _is_unnamed(col: str) -> bool:
    return str(col).strip().lower().startswith("unnamed")


def _safe_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
    stratify: pd.Series | None,
):
    """
    train_test_split, но если stratify невозможен (слишком мало объектов в классе),
    автоматически пробуем без stratify.
    """
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


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
    # 1) Удаляем мусорные имена колонок
    policy = FeaturePolicy(min_nonnull_frac=0.20, min_nonnull_count=10, drop_constant=True)
    df2, _bad = drop_bad_name_columns(df, policy)

    # 2) кандидаты
    cat_cols = [c for c in cfg.cat_candidates if c in df2.columns]
    num_cols = df2.select_dtypes(include=["number", "bool"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in cat_cols]

    cols = [c for c in (num_cols + cat_cols) if c != cfg.target_col]
    cols = [c for c in cols if not _is_unnamed(c)]
    cols = list(dict.fromkeys(cols))

    X = df2[cols].copy()

    # 3) Фильтрация по заполненности/константности + санитайз
    X, cat_cols, stats = apply_feature_policy(X, cat_cols, policy)

    # 4) отчёт по фичам
    REPORTS.mkdir(parents=True, exist_ok=True)
    stats.to_csv(REPORTS / f"features_{cfg.cohort}.csv", index=False, encoding="utf-8-sig")

    return X, cat_cols


#  train 

def train_one(
    cfg: CohortTarget,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    early_stopping_rounds: int = 200,
) -> dict[str, Any]:
    pq_path = DATA_PARQUET / f"{cfg.cohort}.parquet"

    if not pq_path.exists():
        return {"cohort": cfg.cohort, "status": "missing_parquet", "path": str(pq_path)}

    df = pd.read_parquet(pq_path)
    df.columns = make_unique_columns(df.columns)

    if cfg.target_col not in df.columns:
        return {
            "cohort": cfg.cohort,
            "status": "missing_target_col",
            "target_col": cfg.target_col,
            "columns": df.columns.tolist(),
        }

    # drop cols
    for c in cfg.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # target
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
    if X.shape[1] == 0:
        return {
            "cohort": cfg.cohort,
            "status": "no_features_after_policy",
            "n_rows": int(len(df)),
            "task": cfg.task,
            "target_col": cfg.target_col,
        }

    #  splits: train/test then train/val 
    strat = y if cfg.task in ("classification", "multiclass") else None
    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )

    strat2 = y_train if cfg.task in ("classification", "multiclass") else None
    X_tr, X_val, y_tr, y_val = _safe_train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=strat2
    )

    best_iteration = None

    #  models 
    if cfg.task == "regression":
        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=seed,
            verbose=False,
        )
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = getattr(model, "best_iteration_", None)

        pred = np.asarray(model.predict(X_test)).reshape(-1)

        metrics = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

    elif cfg.task == "classification":
        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
        )
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = getattr(model, "best_iteration_", None)

        # --- 1) вероятности на val и test
        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

        proba_val = pd.Series(np.asarray(proba_val).reshape(-1))
        proba_test = pd.Series(np.asarray(proba_test).reshape(-1))

        y_val2 = pd.Series(np.asarray(y_val).reshape(-1))
        y_test2 = pd.Series(np.asarray(y_test).reshape(-1))

        y_val2 = pd.to_numeric(y_val2, errors="coerce")
        y_test2 = pd.to_numeric(y_test2, errors="coerce")

        # --- 2) чистим NaN (на всякий)
        m_val = y_val2.notna()
        m_test = y_test2.notna()

        proba_val = proba_val[m_val]
        y_val2 = y_val2[m_val].astype(int)

        proba_test = proba_test[m_test]
        y_test2 = y_test2[m_test].astype(int)

        # --- 3) подбор порога по F1 на val
        thr_grid = np.linspace(0.05, 0.95, 19)

        best_thr = 0.5
        best_f1 = -1.0

        if y_val2.nunique() >= 2:
            for thr in thr_grid:
                pred_val = (proba_val >= thr).astype(int)
                f1_val = f1_score(y_val2, pred_val, average="binary", zero_division=0)
                if f1_val > best_f1:
                    best_f1 = float(f1_val)
                    best_thr = float(thr)

        # --- 4) финальные предсказания и метрики на test (честно)
        pred = (proba_test >= best_thr).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test2, pred)),
            "f1": float(f1_score(y_test2, pred, average="binary", zero_division=0)),
            "best_thr": float(best_thr),

            # опционально, но полезно для отладки дисбаланса
            "pos_rate_val": float(y_val2.mean()),
            "pos_rate_test": float(y_test2.mean()),
            "pred_pos_rate": float(pred.mean()),
        }




    else:  # multiclass
        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
        )
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = getattr(model, "best_iteration_", None)

        pred = model.predict(X_test, prediction_type="Class")
        pred = np.asarray(pred).reshape(-1).astype(str)
        y_test2 = np.asarray(y_test).reshape(-1).astype(str)

        metrics = {
            "accuracy": float(accuracy_score(y_test2, pred)),
            "f1_macro": float(f1_score(y_test2, pred, average="macro")),
        }

    #  artifacts
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
        "test_size": float(test_size),
        "val_size": float(val_size),
        "seed": int(seed),
        "best_iteration": None if best_iteration is None else int(best_iteration),
    }

    with open(art_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out = {"status": "ok", **meta, **metrics}
    with open(art_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for cohort, cfg in COHORT_TARGETS.items():
        res = train_one(cfg)
        results.append(res)
        print(f"[{cohort}] {res['status']}")

    pd.DataFrame(results).to_csv(REPORTS / "train_summary.csv", index=False, encoding="utf-8-sig")
    print("Wrote reports/train_summary.csv")


if __name__ == "__main__":
    main()
