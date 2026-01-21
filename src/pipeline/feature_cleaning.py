# src/pipeline/feature_cleaning.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeaturePolicy:
    min_nonnull_frac: float = 0.20          # минимум 20% заполненности
    min_nonnull_count: int = 10             # но не меньше 10 значений
    drop_constant: bool = True
    bad_name_patterns: tuple[str, ...] = (
        r"^unnamed(:\s*\d+)?$",
        r"^nan(\.\d+)?$",
        r"^\s*$",
    )
    id_name_patterns: tuple[str, ...] = (
        r"^№(\..*)?$",  # №, №.1, №.2 ...
        r"^пор\.\s*№$",  # Пор.№
        r"^№\s*баз\.$",  # № баз.
        r"^patient[_ ]?id$",  # patient_id / patient id
        r"^id$",  # id
    )


def make_unique_columns(cols: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def drop_bad_name_columns(df: pd.DataFrame, policy: FeaturePolicy) -> tuple[pd.DataFrame, set[str]]:
    bad = set()
    for col in df.columns:
        col_norm = str(col).strip().lower()

        # 1) мусорные имена
        for pat in policy.bad_name_patterns:
            if re.match(pat, col_norm):
                bad.add(col)
                break

        # 2) ID-колонки
        if col not in bad:
            for pat in policy.id_name_patterns:
                if re.match(pat, col_norm):
                    bad.add(col)
                    break

    if bad:
        df = df.drop(columns=[c for c in bad if c in df.columns])
    return df, bad



def feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    n = len(X)
    nn = X.notna().sum(axis=0)
    nunique = X.nunique(dropna=True)
    dtypes = X.dtypes.astype(str)
    return pd.DataFrame({
        "feature": X.columns,
        "dtype": [dtypes[c] for c in X.columns],
        "nonnull_count": [int(nn[c]) for c in X.columns],
        "nonnull_ratio": [float(nn[c] / n) if n else 0.0 for c in X.columns],
        "nunique": [int(nunique[c]) for c in X.columns],
    })


def sanitize_for_catboost(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    # pandas.NA -> np.nan (иначе ловили NAType)
    X = X.replace({pd.NA: np.nan})

    # nullable числовые -> float64, чтобы пропуски стали np.nan
    for c in X.columns:
        if c in cat_cols:
            continue
        if str(X[c].dtype) in (
            "Int64", "Int32", "Int16", "Int8",
            "UInt64", "UInt32", "UInt16", "UInt8",
            "Float64", "Float32", "boolean",
        ):
            X[c] = X[c].astype("float64")

    # категориальные: только строки, пропуски -> "NA"
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("NA")

    return X


def apply_feature_policy(
    X: pd.DataFrame,
    cat_cols: list[str],
    policy: FeaturePolicy,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    st = feature_stats(X)
    n = len(X)
    min_cnt = max(policy.min_nonnull_count, int(np.ceil(policy.min_nonnull_frac * n)))

    drop_sparse = set(st.loc[st["nonnull_count"] < min_cnt, "feature"].tolist())
    drop_const = set()
    if policy.drop_constant:
        drop_const = set(st.loc[st["nunique"] <= 1, "feature"].tolist())

    drop = drop_sparse | drop_const
    keep = [c for c in X.columns if c not in drop]

    X2 = X[keep].copy()
    cat2 = [c for c in cat_cols if c in X2.columns]

    st["drop_reason"] = ""
    st.loc[st["feature"].isin(drop_sparse), "drop_reason"] += "sparse;"
    st.loc[st["feature"].isin(drop_const), "drop_reason"] += "const;"
    st["kept"] = ~st["feature"].isin(drop)

    X2 = sanitize_for_catboost(X2, cat2)
    return X2, cat2, st
