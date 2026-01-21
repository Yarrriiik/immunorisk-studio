from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from datetime import date, datetime

import numpy as np
import pandas as pd

# src/pipeline/convert_all.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PARQUET = PROJECT_ROOT / "data_parquet"
ARTIFACTS = PROJECT_ROOT / "artifacts"

COHORT_FILES = {
    "il2_postcovid": "Baza-s-kontrolem-vmeste-1-IL-2-i-postkovid.xlsx",
    "ihd": "Baza-s-kontrolem-vmeste-2-Ishemicheschkaia-bolezn-serdtsa.xlsx",
    "hepatitis_c": "Baza-s-kontrolem-vmeste-3-Gepatit-S.xlsx",
    "graves": "Baza-s-kontrolem-vmeste-4-Bolezn-Greivsa.xlsx",
    "sepsis": "Baza-s-kontrolem-vmeste-5-Sepsis.xlsx",
    "peritonitis": "Baza-s-kontrolem-vmeste-6-Peritonit.xlsx",
    "peritonitis_legend": "Baza-s-kontrolem-vmeste-7-Legenda-Peritonit.xlsx",
}


def pick_main_sheet(xlsx_path: Path) -> str:
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    for s in sheets:
        if s.strip().lower() != "лист1":
            return s
    return sheets[0]


def normalize_colname(c: Any) -> str:
    if c is None:
        return ""
    if not isinstance(c, str):
        c = str(c)
    c = c.replace("\u00a0", " ")
    c = re.sub(r"\s+", " ", c)
    return c.strip()


def clean_cell(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip().replace("\u00a0", " ")
        if s in {"", "-", "—"}:
            return np.nan
        return s.replace(",", ".")
    return x


_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\d*\.?\d+)$")


def maybe_to_numeric(series: pd.Series, min_share: float = 0.6) -> pd.Series:
    if series.dtype != object:
        return series

    s = series.map(clean_cell)
    as_str = s.dropna().astype(str)
    if len(as_str) == 0:
        return series

    share_numeric = float(as_str.map(lambda v: bool(_NUM_RE.match(v))).mean())
    if share_numeric >= min_share:
        return pd.to_numeric(s, errors="coerce")
    return s


def parse_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    date_cols = [c for c in df.columns if "дата" in c.lower() or c.lower() == "date"]
    parsed = []
    for c in date_cols:
        before_na = float(df[c].isna().mean())
        tmp = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        after_na = float(tmp.isna().mean())
        if after_na <= min(0.95, before_na + 0.30):
            df[c] = tmp
            parsed.append(c)
    return df, parsed


def find_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def fix_mixed_object_columns(df: pd.DataFrame, dt_min_share: float = 0.6, num_min_share: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    Make every object column parquet-safe:
    - if it's mostly datetime-like -> convert to datetime64[ns]
    - else if it's mostly numeric-like -> convert to numeric
    - else -> convert to pandas string dtype (so ints won't remain as ints inside object)
    """
    fixed_cols = []

    def is_dt(v: Any) -> bool:
        return isinstance(v, (datetime, date, pd.Timestamp, np.datetime64))

    def norm_scalar(v: Any) -> Any:
        if v is None:
            return pd.NA
        if isinstance(v, float) and np.isnan(v):
            return pd.NA
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="replace")
            except Exception:
                return str(v)
        if isinstance(v, (datetime, date, pd.Timestamp, np.datetime64)):
            # keep as-is for datetime detection branch; for string branch will be isoformat
            return v
        return v

    for c in df.columns:
        if df[c].dtype != object:
            continue

        s0 = df[c].map(norm_scalar)
        non_na = s0.dropna()
        if len(non_na) == 0:
            # make empty object column a string column (safe)
            df[c] = s0.astype("string")
            fixed_cols.append(c)
            continue

        # 1) datetime-like?
        dt_share = float(non_na.map(is_dt).mean())
        if dt_share >= dt_min_share:
            df[c] = pd.to_datetime(s0, errors="coerce", dayfirst=True)
            fixed_cols.append(c)
            continue

        # 2) try numeric
        # convert datetimes (if any) to iso strings before numeric attempt
        def to_str_if_dt(v: Any) -> Any:
            if isinstance(v, (datetime, date, pd.Timestamp, np.datetime64)):
                try:
                    return pd.to_datetime(v).isoformat()
                except Exception:
                    return str(v)
            return v

        s1 = s0.map(to_str_if_dt)

        # IMPORTANT: ensure strings have "." not ","
        if s1.dtype == object:
            s1 = s1.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v)

        num = pd.to_numeric(s1, errors="coerce")
        num_share = float(num.notna().mean())

        if num_share >= num_min_share:
            df[c] = num
            fixed_cols.append(c)
        else:
            # 3) fallback to string for any mixed garbage
            df[c] = s1.astype("string")
            fixed_cols.append(c)

    return df, fixed_cols



def convert_one(cohort: str, xlsx_path: Path) -> dict[str, Any]:
    sheet = pick_main_sheet(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [normalize_colname(c) for c in df.columns]

    non_empty_cols = [c for c in df.columns if df[c].notna().any()]
    df = df[non_empty_cols].copy()

    # normalize cells + numeric conversion
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean_cell)
            df[c] = maybe_to_numeric(df[c])

    df, parsed_dates = parse_dates(df)

    # drop ultra-sparse columns (keep if at least 2% filled)
    non_null_share = df.notna().mean()
    keep_cols = non_null_share[non_null_share >= 0.02].index.tolist()
    dropped_sparse = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols].copy()

    # detect common fields + create standardized copies
    patient_id_col = find_first(df, ["№", "Пор.№", "Л №", "ID", "№ баз.", "номер", "patient_id"])
    sex_col = find_first(df, ["Пол", "sex", "пол"])
    age_col = find_first(df, ["Возраст", "age", "возраст"])
    visit_dt_col = find_first(df, ["Дата анализа", "Дата", "analysis_dt", "visit_dt"])

    if patient_id_col is not None:
        df["patient_id"] = df[patient_id_col]
    if sex_col is not None:
        df["sex"] = df[sex_col]
    if age_col is not None:
        df["age"] = df[age_col]
    if visit_dt_col is not None:
        df["visit_dt"] = df[visit_dt_col]

    DATA_PARQUET.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PARQUET / f"{cohort}.parquet"
    df, fixed_mixed = fix_mixed_object_columns(df)
    df.to_parquet(out_path, index=False)

    art_dir = ARTIFACTS / cohort
    art_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "cohort": cohort,
        "source_file": xlsx_path.name,
        "sheet": sheet,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "parsed_date_cols": parsed_dates,
        "dropped_sparse_cols": dropped_sparse,
        "detected": {
            "patient_id_col": patient_id_col,
            "sex_col": sex_col,
            "age_col": age_col,
            "visit_dt_col": visit_dt_col,
        },
        "fixed_mixed_object_cols": fixed_mixed,
        "parquet_path": str(out_path.relative_to(PROJECT_ROOT)),
    }
    with open(art_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    return schema


def main() -> None:
    for cohort, fn in COHORT_FILES.items():
        xlsx_path = DATA_RAW / fn
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Missing file for cohort={cohort}: {xlsx_path}")
        schema = convert_one(cohort, xlsx_path)
        print(f"[{cohort}] rows={schema['n_rows']} cols={schema['n_cols']} -> {schema['parquet_path']}")


if __name__ == "__main__":
    main()
