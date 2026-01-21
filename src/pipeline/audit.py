from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# src/pipeline/audit.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data_raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

COHORT_FILES = {
    "il2_postcovid": "Baza-s-kontrolem-vmeste-1-IL-2-i-postkovid.xlsx",
    "ihd": "Baza-s-kontrolem-vmeste-2-Ishemicheschkaia-bolezn-serdtsa.xlsx",
    "hepatitis_c": "Baza-s-kontrolem-vmeste-3-Gepatit-S.xlsx",
    "graves": "Baza-s-kontrolem-vmeste-4-Bolezn-Greivsa.xlsx",
    "sepsis": "Baza-s-kontrolem-vmeste-5-Sepsis.xlsx",
    "peritonitis": "Baza-s-kontrolem-vmeste-6-Peritonit.xlsx",
    "peritonitis_legend": "Baza-s-kontrolem-vmeste-7-Legenda-Peritonit.xlsx",
}

KEY_PATTERNS = [
    "пол",
    "возраст",
    "дата",
    "sofa",
    "saps",
    "глазго",
    "статус",
    "исход",
    "диагноз",
    "умер",
    "жив",
    "crp",
    "циток",
    "ил",
    "лейко",
    "тромбо",
    "креат",
    "билиру",
]


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


def audit_one(cohort: str, xlsx_path: Path) -> dict[str, Any]:
    sheet = pick_main_sheet(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [normalize_colname(c) for c in df.columns]

    # drop fully empty columns
    non_empty_cols = [c for c in df.columns if df[c].notna().any()]
    df = df[non_empty_cols]

    nn_ratio = df.notna().mean().sort_values(ascending=False)

    key_cols = []
    for c in df.columns:
        cl = c.lower()
        if any(p in cl for p in KEY_PATTERNS):
            key_cols.append(c)

    date_like = [c for c in df.columns if "дата" in c.lower() or "date" in c.lower()]

    return {
        "cohort": cohort,
        "file": xlsx_path.name,
        "sheet": sheet,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "date_like_cols": "|".join(date_like[:10]),
        "example_key_cols": " | ".join(key_cols[:20]),
        "top_nonnull_cols": " | ".join(nn_ratio.head(10).index.tolist()),
        "top_nonnull_ratio": " | ".join([f"{x:.2f}" for x in nn_ratio.head(10).tolist()]),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cohort, fn in COHORT_FILES.items():
        xlsx_path = DATA_RAW / fn
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Missing file for cohort={cohort}: {xlsx_path}")

        row = audit_one(cohort, xlsx_path)
        rows.append(row)

        # dump full columns list
        sheet = row["sheet"]
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df.columns = [normalize_colname(c) for c in df.columns]
        pd.DataFrame({"column": df.columns}).to_csv(
            REPORTS_DIR / f"{cohort}_columns.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / "audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Wrote: {REPORTS_DIR / 'audit_summary.csv'}")


if __name__ == "__main__":
    main()
