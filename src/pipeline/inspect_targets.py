'''
Вывод таргетов для ihd, il2
'''

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cohorts.targets import COHORT_TARGETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PARQUET = PROJECT_ROOT / "data_parquet"


def main() -> None:
    for cohort, cfg in COHORT_TARGETS.items():
        task = cfg.get("task")
        target_col = cfg.get("target_col")

        if not task or not target_col:
            print(f"\n[{cohort}] skip (no target yet)")
            continue

        pq_path = DATA_PARQUET / f"{cohort}.parquet"
        if not pq_path.exists():
            print(f"\n[{cohort}] missing parquet: {pq_path}")
            continue

        df = pd.read_parquet(pq_path)

        print(f"\n[{cohort}] task={task} rows={len(df)} cols={df.shape[1]}")
        if target_col not in df.columns:
            print(f"  !! target_col not found: {target_col}")
            # на всякий случай покажем похожие
            similar = [c for c in df.columns if target_col.lower()[:10] in str(c).lower()]
            print("  similar:", similar[:20])
            continue

        s = df[target_col]
        print(f"  target_col: {target_col}")
        print(f"  dtype: {s.dtype}")
        print(f"  missing: {s.isna().sum()} / {len(s)}")

        # value counts with NaN
        vc = s.value_counts(dropna=False).head(30)
        print("  value_counts (top 30):")
        for k, v in vc.items():
            print(f"    {repr(k)}: {int(v)}")


if __name__ == "__main__":
    main()
