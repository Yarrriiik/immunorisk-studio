from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"   # как в train_all.py 
OUT_DIR = PROJECT_ROOT / "templates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_empty_row(features: list[str]) -> dict:
    return {c: None for c in features}

def make_example_row(features: list[str]) -> dict:
    row = {}
    for c in features:
        cl = str(c).lower()
        if cl in ("sex", "пол", "gender"):
            row[c] = "0"
        elif cl in ("age",) or "возраст" in cl:
            row[c] = 50
        else:
            row[c] = 0
    return row

def write_help_md(cohort: str) -> None:
    feat_path = REPORTS / f"features_{cohort}.csv"  # это stats.to_csv(...) из train 
    out_path = OUT_DIR / f"{cohort}_help.md"

    if not feat_path.exists():
        out_path.write_text(
            f"# {cohort}\n\nНет файла {feat_path.name} — сначала запусти обучение, чтобы он появился.\n",
            encoding="utf-8",
        )
        return

    df = pd.read_csv(feat_path)

    # columns как в reports/features_*.csv: feature, dtype, nonnull_count, nonnull_ratio, ..., kept 
    if "kept" in df.columns:
        df = df[df["kept"] == True]

    df = df.sort_values(["nonnull_ratio", "nonnull_count"], ascending=False).head(20)

    lines = []
    lines.append(f"# {cohort}")
    lines.append("")
    lines.append("Топ-20 самых заполненных признаков (nonnull_ratio).")
    lines.append("")
    for _, r in df.iterrows():
        feat = r.get("feature")
        dtype = r.get("dtype")
        nnr = r.get("nonnull_ratio")
        nnc = r.get("nonnull_count")
        lines.append(f"- {feat} — dtype={dtype}, nonnull_ratio={nnr:.3f}, nonnull_count={int(nnc)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

for art_dir in ARTIFACTS.iterdir():
    if not art_dir.is_dir():
        continue
    meta_path = art_dir / "meta.json"
    if not meta_path.exists():
        continue

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cohort = meta["cohort"]
    features = list(meta["features"])  # список фич сохраняется в meta.json 

    # template (1 пустая строка)
    df_template = pd.DataFrame([make_empty_row(features)], columns=features)
    df_template.to_csv(OUT_DIR / f"{cohort}_template.csv", index=False, encoding="utf-8-sig")

    # example (1 строка заглушек)
    df_example = pd.DataFrame([make_example_row(features)], columns=features)
    df_example.to_csv(OUT_DIR / f"{cohort}_example.csv", index=False, encoding="utf-8-sig")

    # help.md
    write_help_md(cohort)

print(f"Wrote templates/help to {OUT_DIR}")
