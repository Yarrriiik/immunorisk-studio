'''
Запуск через: python -m src.predict_cli --cohort ihd --input templates/ihd_template.csv
'''

import argparse
import pandas as pd

from src.predict import predict_df, missing_columns_for_cohort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--input", required=True)  # csv/xlsx
    args = ap.parse_args()

    if args.input.lower().endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    miss = missing_columns_for_cohort(args.cohort, df)
    if miss:
        print(f"Missing columns ({len(miss)}): {miss[:30]}{'...' if len(miss) > 30 else ''}")

    out = predict_df(args.cohort, df)
    print(out)

if __name__ == "__main__":
    main()
