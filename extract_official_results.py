import json

from pathlib import Path


import pandas as pd


TRIALS = ["trial_1", "trial_2", "trial_3"]

MODELS = ["random_forest", "logistic_regression", "gradient_boosting", "svc", "knn", "xgb"]

METHODS = ["anova", "mutual_info", "pca", "rfe"]


def main():

    rows = []

    for trial in TRIALS:

        for model in MODELS:

            for method in METHODS:

                path = Path("results/official_supplied") / trial / model / f"{method}_results.json"

                if not path.exists():

                    continue

                with open(path, "r", encoding="utf-8") as f:

                    data = json.load(f)

                rows.append({

                    "trial": trial,

                    "model": model,

                    "feature_method": method,

                    "overall_adr": data.get("overall_adr"),

                })

    df = pd.DataFrame(rows)

    Path("results").mkdir(exist_ok=True)

    df.to_csv("results/official_supplied_results.csv", index=False)

    summary = df.groupby(["model", "feature_method"], as_index=False)["overall_adr"].mean()

    summary = summary.rename(columns={"overall_adr": "mean_adr"}).sort_values("mean_adr", ascending=False)

    summary.to_csv("results/official_supplied_summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":

    main()
