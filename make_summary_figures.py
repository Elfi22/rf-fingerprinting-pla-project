from pathlib import Path


import matplotlib.pyplot as plt

import pandas as pd


FIG = Path("figures")

FIG.mkdir(exist_ok=True)


                             

official = pd.read_csv("results/official_supplied_summary.csv")

added = pd.read_csv("results/additional_models/summary_results.csv")

added["source"] = "added model"

official["source"] = "original repository result"

combined = pd.concat([official, added], ignore_index=True)

combined.to_csv("results/combined_model_feature_summary.csv", index=False)


anova = combined[combined["feature_method"] == "anova"].copy()

anova = anova.sort_values("mean_adr")

plt.figure(figsize=(9, 5))

plt.barh(anova["model"], anova["mean_adr"] * 100)

plt.xlabel("Mean ADR across trials (%)")

plt.ylabel("Model")

plt.title("ANOVA comparison with two additional ML models")

plt.tight_layout()

plt.savefig(FIG / "anova_comparison.png", dpi=200)

plt.close()


                          

trial_info = pd.DataFrame([

    {"trial": "trial_1", "authorized": 4, "rogue": 8},

    {"trial": "trial_2", "authorized": 6, "rogue": 6},

    {"trial": "trial_3", "authorized": 8, "rogue": 4},

])

trial_info.to_csv("results/trial_device_groups.csv", index=False)


                                                   

levels = pd.read_csv("results/additional_models/device_level_results.csv")

plt.figure(figsize=(9, 5))

for model in levels["model"].unique():

    subset = levels[levels["model"] == model]

    plt.scatter(subset["optimal_n_features"], subset["adr"] * 100, label=model)

plt.xlabel("Selected number of features")

plt.ylabel("ADR for target device (%)")

plt.title("Device-level selected feature counts for added models")

plt.legend()

plt.tight_layout()

plt.savefig(FIG / "device_level_features.png", dpi=200)

plt.close()
