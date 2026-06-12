from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
FIXED_ANOVA_FEATURES = 50
DEFAULT_DATA = Path("dataset/processed_fingerprints_data.h5")

TRIALS: dict[str, dict[str, list[str]]] = {
    "trial_1": {
        "authorized": ["device3", "device2", "device12", "device9"],
        "rogue": ["device8", "device11", "device5", "device1", "device6", "device10", "device7", "device4"],
    },
    "trial_2": {
        "authorized": ["device10", "device6", "device12", "device3", "device11", "device1"],
        "rogue": ["device2", "device5", "device8", "device7", "device4", "device9"],
    },
    "trial_3": {
        "authorized": ["device1", "device10", "device9", "device8", "device12", "device11", "device6", "device7"],
        "rogue": ["device4", "device2", "device5", "device3"],
    },
}

ORIGINAL_MODELS = [
    "random_forest",
    "gradient_boosting",
    "svc",
    "knn",
    "xgb",
    "logistic_regression",
]
ADDED_MODELS = ["extra_trees", "hist_gradient_boosting", "mlp", "lda"]
ALL_MODELS = ORIGINAL_MODELS + ADDED_MODELS


def load_h5_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found: {path}")
    with h5py.File(path, "r") as handle:
        X = handle["data"][:]
        y = handle["labels"][:]
        mapping = ast.literal_eval(handle["device_id_mapping"][()].decode())
    return X, y, mapping


def build_estimator(model_name: str) -> BaseEstimator:
    models: dict[str, BaseEstimator] = {
        "random_forest": RandomForestClassifier(
            n_estimators=75, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=75, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
        "svc": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "xgb": XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", max_depth=8,
            n_estimators=50, learning_rate=0.1, subsample=0.7,
            colsample_bytree=0.7, random_state=RANDOM_STATE, n_jobs=1,
            verbosity=0,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=75, max_depth=None, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=75, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(50,), activation="relu", solver="adam",
            alpha=0.0001, max_iter=500, early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models)}")
    return models[model_name]


def safe_feature_grid(n_samples: int, n_features: int) -> list[int]:
    candidates = [10, 30, 50, 100]
    max_k = min(n_features, max(5, n_samples - 2))
    grid = [k for k in candidates if k <= max_k]
    return grid or [min(5, n_features)]


def tdr_and_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    tdr = tp / (tp + fn) if tp + fn else np.nan
    fdr = fp / (fp + tn) if fp + tn else np.nan
    return float(tdr), float(fdr)


def evaluate_target_device(
    X: np.ndarray,
    y: np.ndarray,
    mapping: dict[str, int],
    trial_name: str,
    target_device: str,
    model_name: str,
    output_models_dir: Path,
) -> list[dict[str, Any]]:
    authorized_names = TRIALS[trial_name]["authorized"]
    rogue_names = TRIALS[trial_name]["rogue"]
    authorized_ids = [mapping[name] for name in authorized_names]
    rogue_ids = [mapping[name] for name in rogue_names]
    target_id = mapping[target_device]

    auth_mask = np.isin(y, authorized_ids)
    rogue_mask = np.isin(y, rogue_ids)
    X_auth = X[auth_mask]
    labels_auth = (y[auth_mask] == target_id).astype(int)
    X_rogue = X[rogue_mask]
    y_rogue = np.zeros(X_rogue.shape[0], dtype=int)

    counts = np.bincount(labels_auth)
    n_splits = min(3, int(counts.min()))
    if n_splits < 2:
        raise ValueError(f"Not enough samples for {target_device} in {trial_name}.")

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_auth, labels_auth), start=1):
        X_train, X_test = X_auth[train_idx], X_auth[test_idx]
        y_train, y_test = labels_auth[train_idx], labels_auth[test_idx]
        k_features = min(FIXED_ANOVA_FEATURES, X.shape[1])
        best_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_classif, k=k_features)),
            ("scale", StandardScaler()),
            ("model", build_estimator(model_name)),
        ])
        best_model.fit(X_train, y_train)

        auth_pred = best_model.predict(X_test)
        rogue_pred = best_model.predict(X_rogue)
        auth_tdr, auth_fdr = tdr_and_fdr(y_test, auth_pred)
        rogue_rejection_rate = float(np.mean(rogue_pred == y_rogue))
        adr = float((auth_tdr + rogue_rejection_rate) / 2)

        model_path = output_models_dir / trial_name / model_name / target_device
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path / f"fold_{fold}.joblib")

        rows.append({
            "trial": trial_name,
            "target_device": target_device,
            "model": model_name,
            "model_group": "added" if model_name in ADDED_MODELS else "existing",
            "feature_method": "ANOVA fitted inside training folds",
            "outer_fold": fold,
            "selected_features": int(k_features),
            "inner_cv_balanced_accuracy": None,
            "auth_tdr": auth_tdr,
            "auth_fdr": auth_fdr,
            "rogue_rejection_rate": rogue_rejection_rate,
            "adr": adr,
            "n_auth_train": int(len(train_idx)),
            "n_auth_test": int(len(test_idx)),
            "n_rogue_external_test": int(X_rogue.shape[0]),
        })
    return rows


def summarise_and_plot(fold_df: pd.DataFrame, results_dir: Path, figures_dir: Path) -> None:
    model_trial = (
        fold_df.groupby(["trial", "model", "model_group"], as_index=False)
        .agg(
            auth_tdr=("auth_tdr", "mean"),
            auth_fdr=("auth_fdr", "mean"),
            rogue_rejection_rate=("rogue_rejection_rate", "mean"),
            adr=("adr", "mean"),
            adr_std=("adr", "std"),
            selected_features_median=("selected_features", "median"),
            folds=("adr", "count"),
        )
    )
    overall = (
        model_trial.groupby(["model", "model_group"], as_index=False)
        .agg(
            auth_tdr=("auth_tdr", "mean"),
            auth_fdr=("auth_fdr", "mean"),
            rogue_rejection_rate=("rogue_rejection_rate", "mean"),
            mean_adr=("adr", "mean"),
            adr_across_trials_std=("adr", "std"),
        )
        .sort_values("mean_adr", ascending=False)
    )
    model_trial.to_csv(results_dir / "model_trial_summary.csv", index=False)
    overall.to_csv(results_dir / "model_overall_summary.csv", index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = overall.sort_values("mean_adr", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["model"], ordered["mean_adr"] * 100)
    plt.xlabel("Mean Average Detection Rate across trials (%)")
    plt.ylabel("ML model")
    plt.title("PLA Unauthorized-Device Detection: Existing vs Added Models")
    plt.xlim(max(0, ordered["mean_adr"].min() * 100 - 5), 100)
    plt.tight_layout()
    plt.savefig(figures_dir / "overall_adr_model_comparison.png", dpi=200)
    plt.close()

    pivot = model_trial.pivot(index="model", columns="trial", values="adr") * 100
    pivot = pivot.loc[overall["model"]]
    plt.figure(figsize=(12, 7))
    x = np.arange(len(pivot.index))
    width = 0.25
    for i, trial in enumerate(["trial_1", "trial_2", "trial_3"]):
        plt.bar(x + (i - 1) * width, pivot[trial], width, label=trial)
    plt.xticks(x, pivot.index, rotation=35, ha="right")
    plt.ylabel("Average Detection Rate (%)")
    plt.title("PLA ADR by Trial and Machine-Learning Model")
    plt.ylim(max(0, pivot.min().min() - 5), 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "adr_by_trial_and_model.png", dpi=200)
    plt.close()

    plotted = overall.sort_values("mean_adr", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.scatter(plotted["auth_fdr"] * 100, plotted["rogue_rejection_rate"] * 100)
    for _, row in plotted.iterrows():
        plt.annotate(row["model"], (row["auth_fdr"] * 100, row["rogue_rejection_rate"] * 100), fontsize=8)
    plt.axhline(95, linestyle="--", linewidth=1, label="TDR target = 95%")
    plt.axvline(5, linestyle="--", linewidth=1, label="FDR limit = 5%")
    plt.xlabel("Authorized-device False Detection/Acceptance Rate (%)")
    plt.ylabel("Rogue Rejection Rate (%)")
    plt.title("Security Criteria View for Extended PLA Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "security_criteria_scatter.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLA RF fingerprint benchmark extension.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--trials", nargs="+", choices=list(TRIALS), default=list(TRIALS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    models_dir = args.output / "saved_models"

    X, y, mapping = load_h5_dataset(args.data)
    rows: list[dict[str, Any]] = []
    for trial in args.trials:
        for model in args.models:
            print(f"Running {trial} / {model}")
            for target in TRIALS[trial]["authorized"]:
                rows.extend(evaluate_target_device(X, y, mapping, trial, target, model, models_dir))

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(args.output / "device_fold_results.csv", index=False)
    summarise_and_plot(fold_df, args.output, args.figures)

    metadata = {
        "dataset": str(args.data),
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "devices": int(len(mapping)),
        "feature_selection": "ANOVA SelectKBest; fixed k=50 selected before testing; selector fitted on authorized training fold only",
        "outer_evaluation": "3-fold stratified cross-validation on authorized samples; rogue devices are external test only",
        "rogue_used_for_selection": False,
        "class_imbalance_control": "Balanced class weighting/priors for executed binary authentication models",
        "random_state": RANDOM_STATE,
        "models": args.models,
        "trials": args.trials,
        "added_models": [m for m in args.models if m in ADDED_MODELS],
    }
    with open(args.output / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = pd.read_csv(args.output / "model_overall_summary.csv")
    print("\nOverall model ranking by mean ADR:\n")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nOutputs saved in {args.output} and {args.figures}")


if __name__ == "__main__":
    main()
