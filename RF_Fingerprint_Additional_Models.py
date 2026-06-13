import argparse

import ast

import json

from pathlib import Path


import h5py

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import GaussianNB


DATA_PATH = Path("dataset/processed_fingerprints_data.h5")

OUTPUT_DIR = Path("results/additional_models")

RANDOM_STATE = 42


TRIALS = {

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


MODEL_CONFIGS = {

    "gaussian_nb": {},

    "extra_trees": {

        "n_estimators": 10,

        "max_depth": 8,

        "random_state": RANDOM_STATE,

        "n_jobs": 1,

        "class_weight": "balanced",

    },

}


FEATURE_METHODS = ["anova", "mutual_info"]

FEATURE_COUNTS = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 505]


def load_dataset(path=DATA_PATH):

    with h5py.File(path, "r") as f:

        X = f["data"][:]

        y = f["labels"][:]

        mapping = ast.literal_eval(f["device_id_mapping"][()].decode())

    return X, y, mapping


def get_model(model_type):

    if model_type == "gaussian_nb":

        return GaussianNB()

    if model_type == "extra_trees":

        return ExtraTreesClassifier(**MODEL_CONFIGS[model_type])

    raise ValueError(f"Unknown model type: {model_type}")


def ranked_feature_indices(X, y, method):

    if method == "anova":

        selector = SelectKBest(f_classif, k="all")

        selector.fit(X, y)

        scores = np.nan_to_num(selector.scores_, nan=0.0)

    elif method == "mutual_info":

        scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

        scores = np.nan_to_num(scores, nan=0.0)

    else:

        raise ValueError(f"Unsupported feature method: {method}")

    return np.argsort(scores)[::-1]


def evaluate_criteria(y_true, y_pred, positive_label):

    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, 1 - positive_label])

    tdr = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0

    fdr = cm[1, 0] / cm[1].sum() if cm[1].sum() else 0.0

    return float(tdr), float(fdr)


def closeness_score(auth_tdr, auth_fdr, rogue_tdr, rogue_fdr):

    return abs(auth_tdr - 0.95) + abs(auth_fdr - 0.05) + abs(rogue_tdr - 0.95) + abs(rogue_fdr - 0.05)


def evaluate_device(device_rf_data, authorized_devices, rogue_devices, target_device, model_type, feature_method):

    X_auth, y_auth, X_rogue = [], [], []


    for device_name, rows in device_rf_data.items():

        if device_name == target_device:

            X_auth.extend(rows)

            y_auth.extend([1] * len(rows))

        elif device_name in authorized_devices:

            X_auth.extend(rows)

            y_auth.extend([0] * len(rows))

        elif device_name in rogue_devices:

            X_rogue.extend(rows)


    X_auth = np.asarray(X_auth)

    y_auth = np.asarray(y_auth)

    X_rogue = np.asarray(X_rogue)

    y_rogue = np.zeros(len(X_rogue), dtype=int)


    ranking = ranked_feature_indices(X_auth, y_auth, feature_method)

    best_record = None

    best_score = float("inf")


    for n_features in [k for k in FEATURE_COUNTS if k <= X_auth.shape[1]]:

        selected_indices = ranking[:n_features]

        X_auth_selected = X_auth[:, selected_indices]

        X_rogue_selected = X_rogue[:, selected_indices]


        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for train_index, test_index in skf.split(X_auth_selected, y_auth):

            X_train, X_test = X_auth_selected[train_index], X_auth_selected[test_index]

            y_train, y_test = y_auth[train_index], y_auth[test_index]


            model = get_model(model_type)

            model.fit(X_train, y_train)


            auth_pred = model.predict(X_test)

            rogue_pred = model.predict(X_rogue_selected)


            auth_tdr, auth_fdr = evaluate_criteria(y_test, auth_pred, positive_label=1)

            rogue_tdr, rogue_fdr = evaluate_criteria(y_rogue, rogue_pred, positive_label=0)

            adr = (auth_tdr + rogue_tdr) / 2

            score = closeness_score(auth_tdr, auth_fdr, rogue_tdr, rogue_fdr)


            record = {

                "optimal_n_features": int(n_features),

                "auth_tvr": auth_tdr,

                "auth_fvr": auth_fdr,

                "rogue_tvr": rogue_tdr,

                "rogue_fvr": rogue_fdr,

                "adr": adr,

                "selected_features": selected_indices.tolist(),

            }


            if score < best_score:

                best_score = score

                best_record = record


            if auth_tdr >= 0.95 and auth_fdr <= 0.05 and rogue_tdr >= 0.95 and rogue_fdr <= 0.05:

                return record


    return best_record


def run_experiment(trials, models, feature_methods):

    X, y, mapping = load_dataset()

    device_rf_data = {name: X[y == idx] for name, idx in mapping.items()}

    detailed_rows = []

    trial_rows = []


    for trial_name in trials:

        authorized = TRIALS[trial_name]["authorized"]

        rogue = TRIALS[trial_name]["rogue"]


        for model_type in models:

            for feature_method in feature_methods:

                devices = {}

                for target_device in authorized:

                    print(f"{trial_name} | {model_type} | {feature_method} | {target_device}")

                    record = evaluate_device(

                        device_rf_data, authorized, rogue, target_device, model_type, feature_method

                    )

                    devices[target_device] = record

                    detailed_rows.append({

                        "trial": trial_name,

                        "model": model_type,

                        "feature_method": feature_method,

                        "target_device": target_device,

                        "optimal_n_features": record["optimal_n_features"],

                        "auth_tdr": record["auth_tvr"],

                        "auth_fdr": record["auth_fvr"],

                        "rogue_tdr": record["rogue_tvr"],

                        "rogue_fdr": record["rogue_fvr"],

                        "adr": record["adr"],

                    })


                overall_adr = float(np.mean([v["adr"] for v in devices.values()]))

                trial_rows.append({

                    "trial": trial_name,

                    "model": model_type,

                    "feature_method": feature_method,

                    "overall_adr": overall_adr,

                })


                out_dir = OUTPUT_DIR / trial_name / model_type

                out_dir.mkdir(parents=True, exist_ok=True)

                with open(out_dir / f"{feature_method}_results.json", "w", encoding="utf-8") as f:

                    json.dump({

                        "trial_name": trial_name,

                        "model_type": model_type,

                        "method": feature_method,

                        "overall_adr": overall_adr,

                        "authorized_devices": authorized,

                        "rogue_devices": rogue,

                        "devices": devices,

                    }, f, indent=2)


    detailed = pd.DataFrame(detailed_rows)

    trials_df = pd.DataFrame(trial_rows)

    summary = trials_df.groupby(["model", "feature_method"], as_index=False)["overall_adr"].mean()

    summary = summary.rename(columns={"overall_adr": "mean_adr"}).sort_values("mean_adr", ascending=False)


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed.to_csv(OUTPUT_DIR / "device_level_results.csv", index=False)

    trials_df.to_csv(OUTPUT_DIR / "trial_results.csv", index=False)

    summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)

    return detailed, trials_df, summary


def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate additional ML models using the PLA assessment logic.")

    parser.add_argument("--trial", nargs="+", default=list(TRIALS.keys()), choices=list(TRIALS.keys()))

    parser.add_argument("--model", nargs="+", default=list(MODEL_CONFIGS.keys()), choices=list(MODEL_CONFIGS.keys()))

    parser.add_argument("--feature", nargs="+", default=FEATURE_METHODS, choices=FEATURE_METHODS)

    return parser.parse_args()


def main():

    args = parse_args()

    _, _, summary = run_experiment(args.trial, args.model, args.feature)

    print("\nSummary")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":

    main()
