from __future__ import annotations
import ast
import json
from pathlib import Path
from typing import Any
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('dataset/processed_fingerprints_data.h5')
OUTPUT = Path('results/repository_compatible_added_models')
FIGURES = Path('figures')
RANDOM_STATE = 42
FEATURE_CANDIDATES = [10, 50, 100, 200, 505]
TRIALS = {
    'trial_1': {'authorized': ['device3', 'device2', 'device12', 'device9'], 'rogue': ['device8', 'device11', 'device5', 'device1', 'device6', 'device10', 'device7', 'device4']},
    'trial_2': {'authorized': ['device10', 'device6', 'device12', 'device3', 'device11', 'device1'], 'rogue': ['device2', 'device5', 'device8', 'device7', 'device4', 'device9']},
    'trial_3': {'authorized': ['device1', 'device10', 'device9', 'device8', 'device12', 'device11', 'device6', 'device7'], 'rogue': ['device4', 'device2', 'device5', 'device3']},
}
NEW_MODELS = ['gaussian_nb', 'extra_trees']


def load_dataset():
    with h5py.File(DATA, 'r') as f:
        X = f['data'][:]
        y = f['labels'][:]
        mapping = ast.literal_eval(f['device_id_mapping'][()].decode())
    return X, y, mapping


def model(name: str):
    if name == 'extra_trees':
        return ExtraTreesClassifier(n_estimators=10, max_depth=None, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    if name == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])
    if name == 'gaussian_nb':
        return GaussianNB()
    raise ValueError(name)


def criteria(y_true, pred, positive: int):
    cm = confusion_matrix(y_true, pred, labels=[positive, 1-positive])
    tdr = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    fdr = cm[1, 0] / cm[1].sum() if cm[1].sum() else 0.0
    return float(tdr), float(fdr)


def closeness(a_tdr, a_fdr, r_tdr, r_fdr):
    return abs(a_tdr-.95) + abs(a_fdr-.05) + abs(r_tdr-.95) + abs(r_fdr-.05)


def evaluate_model_trial(X, y, mapping, trial: str, model_name: str) -> dict[str, Any]:
    auth_names = TRIALS[trial]['authorized']
    rogue_names = TRIALS[trial]['rogue']
    auth_ids = [mapping[d] for d in auth_names]
    rogue_ids = [mapping[d] for d in rogue_names]
    devices = {}
    for target_name in auth_names:
        target_id = mapping[target_name]
        auth_mask = np.isin(y, auth_ids)
        rogue_mask = np.isin(y, rogue_ids)
        X_auth = X[auth_mask]
        y_auth = (y[auth_mask] == target_id).astype(int)
        X_rogue = X[rogue_mask]
        y_rogue = np.zeros(len(X_rogue), dtype=int)
        best = None
        best_score = float('inf')
        accepted = None
        max_k = X_auth.shape[1]
        for k in [value for value in FEATURE_CANDIDATES if value <= max_k]:
            selector = SelectKBest(f_classif, k=k)
            X_auth_s = selector.fit_transform(X_auth, y_auth)
            X_rogue_s = selector.transform(X_rogue)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            for fold, (train, test) in enumerate(cv.split(X_auth_s, y_auth), start=1):
                clf = Pipeline([('scale', StandardScaler()), ('model', model(model_name))])
                clf.fit(X_auth_s[train], y_auth[train])
                auth_pred = clf.predict(X_auth_s[test])
                rogue_pred = clf.predict(X_rogue_s)
                auth_tdr, auth_fdr = criteria(y_auth[test], auth_pred, positive=1)
                rogue_tdr, rogue_fdr = criteria(y_rogue, rogue_pred, positive=0)
                row = {
                    'optimal_n_features': k,
                    'selected_fold': fold,
                    'auth_tvr': auth_tdr,
                    'auth_fvr': auth_fdr,
                    'rogue_tvr': rogue_tdr,
                    'rogue_fvr': rogue_fdr,
                    'adr': (auth_tdr + rogue_tdr) / 2,
                }
                score = closeness(auth_tdr, auth_fdr, rogue_tdr, rogue_fdr)
                if auth_tdr >= .95 and auth_fdr <= .05 and rogue_tdr >= .95 and rogue_fdr <= .05:
                    accepted = row
                    break
                if score < best_score:
                    best_score = score
                    best = row
            if accepted is not None:
                break
        devices[target_name] = accepted if accepted is not None else best
    overall = float(np.mean([record['adr'] for record in devices.values()]))
    return {'trial_name': trial, 'model_type': model_name, 'method': 'anova', 'overall_adr': overall, 'devices': devices,
            'protocol_note': 'ANOVA comparison using the supplied PLA trial structure.'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    X, y, mapping = load_dataset()
    new_rows = []
    for trial in TRIALS:
        for name in NEW_MODELS:
            print('Running', trial, name)
            result = evaluate_model_trial(X, y, mapping, trial, name)
            out_dir = OUTPUT / trial / name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / 'anova_results.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            new_rows.append({'trial': trial, 'model': name, 'model_group': 'added', 'overall_adr': result['overall_adr']})
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(OUTPUT / 'added_models_anova_results.csv', index=False)
    new_summary = new_df.groupby(['model', 'model_group'], as_index=False)['overall_adr'].mean().rename(columns={'overall_adr': 'mean_adr'})
    new_summary.to_csv(OUTPUT / 'added_models_anova_summary.csv', index=False)

    official = pd.read_csv('results/official_supplied_results.csv')
    official = official[official['feature_method'] == 'anova'].copy()
    official['model_group'] = 'official supplied baseline'
    combined = pd.concat([
        official[['trial','model','model_group','overall_adr']],
        new_df[['trial','model','model_group','overall_adr']]
    ], ignore_index=True)
    combined_summary = combined.groupby(['model', 'model_group'], as_index=False)['overall_adr'].mean().rename(columns={'overall_adr': 'mean_adr'}).sort_values('mean_adr', ascending=False)
    combined.to_csv(OUTPUT / 'combined_anova_trial_results.csv', index=False)
    combined_summary.to_csv(OUTPUT / 'combined_anova_summary.csv', index=False)
    ordered = combined_summary.sort_values('mean_adr')
    plt.figure(figsize=(10,6))
    plt.barh(ordered['model'], ordered['mean_adr']*100)
    plt.xlabel('Mean ADR across three trials (%)')
    plt.ylabel('Machine-learning model')
    plt.title('Repository-Compatible ANOVA Comparison: Official and Added Models')
    plt.xlim(max(0, ordered['mean_adr'].min()*100-5), 100)
    plt.tight_layout()
    plt.savefig(FIGURES / 'repository_compatible_anova_comparison.png', dpi=200)
    plt.close()
    print('\nCombined ANOVA ranking:')
    print(combined_summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))

if __name__ == '__main__':
    main()
