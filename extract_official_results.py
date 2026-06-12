from pathlib import Path
import json
import pandas as pd

root = Path('original_reference_results/official_results')
rows = []
for path in root.glob('trial_*/*/*_results.json'):
    with open(path, encoding='utf-8') as f:
        record = json.load(f)
    rows.append({
        'trial': record['trial_name'],
        'model': record['model_type'],
        'feature_method': record['method'],
        'overall_adr': record['overall_adr'],
        'source_file': str(path),
    })
result = pd.DataFrame(rows).sort_values(['feature_method', 'model', 'trial'])
result.to_csv('results/official_supplied_results.csv', index=False)
anova = result[result['feature_method'] == 'anova']
anova_summary = (anova.groupby('model', as_index=False)['overall_adr']
                 .mean().rename(columns={'overall_adr':'mean_adr_across_trials'})
                 .sort_values('mean_adr_across_trials', ascending=False))
anova_summary.to_csv('results/official_supplied_anova_summary.csv', index=False)
print(anova_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
