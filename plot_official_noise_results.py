from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

rows = []
for p in Path('original_reference_results/official_results_noise').glob('*_dB/*.json'):
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    for i, adr in enumerate(data['avg_detection_rates'], start=1):
        rows.append({'snr_db': data['snr'], 'scenario': f'Scenario {i}', 'adr_percent': adr})
df = pd.DataFrame(rows).sort_values(['scenario', 'snr_db'])
df.to_csv('results/official_supplied_noise_summary.csv', index=False)
plt.figure(figsize=(8, 5))
for scenario, group in df.groupby('scenario'):
    plt.plot(group['snr_db'], group['adr_percent'], marker='o', label=scenario)
plt.xlabel('SNR (dB)')
plt.ylabel('Average Detection Rate (%)')
plt.title('Official PLA Supplied Results Under AWGN Noise')
plt.legend()
plt.tight_layout()
plt.savefig('figures/official_supplied_noise_adr.png', dpi=200)
plt.close()
print(df.to_string(index=False))
