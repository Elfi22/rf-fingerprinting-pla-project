# RF Fingerprinting / Physical Layer Authentication (PLA)

**Student:** Filip Kanchev  
**Student ID:** 2185972  
**Course:** Security of Advanced Networking and Services

## Overview

This repository contains the code, dataset, results, figures, and report for the RF Fingerprinting / Physical Layer Authentication project.

The work uses the PLA framework and processed RF fingerprint features from:

https://github.com/PLA-AP/PLA

The project compares additional machine-learning classifiers for unauthorized-device detection using the processed DGT-based RF fingerprint dataset.

## Files

```text
RF_Fingerprint_Repository_Compatible_Extension.py   Main ANOVA comparison script
RF_Fingerprint_Extended.py                          Additional rogue-held-out test
extract_official_results.py                         Parses supplied baseline results
plot_official_noise_results.py                      Plots supplied noise summaries
requirements.txt                                    Python dependencies

dataset/processed_fingerprints_data.h5              Processed RF fingerprint data
results/                                            CSV and JSON experiment results
figures/                                            Generated plots
report/                                             Final PDF report
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the main experiment

```bash
python RF_Fingerprint_Repository_Compatible_Extension.py
```

## Run the additional security test

```bash
python RF_Fingerprint_Extended.py
```

## Dataset note

The repository includes the processed fingerprint dataset used in the report. The larger raw dataset and noise dataset are not included because of their size.
