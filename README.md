# RF Fingerprinting / Physical Layer Authentication

**Student:** Filip Kanchev  
**Student ID:** 2185972  
**Course:** Security of Advanced Networking and Services

This repository contains the project files for an RF Fingerprinting / Physical Layer Authentication (PLA) study based on the reference implementation by Alla et al., *Robust Device Authentication in Multi-Node Networks: ML-Assisted Hybrid PLA Exploiting Hardware Impairments*.

The aim of the project is to keep the original PLA assessment logic and add two additional machine-learning models for comparison. The work uses the processed DGT-based RF fingerprint dataset included in the official repository.

## Project focus

The reference work studies whether hardware impairments can create reliable RF fingerprints when combined with Gabor-transform based signal processing and diagonal spectrum reading. The ML stage evaluates which models and feature selections are more robust for recognizing the fingerprints.

This project adds and evaluates:

- Gaussian Naive Bayes
- Extra Trees Classifier

The added models are compared with the original repository results under the same three trial scenarios.

## Files

```text
RF_Fingerprint_Additional_Models.py   Additional model evaluation script
extract_official_results.py           Extracts original repository result summaries
make_summary_figures.py               Generates comparison figures
requirements.txt                      Python dependencies

dataset/processed_fingerprints_data.h5
results/                              CSV and JSON results
figures/                              Generated figures
report/                               Final PDF report
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the added-model experiment

```bash
python RF_Fingerprint_Additional_Models.py --model gaussian_nb extra_trees --feature anova
```

## Recreate summary files and figures

```bash
python extract_official_results.py
python make_summary_figures.py
```

## Dataset note

The repository includes the processed fingerprint dataset used for the submitted experiments. The larger raw signal dataset and noise dataset are not included because of their size.
