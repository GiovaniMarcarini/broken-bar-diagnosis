# Broken Bar Fault Diagnosis in Three-Phase Induction Motors Using Deep Learning

This repository contains the full pipeline for diagnosing broken rotor bar faults in three-phase squirrel-cage induction motors using deep learning and current signals. We evaluate three modern time-series classification models: **LSTM**, **Temporal Signal Transformer (TST)**, and **InceptionTime**.

All experiments are based on an open-access dataset from **IEEE DataPort**.

---

## Project Workflow

1. Convert the raw `.mat` files to `.npy`.  
2. Preprocess the data with windowing and class balancing.  
3. Train and evaluate deep learning models.  
4. Analyze performance using multiple metrics and confusion matrices.  
5. Evaluate robustness under additive noise and missing data.

---

## Repository Structure

```text
├── 01_convert_ieee_mat_to_npy.py                # Converts MATLAB files to NumPy format (clean, noisy, missing)
├── 02_split_train_test_sets.py                  # Creates overlapping windows and train/test splits
├── 03_load_numpy_datasets.py                    # Loads processed data into training format
├── 04_train_lstm_fault_classifier.py            # Trains an LSTM model
├── 05_train_tst_fault_classifier.py             # Trains a TST model
├── 06_train_inceptiontime_fault_classifier.py   # Trains an InceptionTime model
├── 07_evaluate_noise_missing.py                 # Evaluates models under noise and missing-data perturbations
```
## Dataset Description

We use the publicly available dataset from IEEE DataPort:

Experimental Database for Detecting and Diagnosing Rotor Broken Bar in Three-Phase Induction Motors

Key features:

  - 1 HP induction motor (healthy and faulty)

  - Rotor conditions: healthy and multiple levels of broken bars

  - Load levels from 12.5% to 100%

  - Synchronized electrical (current) and mechanical sensors

  - 18-second recordings with high sampling frequency

  - Only phase A current (Ia) is used in this project


## Step-by-Step Pipeline
  1. Convert .mat Files to .npy

    This script:
    -  Converts the original MATLAB files (.mat) into NumPy arrays (.npy).
    -  Focuses on phase A current (Ia).
    -  Optionally generates:
    -  Noisy versions of the signals using AWGN with different SNR levels.
    - Missing-data versions with randomly zeroed samples (simulating data gaps).





