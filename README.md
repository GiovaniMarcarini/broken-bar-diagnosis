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
    -  Missing-data versions with randomly zeroed samples (simulating data gaps).

  2. Split into Training and Test Sets
    -  Extracts fixed-length segments from the current signals
      -  Typical example: segments of 217 samples (~2 cycles at 60 Hz)
    -  Uses a sliding window with stride = 10
    -  Increases the number of available samples while preserving temporal structure
    -  Applies oversampling to balance underrepresented classes
    -  Saves the resulting arrays in a format compatible with the deep learning models

  3. Load the Data
    -  Loads the preprocessed .npy files (clean and, optionally, corrupted)
    -  Organizes the data into the format:
        (n_samples, n_timesteps, n_features)

  4. Train the Models
    -  All training scripts use:
    -  The timeseriesAI library built on top of fastai
    -  Batch standardization with TSStandardize
    -  TSClassification for label handling
    -  fit_one_cycle as the training policy
    -  10 epochs of training
    -  Fixed maximum learning rate: 1e-3
     
  A unified training configuration (same learning rate, number of epochs, and preprocessing) is used for all three models to ensure a fair and reproducible comparison.

## Evaluation Metrics

The following metrics are computed using scikit-learn:
  -  Accuracy
  -  Precision (weighted)
  -  Recall (weighted)
  -  F1-score (weighted)
  -  Balanced Accuracy
  -  Cohen’s Kappa
  -  Classification Report (per class)
  -  Normalized Confusion Matrix
These metrics are used both for the clean test set and for the perturbed test sets (with noise and missing data).

## Robustness Evaluation: Noise and Missing Data

In addition to the evaluation on clean signals, the models are tested under controlled degradations to simulate realistic industrial scenarios.

1. Additive Noise (AWGN)
  Additive White Gaussian Noise is injected into the test signals at different Signal-to-Noise Ratios (SNR):
    -  SNR = 30 dB → low noise
    -  SNR = 20 dB → moderate noise
    -  SNR = 10 dB → strong noise
    -  SNR = 0 dB → severe noise
  The training data remain clean, and only the test set is corrupted, which allows evaluating the generalization robustness of the classifiers.

2. Missing Data Simulation
  To emulate sensor dropouts or acquisition failures, missing data are simulated by randomly zeroing samples in the test segments:
    -  10% missing → mild corruption
    -  30% missing → moderate corruption
    -  50% missing → severe corruption
  This corresponds to a Missing Completely At Random (MCAR) mechanism and stresses the ability of each model to handle incomplete temporal information.

3. Metrics Under Perturbations
  For each combination of:
    -  Model (LSTM, TST, InceptionTime)
    -  Noise level (SNR)
    -  Missing rate (% of zeroed samples)

the script computes:
  -  Accuracy
  -  Weighted F1-score
  -  Weighted Precision and Recall
  -  Balanced Accuracy
  -  Cohen’s Kappa
  -  Full classification report
  -  Normalized confusion matrix
