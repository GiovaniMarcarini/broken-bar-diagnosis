Broken Bar Fault Diagnosis in Three-Phase Induction Motors Using Deep Learning

This repository contains the full pipeline for diagnosing broken rotor bar faults in three-phase squirrel-cage induction motors using deep learning applied to current signals (MCSA).
Three state-of-the-art time-series classification models are evaluated:

LSTM

Temporal Signal Transformer (TST)

InceptionTime

All experiments are based on the official open-access dataset from IEEE DataPort.

Project Workflow

Convert the raw .mat files to .npy (including clean, noisy, and missing-data versions).

Preprocess the signals with sliding windows and class balancing.

Train deep learning models (LSTM, TST, InceptionTime).

Evaluate performance using multiple classification metrics and confusion matrices.

Test model robustness under noise and missing data.

Repository Structure
├── 01_convert_ieee_mat_to_npy.py           # Converts MATLAB files → NumPy (.npy)
│                                           # Generates noisy and missing-data versions
├── 02_split_train_test_sets.py             # Creates overlapping windows and balanced splits
├── 03_load_numpy_datasets.py               # Loads training/validation datasets
│
├── 04_train_lstm_fault_classifier.py       # Trains LSTM model
├── 05_train_tst_fault_classifier.py        # Trains Temporal Signal Transformer model
├── 06_train_inceptiontime_fault_classifier.py # Trains InceptionTime model
│
├── 07_evaluate_noise_missing.py            # Evaluates robustness under noise + missing data
│
└── README.md

Requirements

To install the necessary dependencies:

pip install numpy pandas scipy
pip install matplotlib scikit-learn
pip install torch fastai tsai

Dataset Description

We use the publicly available dataset from IEEE DataPort:

“Experimental Database for Detecting and Diagnosing Rotor Broken Bar in Three-Phase Induction Motors.”

Key features:

1 HP three-phase induction motor (healthy + faulty conditions)

5 rotor conditions:

0 broken bars (healthy)

1, 2, 3, 4 adjacent broken bars

Load levels from 12.5% to 100%

18-second recordings (10 repetitions per condition)

Signals recorded:

Three-phase currents (only phase Ia used in this project)

Accelerometers (not used here)

High sampling frequency suitable for MCSA-based diagnosis

Step-by-Step Pipeline
1. Convert .mat Files to .npy
python 01_convert_ieee_mat_to_npy.py


This script:

Extracts phase-A current (Ia)

Produces clean datasets

Generates AWGN-corrupted datasets with SNR:

30 dB

20 dB

10 dB

0 dB

Generates missing-data datasets with:

10% zeroed samples

30% zeroed samples

50% zeroed samples

2. Split into Training and Test Sets
python 02_split_train_test_sets.py


This script:

Extracts 217-sample windows (~2 cycles at 60 Hz)

Uses sliding windows with stride = 10

Maximizes temporal pattern extraction

Produces 2+ million windows for training

Applies oversampling to balance the dataset

3. Load the Data
python 03_load_numpy_datasets.py


All signals are prepared in the format:

(n_samples, n_timesteps, n_features)


Compatible with the tsai/fastai API.

4. Train the Models

Each script trains a specific architecture:

LSTM
python 04_train_lstm_fault_classifier.py

Temporal Signal Transformer (TST)
python 05_train_tst_fault_classifier.py

InceptionTime
python 06_train_inceptiontime_fault_classifier.py


Training configuration (same for all models):

10 epochs

Learning rate = 1e-3

fit_one_cycle optimization

TSStandardize (batch normalization)

TSClassification (label encoding)

CrossEntropyLoss

This unified setup ensures fair comparison across architectures.

Evaluation Metrics

All evaluations use scikit-learn and include:

Accuracy

Weighted Precision

Weighted Recall

Weighted F1-score

Balanced Accuracy

Cohen’s Kappa

Classification Report (per class)

Normalized Confusion Matrix

Metrics are computed for each model on:

Clean test set

All noise levels (SNR 30/20/10/0 dB)

All missing levels (10/30/50%)

Robustness Testing: Noise and Missing Data

The script below evaluates all models across all perturbations:

python 07_evaluate_noise_missing.py


This script provides:

Predictions for each perturbation

Full metric tables

Confusion matrices (PNG)

Global robustness summary CSV

This analysis assesses the sensitivity of each model to:

Noise (AWGN)

SNR = 30 dB (light noise)

SNR = 20 dB (moderate noise)

SNR = 10 dB (strong noise)

SNR = 0 dB (extreme noise)

Missing Data

10% randomly removed

30% removed

50% removed
