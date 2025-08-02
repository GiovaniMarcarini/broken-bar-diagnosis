# Broken Bar Fault Diagnosis in Three-Phase Induction Motors Using Deep Learning

This repository contains the full pipeline for diagnosing broken rotor bar faults in three-phase squirrel-cage induction motors using deep learning and current signals. We evaluate three modern time-series classification models: **LSTM**, **Temporal Signal Transformer (TST)**, and **InceptionTime**.

All experiments are based on an open-access dataset from **IEEE DataPort**.

---

## Project Workflow

1. Convert the raw `.mat` files to `.npy`.
2. Preprocess the data with windowing and class balancing.
3. Train and evaluate deep learning models.
4. Analyze performance using multiple metrics and confusion matrices.

---

## Repository Structure

├── 01_convert_ieee_mat_to_npy.py # Converts MATLAB files to NumPy format
├── 02_split_train_test_sets.py # Creates overlapping windows and splits data
├── 03_load_numpy_datasets.py # Loads processed data into training format
├── 04_train_lstm_fault_classifier.py # Trains an LSTM model
├── 05_train_tst_fault_classifier.py # Trains a TST model
├── 06_train_inceptiontime_fault_classifier.py# Trains an InceptionTime model

yaml
Copiar
Editar

---

## Requirements

Install the required dependencies:

```bash
Essential libraries:

numpy, pandas, scipy

matplotlib, scikit-learn

torch, fastai, timeseriesAI

Dataset Description
We use the publicly available dataset from IEEE DataPort:

Experimental Database for Detecting and Diagnosing Rotor Broken Bar in Three-Phase Induction Motors

Key features:

1 HP induction motor (healthy and faulty)

3 rotor conditions: healthy, 2 broken bars, 4 broken bars

Load levels from 12.5% to 100%

Synchronized electrical (current) and mechanical sensors

Current sampled at high frequency across 18-second recordings

Step-by-Step Pipeline
1. Convert .mat Files to .npy
bash
Copiar
Editar
python 01_convert_ieee_mat_to_npy.py
This script converts raw MATLAB files into .npy arrays, focusing on phase A current (Ia).

2. Split into Training and Test Sets
bash
Copiar
Editar
python 02_split_train_test_sets.py
Extracts segments of 217 samples (~2 cycles at 60 Hz)

Sliding window with stride=10

Overlapping segments increase sample count and pattern generalization

Oversampling is applied to balance underrepresented classes

3. Load the Data
bash
Copiar
Editar
python 03_load_numpy_datasets.py
This script loads datasets in shape (n_samples, n_timesteps, n_features) for model training.

4. Train the Models
Choose one of the following scripts:

LSTM:
bash
Copiar
Editar
python 04_train_lstm_fault_classifier.py
Temporal Signal Transformer (TST):
bash
Copiar
Editar
python 05_train_tst_fault_classifier.py
InceptionTime:
bash
Copiar
Editar
python 06_train_inceptiontime_fault_classifier.py
All use the timeseriesAI library built on fastai

Batch standardization (TSStandardize) applied

Trained for 10 epochs using fit_one_cycle

Fixed learning rate: 1e-3

Evaluation Metrics
The following metrics are computed using scikit-learn:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Balanced Accuracy

Cohen’s Kappa

Classification Report

Normalized Confusion Matrix
