# Hete-Ensemble: Heterogeneous Ensemble Prediction of Epilepsy Surgery Outcome from Intracranial EEG

This project targets **drug-resistant epilepsy surgery outcome prediction** from multi-center intracranial EEG (iEEG, covering both SEEG and ECoG modalities). It builds a **Heterogeneous Ensemble** composed of **7 feature domains × 5 selectors × 6 classifiers × 6 time windows**, and evaluates its generalization ability under a **Leave-One-Center-Out (LOCO)** cross-center validation protocol.

---

## 1. Background and Objectives

- **Task**: Binary prediction of whether a patient achieves **Engel Class I (seizure-free; positive class)** after surgery; Engel II–IV are treated as the negative class.
- **Data**: 155 patients and 536 seizure samples from 4 independent epilepsy centers (LZUSH, CHFU, HUP, FMRS), covering both SEEG and ECoG modalities.
- **Unit of computation**: **EZ (epileptogenic zone) / NEZ (non-epileptogenic zone)** channel groups; statistics within each group are aggregated into a fixed-dimensional feature vector.

---

## 2. Method Overview

The full pipeline consists of three stages, implemented as three main scripts under `src/Run/`:

```text
Run_Data_Preprocess.py     →  Data preprocessing (time-window cropping + notch + band-splitting filters)
Run_Compute_Feature.py     →  Seven-domain feature extraction
Run_train_voting_center.py →  Heterogeneous expert training, soft-voting aggregation, LOCO evaluation
```

### 2.1 Preprocessing (`Run_Data_Preprocess.py`)

Unified preprocessing pipeline:

1. **Resampling**: performed at data-ingestion time, converting `orig_sfreq` to a common `target_sfreq`.
2. **Time-window extraction**: according to 6 configurations.
3. **Bad-channel removal**.
4. **Notch filtering**: 50 Hz or 60 Hz and their harmonics, depending on the center.
5. **Band-splitting filtering**: three filter streams are produced in parallel.

Six time-window configurations:

| Configuration | Duration | Cropping Rule | Directory Name |
|---------------|----------|---------------|----------------|
| start | 10 / 20 / 30 s | Take backward from seizure onset | `10start` / `20start` / `30start` |
| mid   | 10 / 20 / 30 s | Centered at the seizure midpoint | `10mid` / `20mid` / `30mid` |

Three filter streams:

- `filter_01`: 0.5 – 125 Hz band-pass (used for DWT / CWT / neural fragility).
- `filter_02`: 80 – 500 Hz band-pass (used for HFO detection).
- `filter_03`: 7-band split (delta / theta / alpha / beta / gamma1 / gamma2 / gamma3, used for Spectral / Connectivity / Graph).

### 2.2 Seven-Domain Features (`Run_Compute_Feature.py`)

Taking EZ / NEZ channel groups as the unit of computation, statistics within each group are aggregated into a fixed-dimensional vector:

| Feature Domain | Dimension | Main Methods |
|----------------|-----------|--------------|
| Spectral | 364 | 7 bands × 10 statistics + inter-band ratios |
| DWT | 126 | Daubechies-4 wavelet, 6-level decomposition × 9 statistics |
| CWT | 398 | Morlet wavelet, 8 scales × amplitude / phase statistics |
| Functional Connectivity | 210 | Coherence / PLV / wPLI, 7 bands |
| Graph Theory | 126 | Weighted strength / betweenness centrality / clustering coefficient |
| HFOs | 100 | Hilbert-envelope detection of Ripple and Fast Ripple |
| Neural Fragility | 390 | Linear state-transition model, 15 parameter groups, three-tier features |

Under each time-window directory, `feature/` contains 7 CSVs: `spectral_feature.csv`, `dwt_feature.csv`, `cwt_feature.csv`, `connectivity_feature.csv`, `graph_feature.csv`, `hfo_feature.csv`, `fragility_feature.csv`.

### 2.3 Heterogeneous Ensemble (`Run_train_voting_center.py`)

Each expert is a 4-tuple **(time window, feature domain, feature selector, classifier)**.

**5 feature selectors** (`src/selecor/`):

| Selector | File | Description |
|----------|------|-------------|
| ANOVA F | `f_classif.py` | F-test |
| MI | `MI.py` | Mutual information |
| L1 | `L1.py` | L1-regularized sparse selection |
| Boruta | `BR.py` | Boruta all-relevant feature selection |
| mRMR | `mRMR.py` | Minimum redundancy maximum relevance |

> Except for Boruta, all selectors reduce the feature set to **40 dimensions**.

**6 classifiers** (`src/classifier/`):

| Classifier | File |
|------------|------|
| LDA | `LDA.py` |
| LR1 (L1 logistic regression) | `LR1.py` |
| LR2 (L2 logistic regression) | `LR2.py` |
| MLP | `mlp.py` |
| XGBoost | `XGB.py` |
| KNN | `KNN.py` |

**Fixed combination of 15 experts**: manually selected by a coverage rule that guarantees: at least one expert per feature domain, full coverage of all 5 selectors and all 6 classifiers, full coverage of all 6 time windows, and that any two experts differ in at least one dimension. This rule is independent of any performance metric, avoiding data leakage.

**Aggregation**: soft voting — arithmetic mean of expert probabilities, thresholded with a global value **τ = 0.56** to obtain binary predictions.

### 2.4 Validation Protocol

**Leave-One-Center-Out (LOCO)**: each fold holds out one entire clinical center as the test set, using the remaining three centers for training; 4 folds in total. Feature selection, model training, hyperparameters, and expert-pool construction never touch the test center. Final metrics are aggregated over **9 random seeds**.

---

## 3. Repository Structure

```text
Hete-Ensemble/
├── data/
│   ├── data.csv                         # Patient metadata (EZ/NEZ channels, labels, center, sampling rate, etc.)
│   ├── iEEG/                            # Raw iEEG data (.fif)
│   ├── Example/                         # Preprocessing and feature intermediates for the example data
│   │   └── {10,20,30}{start,mid}/
│   │       ├── crop/  notch/  filter_01/  filter_02/  filter_03/  feature/
│   └── {10,20,30}{start,mid}/feature/   # Full-set seven-domain feature CSVs
├── src/
│   ├── Run/                             # Three main entry scripts
│   │   ├── Run_Data_Preprocess.py
│   │   ├── Run_Compute_Feature.py
│   │   └── Run_train_voting_center.py
│   ├── preprocess/                      # Preprocessing
│   │   ├── crop.py / filter.py / utils.py
│   ├── feature/                         # Seven-domain feature extraction
│   │   ├── spectral_feature.py / dwt_feature.py / cwt_feature.py
│   │   ├── connectivity_feature.py / graph_feature.py
│   │   ├── hfo_feature.py / HFOdetection_Hilbert.py
│   │   └── fragility_feature.py / neural_fragility.py
│   ├── selecor/                         # Feature selectors
│   │   ├── f_classif.py / MI.py / L1.py / BR.py / mRMR.py
│   ├── classifier/                      # Classifiers
│   │   ├── LDA.py / LR1.py / LR2.py / mlp.py / XGB.py / KNN.py
│   └── train/                           # Training and evaluation
│       ├── load_data.py / split.py / expert_probs.py
│       ├── ensemble_eval.py / expert_eval.py / ablation_eval.py
│       ├── threshold_scan.py / eval_utils.py
└── README.md
```

Note: the `Example` folder only contains `10start`; the other windows can be produced by running the scripts.

---

## 4. Data Format

`data/data.csv` holds patient-level metadata. Key fields:

| Field | Description |
|-------|-------------|
| `Dataset` | Clinical center (LZUSH / CHFU / HUP / FMRS / Example) |
| `patient_name` | Patient identifier |
| `sample_name` | Seizure-sample identifier (.fif filename without extension) |
| `orig_sfreq` / `target_sfreq` | Original sampling rate / resampling target rate |
| `EZ` | List of epileptogenic-zone channel indices |
| `NEZ` | List of non-epileptogenic-zone channel indices |
| `isCured` | Label: True = Engel I (positive), False = Engel II–IV |
| `Implant` | Implant modality (SEEG / ECoG) |
| `start_time` / `end_time` | Seizure onset / offset times |

Raw iEEG is stored as `.fif` files at `data/iEEG/{sample_name}.fif`.

---

## 5. Quick Start

### 5.1 Environment

- Python 3.8+
- Core libraries: `numpy`, `pandas`, `scipy`, `mne`, `scikit-learn`, `xgboost`, `boruta`, `tqdm`, `matplotlib`

### 5.2 Running Steps

From the project root, run the three scripts in order:

**Step 1 — Data preprocessing**

```bash
python -m src.Run.Run_Data_Preprocess
```

Performs time-window cropping, notch filtering, and band-splitting filtering; writes results to `data/Example/` (example data).

**Step 2 — Seven-domain feature extraction**

```bash
python -m src.Run.Run_Compute_Feature
```

Computes seven-domain features for all 6 time windows and writes them to `data/Example/{window}/feature/`.

**Step 3 — Heterogeneous ensemble training and evaluation**

```bash
python -m src.Run.Run_train_voting_center
```

This performs, in order:

1. `train_center_voting` — trains the 15 experts and saves each expert's predicted probabilities per LOCO fold (`results/Expert_prob/seed_{seed}/{expert}/{center}.csv`).
2. `evaluate_ensemble` — ensemble-level metrics; writes `ensemble_metrics.csv` and ROC / PR curves (`figures/`).
3. `evaluate_experts` — per-expert metrics; writes `expert_metrics.csv`.
4. `evaluate_ablation` — leave-one-expert-out ablation; writes `leave_one_expert_out.csv`.
5. `scan_thresholds` — threshold sweep (0.30 – 0.80, step 0.05); writes `threshold_scan.csv` and marks the target threshold 0.56.

> To change the expert combination, threshold, or random seeds, simply edit `experts_config`, `threshold`, or `base_seed` (or `seed_list`) in `Run_train_voting_center.py`.

### 5.3 Mapping Between Data and Scripts

To make validation and reproduction straightforward, the repository treats three inputs differently:

- **`data/Example/`**: provides a single example `.fif` (`Example.fif`) corresponding to the sample with `Dataset == "Example"` in `data.csv`. It is meant to validate **Step 1 (preprocessing)** and **Step 2 (feature extraction)**.
- **`data/{10,20,30}{start,mid}/feature/`**: **full seven-domain feature tables** are already provided (4 centers × 6 time windows). There is **no need to re-run Step 1 or Step 2** — you can directly execute **Step 3 (ensemble training and evaluation)**.
- **Before running Step 3, remove the rows with `Dataset == "Example"` from `data/data.csv`**; otherwise the example sample would act as an extra center in the Leave-One-Center-Out split and contaminate the evaluation.

Choose the path you need:

| Goal | Scripts to Run |
|------|----------------|
| Validate the preprocessing / feature-extraction pipeline | `Run_Data_Preprocess.py` → `Run_Compute_Feature.py` |
| Reproduce the ensemble results directly | Remove Example rows from `data.csv` → `Run_train_voting_center.py` |

---

## 6. Evaluation Metrics

The following 11 metrics are used uniformly (see `src/train/eval_utils.py`):

`Accuracy`, `ROC_AUC`, `F1`, `Kappa`, `MCC`, `Precision`, `Recall`, `Specificity`, `NPV`, `AUC_PR`, `Balanced_Accuracy`

Here `Balanced_Accuracy` is defined as `(Recall + Specificity) / 2`. All metrics are aggregated over 9 random seeds × 4 test centers and reported as **mean ± standard deviation**.

---

## 7. Key Design Principles

- **No leakage**: under the LOCO split, feature selection, model training, threshold determination, and expert-pool construction all take place on training centers only; the test center is never visible.
- **Coverage over performance**: the 15 experts are chosen solely by the "full coverage of 7 domains / 5 selectors / 6 classifiers / 6 time windows" rule, not by any performance metric.
- **Cross-center generalization**: holding out an entire center emulates real-world cross-center deployment.
- **Reusable probability cache**: training only saves per-expert, per-fold, per-seed probabilities; subsequent threshold tuning, ablation, or subgroup analyses require no retraining.
