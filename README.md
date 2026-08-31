# BUESA MATLAB code

MATLAB implementation of the analysis workflow used for **Bayesian uncertainty-aware EEG signature analysis (BUESA)** with longitudinal two-channel EEG.

The repository is organized into four conceptual stages and six executable scripts. Signature discovery and fixed-signature validation are intentionally separated so that the stochastic CaBESS search is not confused with the prespecified seven-feature signature used for the manuscript validation.

## Repository workflow

| Order | Script | Purpose |
|---|---|---|
| 1 | `construct_longitudinal_labels_01.m` | Match EEG examinations to longitudinal diagnoses and construct strict progression labels. |
| 2 | `extract_two_channel_eeg_features_02.m` | Preprocess two-channel EEG, reject invalid segments, extract segment descriptors, and aggregate them into 64 file-level EEG features. |
| 3 | `compare_ml_and_bayesian_models_03.m` | Exploratory comparison of preprocessing, Top-K feature sets, conventional ML classifiers, and Bayesian logistic regression. |
| 4 | `run_cabess_signature_discovery_04.m` | Run Class-aware Bayesian Evolutionary Signature Selection (CaBESS) and export candidate signatures from the current search run. |
| 5 | `validate_fixed_cabess_signature_05.m` | Evaluate the fixed seven-feature manuscript signature using repeated subject-level five-fold cross-validation. |
| 6 | `run_bopa_analysis_06.m` | Run Bayesian Output-Preserving Attribution (BOPA) for posterior contribution and local output-stability analysis. |

The four high-level blocks are therefore:

1. **Longitudinal label construction + two-channel EEG preprocessing/feature extraction**: scripts 01-02
2. **ML/Bayesian benchmark**: script 03
3. **CaBESS discovery + fixed-signature validation**: scripts 04-05
4. **BOPA interpretation**: script 06

## Setup

1. Copy `configs/config_paths_template.m` to `configs/config_paths_local.m`.
2. Edit only `config_paths_local.m` so that it points to the private local data directories.
3. Run the scripts in numerical order.

`configs/config_paths_local.m`, private data folders, and generated outputs are excluded by `.gitignore`.

See [`docs/INPUT_SCHEMA.md`](docs/INPUT_SCHEMA.md) for the expected input structure.

## Primary longitudinal task

The primary binary task is constructed from the simplified longitudinal subtype:

- **Non-decline (0)**: `Maintain_CN`, `Maintain_SCD`
- **Decline to dementia (1)**: `Decline_MCI_to_DEM`, `Decline_Normal_to_DEM`

For compound declining trajectories, subtype simplification follows the final analysis priority:

1. `CN_to_MCI` / `SCD_to_MCI`
2. `CN_to_DEM` / `SCD_to_DEM`
3. `MCI_to_DEM`

Therefore, a compound `CN/SCD -> MCI -> DEM` trajectory is assigned to `Decline_to_MCI` at the simplification stage and is not part of the primary decline-to-dementia task.

The fixed validation script contains an audit for the manuscript cohort counts of 325 non-decline and 45 decline records. Modify or disable these expected counts when adapting the code to another dataset.

## Two-channel EEG preprocessing and features

`02_extract_two_channel_eeg_features.m` reproduces the main signal-processing settings used in the study:

- sampling frequency: 250 Hz
- channels: 1 and 2
- missing samples: linear interpolation with nearest endpoint filling
- median centering and detrending
- fourth-order Butterworth band-pass filter: 1-45 Hz, zero-phase `filtfilt`
- robust z-normalization using median and scaled MAD (`1.4826 * MAD`)
- clipping to `[-8, 8]`
- 10-s segments with 50% overlap
- maximum 40 accepted segments per EEG file
- segment rejection using amplitude, variance, derivative, and high-amplitude-ratio criteria
- Welch PSD using a 2-s Hamming window with 50% overlap and a 0.5-Hz frequency grid

Six descriptors are extracted per channel: theta relative power, alpha relative power, beta relative power, theta/alpha ratio, median frequency, and SEF95. Inter-channel Pearson correlation plus three across-channel summary descriptors yield 16 segment-level descriptors. Mean, standard deviation, median, and IQR aggregation across segments yields **64 file-level features**.

## Exploratory benchmark

`03_compare_ml_and_bayesian_models.m` exposes the four exploratory preprocessing configurations retained in the manuscript/supplement:

- Z-score
- winsorization + Z-score
- signed-log + Z-score
- winsorization + signed-log + Z-score

The benchmark compares:

- Bayesian logistic regression with Laplace posterior approximation
- logistic regression
- linear discriminant analysis
- RBF-SVM
- bagged trees

All imputation, preprocessing parameters, Top-K feature ranking, class weights, and training-derived thresholds are fitted using the training fold only. Records sharing the same `ObjectID` are assigned to the same fold.

`Benchmark_Summary.xlsx` is generated from the successful held-out fold evaluations produced by the exploratory script. The repeat-level manuscript tables were obtained by first aggregating folds within each repeat and then summarizing the 30 repeated runs. Small numerical differences between these two summaries therefore reflect the aggregation level rather than a different classifier.

## CaBESS discovery

`04_run_cabess_signature_discovery.m` implements the class-aware evolutionary search with:

- decline-specialist and non-decline-specialist populations
- class-aware specialist crossover
- Bayesian uncertainty-guided mutation
- feature-count and redundancy penalties
- inverse-frequency training weights
- subject-level cross-validation for candidate evaluation

Search-stage CV values are **search diagnostics**, not a fully nested estimate of generalization performance.

### Discovery rerun versus the fixed manuscript signature

The CaBESS search is stochastic in principle, although fixed random seeds are provided to improve reproducibility. Exact candidate feature lists can still differ when the software environment, implementation details, candidate-selection rule, or search configuration changes.

The output `ClassAwareGA_FinalSignatures.xlsx` reports the highest-fitness decline specialist, non-decline specialist, and balanced child identified in the **current rerun**. This balanced-child feature set is not assumed to be identical to the seven-feature signature reported in the manuscript.

The script additionally exports `BalancedChild_CandidateRanking.xlsx`, which ranks unique balanced-child candidates from the current run by balanced accuracy, with AUC, decline recall, fitness, and feature count used as secondary diagnostics. This file is intended to make the candidate-selection stage transparent.

During manuscript development, the final fixed signature was archived from the discovery-stage candidate pool under the selection label `TopChild_ByBalancedAccuracy_Rank2`. That signature was then frozen before the repeated validation analysis. Scripts 05 and 06 therefore use the **manuscript reference signature**, rather than automatically replacing it with the best child from a new discovery rerun.

The manuscript seven-feature signature is:

1. `ch1_theta_rel_mean`
2. `ch1_theta_rel_iqr`
3. `ch1_alpha_rel_mean`
4. `ch1_beta_rel_mean`
5. `ch1_theta_alpha_ratio_med`
6. `ch2_theta_alpha_ratio_std`
7. `ch1_sef95_std`

This distinction is intentional:

- **Script 04** repeats the CaBESS search and exports the candidate solutions obtained in that run.
- **Script 05** evaluates the prespecified seven-feature signature reported in the manuscript.
- **Script 06** performs BOPA using the same fixed manuscript signature.

Accordingly, a new run of script 04 may produce a different compact feature list while still showing similar class-balanced performance. Such variation should not be interpreted as a change to the manuscript validation target.

## Fixed-signature validation

`05_validate_fixed_cabess_signature.m` uses:

- 30 repeated subject-level five-fold CV runs
- no random majority-class downsampling
- inverse-frequency class weights estimated from each training fold
- training-fold-only imputation/transformation/scaling
- Bayesian logistic regression with Gaussian priors, MAP estimation, and Laplace posterior approximation
- classification threshold `0.5`

The fixed seven-feature manuscript signature and the all-feature Bayesian baseline are compared on identical repeated subject-level splits.

The validation code is the primary script for reproducing the performance comparison reported for the final signature. Because the complete CaBESS search is not repeated inside every outer fold, these results should not be interpreted as fully nested feature-selection estimates.

## BOPA

`06_run_bopa_analysis.m` fits an explanation model using the eligible records and draws posterior coefficient samples for visualization. This model is **not** used to estimate the repeated-CV predictive performance reported by script 05.

BOPA provides:

- posterior logit-level feature contributions
- posterior contribution uncertainty
- one-feature perturbation curves
- posterior agreement with the baseline class
- output-preserving ranges
- representative confident, boundary, and high-uncertainty cases

The default perturbation grid is `[-2.5, 2.5]` standardized units with 101 points, restricted to the empirical 1st-99th percentile feature range. The minimum posterior-agreement threshold is 0.80.

BOPA is a **local, model-dependent sensitivity analysis**. Its output-preserving ranges are not causal effects, clinical intervention ranges, or multivariate counterfactual solutions.

## Reproducibility interpretation

The cleaned repository has been checked against the archived analysis workflow at the level of:

- primary task counts
- overall benchmark behavior
- final fixed-signature validation performance
- qualitative BOPA contribution and perturbation patterns

Minor changes in posterior samples, discovery-stage feature lists, or individual CaBESS candidates can occur without materially changing the reproduced performance pattern.

For manuscript reproduction, the important distinction is that **discovery variability is separated from fixed-signature evaluation**. Do not overwrite the manuscript reference signature in scripts 05 or 06 simply because a new script-04 rerun finds another high-performing child.

## MATLAB requirements

The workflow uses:

- MATLAB
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Optimization Toolbox is recommended for the exploratory Bayesian benchmark (`fminunc`); the benchmark contains a fallback optimization path when unavailable.

## Data availability and privacy

Participant-level GARD data and raw EEG recordings are not included in this repository. The code documents the analytical procedure for use with the corresponding authorized data.

Do not commit:

- raw participant EEG
- clinical spreadsheets containing participant-level information
- local absolute file paths
- generated private result folders containing identifiers

The repository should contain only code, documentation, configuration templates, and non-identifying synthetic examples.

## Suggested repository contents

```text
BUESA-MATLAB/
├─ README.md
├─ .gitignore
├─ 01_construct_longitudinal_labels.m
├─ 02_extract_two_channel_eeg_features.m
├─ 03_compare_ml_and_bayesian_models.m
├─ 04_run_cabess_signature_discovery.m
├─ 05_validate_fixed_cabess_signature.m
├─ 06_run_bopa_analysis.m
├─ configs/
│  └─ config_paths_template.m
├─ docs/
│  └─ INPUT_SCHEMA.md
└─ examples/
   └─ README.md
```

## Before creating a paper release

Before tagging a public release, verify that:

- the primary task contains the intended 325/45 records in the authorized manuscript dataset
- the extracted feature table contains the expected 64 EEG features
- scripts 05 and 06 contain the exact seven-feature manuscript reference signature
- repeated subject-level validation remains consistent with the archived manuscript results
- BOPA representative cases show the expected qualitative behavior
- no participant-level GARD data or private local paths are present
- the repository URL has been inserted into the final manuscript only after the public repository is ready

Add a repository license only after confirming the permitted license with the laboratory and data provider.
