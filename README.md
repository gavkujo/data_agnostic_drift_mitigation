# Drift Detection & Mitigation Pipeline
### NAISC2026 — Technical Documentation

---

## Overview

This pipeline detects and mitigates distributional shift between a training dataset and a test dataset, then trains a gradient-boosted model on the corrected data. The goal is to maximise test accuracy by ensuring the model sees training data that is as representative of the test distribution as possible.

The pipeline runs in three stages:

```
Train CSV + Test CSV
       │
       ▼
 [1] DETECTION        — detect shift, classify its type, rank affected features
       │
       ▼
 [2] MITIGATION       — apply targeted corrections based on detected shift type
       │
       ▼
 [3] MODEL TRAINING   — train XGBoost/LightGBM on corrected data with sample weights
```

---

## Usage

```bash
python main.py \
  --train_data_filepath path/to/train.csv \
  --test_data_filepath  path/to/test.csv
```

The training CSV must include a `target` column. The test CSV must not.

---

## Stage 1: Detection

Detection runs four methods in sequence. Each one feeds into the next.

### 1.1 Statistical Tests — `statistical_tests.py`

Runs three independent per-feature distributional tests on every numeric column:

| Test | What it measures | Threshold |
|------|-----------------|-----------|
| **Kolmogorov-Smirnov (KS)** | Maximum distance between empirical CDFs | p-value < 0.05 |
| **Population Stability Index (PSI)** | Weighted divergence between binned distributions | PSI > 0.10 |
| **Wasserstein / Earth Mover's Distance (EMD)** | Minimum "work" to reshape one distribution into the other | > 0.2 normalised std units |

A feature is flagged as drifted if **at least 2 of 3 tests agree**. This majority-vote approach reduces both false positives (one test fires on noise) and false negatives (one test misses subtle shift).

A composite drift score per feature (0–1) is also computed as the normalised average of all three statistics. This score is used downstream by feature selection to rank severity.

**Why three tests?** KS is sensitive to location and shape differences but loses power on large samples. PSI is interpretable and industry-standard but requires binning. Wasserstein gives a continuous magnitude of shift that the others don't. Together they catch what each alone misses.

---

### 1.2 Adversarial Validation — `adversarial_validation.py`

Trains a LightGBM binary classifier to distinguish training rows (label=0) from test rows (label=1). If the datasets are similar, the classifier cannot do better than random (AUC ≈ 0.5). If they are very different, it achieves high AUC.

**What it produces:**

- **Domain AUC** — overall shift severity score. AUC > 0.6 = meaningful shift detected.
- **Per-sample importance weights** — each training row gets a weight `w(x) = P(test|x) / P(train|x)`. Rows that look more like test data receive higher weight. This is passed directly to LightGBM's `sample_weight` parameter during model training.
- **Feature importances** — which features the domain classifier relied on most. These are the features whose distributions differ most between train and test.

**Weight averaging (P21):** The domain classifier is trained `n_seeds=5` times with different random seeds. The predicted probabilities are averaged before computing weights. This reduces variance in the weight estimates caused by random initialisation — a single run can produce unstable weights, especially on smaller datasets.

**Shift severity interpretation:**

| Domain AUC | Interpretation |
|------------|---------------|
| 0.50 – 0.60 | No meaningful shift |
| 0.60 – 0.70 | Moderate shift |
| 0.70 – 0.85 | Strong shift |
| 0.85 – 1.00 | Severe shift |

---

### 1.3 Shift Type Classification — `shift_type_classifier.py`

Diagnoses which *type* of shift is present before applying any correction. This step is critical because applying the wrong mitigation (e.g., importance weighting when concept drift is present) can actively harm performance.

Three shift types are tested:

| Type | Definition | Detection signal |
|------|-----------|-----------------|
| **Covariate shift** | P(X) changed, P(Y\|X) stable | Domain AUC > 0.60 |
| **Label shift** | P(Y) changed, P(X\|Y) stable | TV distance between train label dist and proxy-predicted test label dist > 0.05 |
| **Concept drift** | P(Y\|X) changed | Significant drop in model prediction confidence on test vs train |

A proxy model is trained on the training data and used to generate soft predictions on the unlabeled test set. These predictions approximate the test label distribution, enabling label shift estimation without test labels.

The dominant shift type is returned and used to route the mitigation step.

---

### 1.4 Explanation Shift — `explanation_shift.py`

Computes SHAP values for training and test samples using the proxy model, then compares the SHAP value distributions per feature.

**Why this matters:** Standard statistical tests (KS, PSI) compare raw feature values. They will miss cases where two features' marginal distributions are individually stable, but their *joint relationship* (interaction) has changed. SHAP values capture feature interactions because they measure each feature's *contribution to predictions* — not just its raw value. A change in the SHAP distribution of a feature means the model's reliance on that feature has shifted, which directly translates to prediction degradation.

**Two outputs:**

1. **Per-feature SHAP-space KS test** — ranks features by how much their contribution pattern changed. These are features where the model's interpretation has shifted, not just the raw values.
2. **Explanation Shift Detector (ESD)** — a secondary LightGBM classifier trained to distinguish train SHAP vectors from test SHAP vectors. Its AUC quantifies overall explanation shift. Its feature importances provide a second, complementary ranking of which features' behaviour changed most.

> **Note:** Requires `shap` to be installed. If not available, this step is skipped gracefully without breaking the pipeline.

---

## Stage 2: Mitigation

Mitigation is routed based on the shift type diagnosed in Step 1.3:

| Shift type | Techniques applied |
|------------|-------------------|
| `covariate` | Importance weighting → Feature alignment |
| `label` | Label shift correction → Feature alignment |
| `concept` | Feature alignment → Importance weighting |
| `mixed` | Importance weighting → Label shift correction → Feature alignment + decorrelation |
| `none` | No mitigation |

Feature selection (drop high-drift/low-importance features) always runs last regardless of shift type.

---

### 2.1 Importance Weighting — `importance_weighting.py`

Applies the per-sample weights from adversarial validation to the training dataframe, adding a `__sample_weight__` column that XGBoost/LightGBM reads during training.

Training rows that look similar to the test distribution receive high weight. Rows that look unlike anything in the test set receive low weight. The model effectively focuses its learning on the part of the training distribution that is relevant at test time.

**Doubly-robust correction (P16 — Kato et al. 2023):** A known failure mode of plain importance weighting is that if the density ratio estimator (the domain classifier) is inaccurate, the resulting weights are biased and can hurt more than they help. The doubly-robust correction addresses this by combining the density ratio estimate with a separately cross-fitted regression function:

```
w_DR(x, y) = w(x) - correction_term(w(x), y, mu(x))
```

where `mu(x)` is the cross-fitted prediction of y. The corrected estimator remains consistent if *either* the density ratio OR the regression function is correctly specified — not both. This acts as a safety net when the domain classifier is imperfect.

**Weight clipping:** Extreme weights are clipped at the 99th percentile before normalisation to prevent a small number of training rows from dominating the weighted loss.

---

### 2.2 Feature Alignment — `feature_alignment.py`

For features flagged as drifted, aligns the training distribution to match the test distribution via **quantile normalisation**.

The process per feature:
1. Fit a quantile transformer on test values (the target distribution)
2. Fit a quantile transformer on train values (the source distribution)
3. Map: train values → uniform space → inverse CDF of test → aligned values

This eliminates location, scale, and shape differences in the marginal distribution of each drifted feature. Unlike simple standardisation (which only corrects mean and variance), quantile normalisation corrects the full distributional shape including skew and multimodality.

**Optional decorrelation:** When `mixed` shift is detected, an additional step scales each feature's standard deviation in the training set to match the test set's standard deviation. This addresses inter-feature correlation drift — a source of instability in importance weighting identified by Xu et al. (P19).

---

### 2.3 Feature Selection — `feature_selection.py`

Conservatively removes features that are both highly drifted AND have near-zero predictive importance. This is intentionally a strict threshold — both conditions must hold simultaneously.

**What is NOT done:** IRM-style causal feature filtering. Nastl & Hardt (NeurIPS 2024, P26) tested causal feature selection across 16 tabular tasks and found that without exception, using all features outperformed restricting to causal features both in-distribution and out-of-distribution. Dropping features based on causal or spurious-correlation assumptions reliably degrades tabular performance.

**What IS done:**
- **Always drop:** features with zero variance in the test set (they carry no signal at test time)
- **Conditionally drop:** features where drift score > 0.6 AND feature importance is in the bottom 10% of the proxy model's importance ranking

Features that drift but remain predictive are kept and handled by alignment and reweighting instead.

---

### 2.4 Label Shift Correction — `label_shift_correction.py`

When label shift is detected as the dominant shift type, estimates the target label distribution from unlabeled test data and reweights training samples accordingly.

**ELSA — Efficient Label Shift Adaptation (P22 — Tian et al., ICML 2023):**

Frames label shift correction as a semiparametric moment-matching problem. Rather than inverting the full confusion matrix (as BBSE does), ELSA solves a linear system:

```
M^T · q = mu_test
```

where:
- `M[k, j]` = average predicted probability of class j given true class k (estimated on training data via cross-validation)
- `mu_test` = average soft predictions on the unlabeled test set
- `q` = estimated target class prior vector (what we solve for)

Training sample weights are then `w(x, y) = q_hat[y] / p_train[y]` — classes underrepresented in training relative to the estimated test distribution are upweighted.

ELSA is preferred over BBSE because it does not require a well-calibrated classifier, is √n-consistent (faster convergence), and is computationally cheaper (linear solve vs matrix inversion).

**Combined weighting:** If importance weights from adversarial validation are already present (mixed shift case), label shift weights are multiplied with them elementwise and renormalised. The two corrections are complementary and stack correctly.

---

## Output

The mitigated training data is saved as a CSV at:
```
{original_train_path}_mitigated.csv
```

It contains all original columns plus a `__sample_weight__` column. `models.py` reads this column and passes it to LightGBM as `sample_weight` during `.fit()`. This is the standard way to apply importance weights to gradient-boosted trees without modifying the model architecture.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | ≥ 4.0.0 | Domain classifier, proxy model, final model |
| `scikit-learn` | ≥ 1.3.0 | Cross-validation, preprocessing, metrics |
| `pandas` | ≥ 2.0.0 | Data loading and manipulation |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scipy` | ≥ 1.11.0 | KS test, Wasserstein distance, NNLS solver |
| `tabulate` | ≥ 0.9.0 | Console output formatting |
| `shap` | ≥ 0.44.0 | SHAP-based explanation shift (optional) |

---

## Key Papers

| ID | Paper | Method used |
|----|-------|-------------|
| P01 | Gardner et al., NeurIPS 2023 — *TableShift* | Benchmark grounding; confirms XGBoost/LightGBM baseline |
| P02 | Liu et al., NeurIPS 2023 — *WhyShift* | Motivates shift type classification before mitigation |
| P03 | Polo et al., 2022 — *Unified Shift Diagnostics* | Shift type classifier design |
| P09 | Mougan et al., 2022 — *Explanation Shift* | SHAP-based drift detection |
| P14 | Qian et al., 2021 — *Adversarial Validation* | Adversarial validation + reweighting pipeline |
| P16 | Kato et al., 2023 — *Double Debiased Covariate Shift* | Doubly-robust importance weighting |
| P19 | Xu et al., ICML 2022 — *Independence-driven IW* | Decorrelation in feature alignment |
| P21 | Anonymous, 2025 — *Sample Weight Averaging* | Weight averaging across seeds |
| P22 | Tian et al., ICML 2023 — *ELSA* | Label shift correction |
| P26 | Nastl & Hardt, NeurIPS 2024 — *Causal Predictors* | Motivates NOT applying causal feature filtering |
| P39 | Mougan et al., AAAI 2023 — *Explanation Shift Detector* | ESD classifier design |

---

## File Reference

```
detection/
    __init__.py                 — exposes run_detection()
    detection_main.py           — orchestrates all four detectors
    adversarial_validation.py   — domain classifier + sample weights
    statistical_tests.py        — KS, PSI, Wasserstein per feature
    shift_type_classifier.py    — covariate / label / concept diagnosis
    explanation_shift.py        — SHAP-based drift + ESD classifier

mitigation/
    __init__.py                 — exposes run_mitigation()
    mitigation_main.py          — routes by shift type, applies in order
    importance_weighting.py     — adversarial reweighting + doubly-robust
    feature_alignment.py        — quantile normalisation + decorrelation
    feature_selection.py        — drop high-drift / low-importance only
    label_shift_correction.py   — ELSA moment matching
```