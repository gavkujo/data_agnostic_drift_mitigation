"""
Detection Main — orchestrates all drift detectors and produces a unified drift_info dict.

Order of operations:
  1. Statistical tests (KS, PSI, EMD) per feature         -> fast, always runs
  2. Adversarial validation (domain classifier)            -> moderate cost, always runs
  3. Shift type classification (covariate/label/concept)   -> uses signals from 1+2
  4. Explanation shift (SHAP-based)                        -> slowest, runs if shap installed

The resulting drift_info dict is the contract between detection and mitigation.
"""

import pandas as pd
from tabulate import tabulate

from .statistical_tests       import run_statistical_tests
from .adversarial_validation  import run_adversarial_validation
from .shift_type_classifier   import classify_shift_type
from .explanation_shift       import run_explanation_shift


def run_detection(train_path: str, test_path: str, verbose: bool = True) -> dict:
    """
    Full drift detection pipeline.

    Parameters
    ----------
    train_path : path to training CSV (must include 'target' column)
    test_path  : path to test CSV (no target column)

    Returns
    -------
    drift_info : dict — unified contract passed to mitigation
        {
          'shift_type'        : str   - 'none'|'covariate'|'label'|'concept'|'mixed'
          'columns_with_drift': list  - features flagged by statistical tests
          'drift_scores'      : pd.Series - composite drift score per feature
          'sample_weights'    : np.ndarray - importance weights for train rows
          'adv_auc'           : float - domain classifier AUC
          'feature_importances': pd.Series - which features drive shift (adv. validation)
          'shap_drifted_cols' : list  - features with SHAP-space shift
          'esd_auc'           : float|None - explanation shift detector AUC
          'stat_results'      : pd.DataFrame - full per-feature stats table
          'shift_diagnosis'   : dict  - full output from shift type classifier
        }
    """
    print("\n[DETECTION] Loading data...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Normalise target column name to "target"
    if "ChurnStatus" in train_df.columns:
        train_df = train_df.rename(columns={"ChurnStatus": "target"})

    _drop_cols = {"target", "CustomerID", "Month", "ChurnStatus"}
    feature_cols = [c for c in train_df.columns if c not in _drop_cols]
    test_feature_cols = [c for c in test_df.columns if c not in _drop_cols]

    # Align columns: test may not have target
    common_features = [c for c in feature_cols if c in test_feature_cols]
    train_features_df = train_df[common_features]
    test_features_df  = test_df[common_features]

    # ── Step 1: Statistical tests ─────────────────────────────────────────────
    print("\n[DETECTION] Step 1/4 — Statistical tests (KS, PSI, Wasserstein)...")
    stat_results = run_statistical_tests(
        train_df=train_features_df,
        test_df=test_features_df,
        verbose=verbose,
    )

    # ── Step 2: Adversarial validation ───────────────────────────────────────
    print("\n[DETECTION] Step 2/4 — Adversarial validation...")
    adv_results = run_adversarial_validation(
        train_df=train_features_df,
        test_df=test_features_df,
        drift_scores=stat_results["drift_scores"],
        verbose=verbose,
    )

    # ── Step 3: Shift type classification ────────────────────────────────────
    print("\n[DETECTION] Step 3/4 — Shift type classification...")
    shift_diagnosis = classify_shift_type(
        train_df=train_df,        # needs target column
        test_df=test_features_df,
        adv_val_auc=adv_results["auc"],
        verbose=verbose,
    )

    # ── Step 4: Explanation shift (SHAP) ─────────────────────────────────────
    print("\n[DETECTION] Step 4/4 — Explanation shift (SHAP-based)...")
    expl_results = run_explanation_shift(
        train_df=train_df,
        test_df=test_features_df,
        verbose=verbose,
    )

    # ── Assemble unified drift_info ───────────────────────────────────────────
    drift_info = {
        "shift_type":          shift_diagnosis["shift_type"],
        "columns_with_drift":  stat_results["drifted_columns"],
        "drift_scores":        stat_results["drift_scores"],
        "sample_weights":      adv_results["sample_weights"],
        "adv_auc":             adv_results["auc"],
        "feature_importances": adv_results["feature_scores"],
        "shap_drifted_cols":   expl_results["shap_drifted_cols"],
        "esd_auc":             expl_results["esd_auc"],
        "shap_drift_scores":   expl_results["shap_drift_scores"],
        "stat_results":        stat_results["results"],
        "shift_diagnosis":     shift_diagnosis,
        # Pass through raw train/test for mitigation to use if needed
        "_train_df":           train_df,
        "_test_df":            test_df,
    }

    if verbose:
        _print_detection_summary(drift_info)

    return drift_info


_DRIFT_DESCRIPTIONS = {
    "covariate_shape":       "Full distribution shift (location + shape)",
    "covariate_location":    "Location/median shift only",
    "covariate_categorical": "Category proportion shift",
    "none":                  "No significant drift",
}

_MITIGATION_DESCRIPTIONS = {
    "covariate_shape":       "Quantile normalisation",
    "covariate_location":    "Median centering",
    "covariate_categorical": "Test-frequency encoding",
    "none":                  "None",
}


def _print_detection_summary(drift_info: dict) -> None:
    print("\n" + "=" * 60)
    print("  DRIFT DETECTION SUMMARY")
    print("=" * 60)

    diag = drift_info["shift_diagnosis"]
    print(f"  Dominant shift type : {drift_info['shift_type'].upper()}")
    print(f"  Covariate shift     : {'YES' if diag['covariate_detected'] else 'NO'}")
    print(f"  Label shift         : {'YES' if diag['label_detected'] else 'NO'}")
    print(f"  Concept drift       : {'YES' if diag['concept_detected'] else 'NO'}")
    print(f"  Domain AUC          : {drift_info['adv_auc']:.4f}")
    if drift_info["esd_auc"] is not None:
        print(f"  ESD AUC (SHAP)      : {drift_info['esd_auc']:.4f}")
    print(f"  Notes               : {diag['notes']}")

    stat_results = drift_info.get("stat_results", pd.DataFrame())
    if not stat_results.empty and "feature_shift_type" in stat_results.columns:
        drifted = stat_results[stat_results["feature_shift_type"] != "none"].copy()
        if not drifted.empty:
            train_df = drift_info.get("_train_df", pd.DataFrame())
            rows = []
            for _, row in drifted.iterrows():
                col   = row["column"]
                ftype = row["feature_shift_type"]
                dtype = "categorical" if (
                    col in train_df.columns and
                    not pd.api.types.is_numeric_dtype(train_df[col])
                ) else "numeric"
                rows.append({
                    "feature":     col,
                    "type":        dtype,
                    "drift":       _DRIFT_DESCRIPTIONS.get(ftype, ftype),
                    "mitigation":  _MITIGATION_DESCRIPTIONS.get(ftype, ftype),
                    "drift_score": f"{row['drift_score']:.3f}",
                })
            print(f"\n  Per-feature drift plan ({len(rows)} features):")
            print(tabulate(
                rows,
                headers={"feature": "Feature", "type": "Type",
                         "drift": "Drift detected", "mitigation": "Mitigation",
                         "drift_score": "Score"},
                tablefmt="simple",
                showindex=False,
            ))

    print("=" * 60 + "\n")