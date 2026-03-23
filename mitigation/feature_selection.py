"""
Feature Selection Under Drift
(P26: Nastl & Hardt 2024 — do NOT drop based on causal assumptions)
(P14: domain classifier feature importances as practical guide)

Key finding from P26: dropping non-causal features reliably HURTS performance.
Do NOT apply IRM-style causal filtering.

What we DO instead:
  - Drop features with BOTH high drift AND near-zero predictive importance
  - "High drift" = flagged by ≥2 statistical tests
  - "Low importance" = bottom 10% of feature importance in a proxy model
  - Features that drift but are still predictive: keep and align instead (feature_alignment.py)
  - Features with zero variance in test: always drop

This is a conservative strategy: we only drop features where drift
AND uselessness coincide. Everything else stays.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder


_DROP_COLS = {"target", "CustomerID", "Month", "__sample_weight__"}


def _prepare_X(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Align to feature_cols, fill numeric NaNs, encode categoricals for LightGBM."""
    X = df.reindex(columns=feature_cols).copy()
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(-9999)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X


def _get_feature_importances(train_df: pd.DataFrame) -> pd.Series:
    """Trains a quick proxy model and returns feature importances."""
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    X = _prepare_X(train_df, feature_cols)
    y = train_df["target"]

    is_clf = y.nunique() <= 20 or pd.api.types.is_object_dtype(y)

    if is_clf:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X, y_enc)
    else:
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1,
                               num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X, y.astype(float))

    return pd.Series(model.feature_importances_, index=feature_cols)


def apply_feature_selection(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    drifted_columns: list,
    drift_scores: pd.Series,
    importance_threshold_pct: float = 0.05,
    drift_score_threshold: float = 0.8,
    verbose: bool = True,
) -> tuple:
    """
    Conservatively drops features with high drift AND low predictive importance.

    Parameters
    ----------
    train_df                  : training dataframe with target
    test_df                   : test dataframe
    drifted_columns           : columns flagged by statistical tests
    drift_scores              : composite drift score per feature (from stat tests)
    importance_threshold_pct  : drop if feature is in bottom X% of importances
    drift_score_threshold     : only consider dropping if drift_score > this value

    Returns
    -------
    (train_df_filtered, test_df_filtered, dropped_columns)
    """
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    dropped = []

    # ── Zero-variance features in test: always drop ───────────────────────────
    zero_var_test = [
        c for c in feature_cols
        if c in test_df.columns
        and pd.api.types.is_numeric_dtype(test_df[c])
        and test_df[c].std() < 1e-8
    ]
    if zero_var_test and verbose:
        print(f"  [FeatSel] Dropping {len(zero_var_test)} zero-variance test features: {zero_var_test}")
    dropped.extend(zero_var_test)

    # ── High drift + low importance: drop ────────────────────────────────────
    if drifted_columns and "target" in train_df.columns:
        feat_importances = _get_feature_importances(train_df)
        importance_cutoff = feat_importances.quantile(importance_threshold_pct)

        for col in drifted_columns:
            if col in dropped:
                continue
            col_drift  = float(drift_scores.get(col, 0.0))
            col_import = float(feat_importances.get(col, 0.0))

            # Drop only if BOTH conditions hold
            is_high_drift   = col_drift >= drift_score_threshold
            is_low_import   = col_import <= importance_cutoff

            if is_high_drift and is_low_import:
                dropped.append(col)
                if verbose:
                    print(f"  [FeatSel] Dropping '{col}': drift={col_drift:.3f}, importance={col_import:.1f}")

    if not dropped:
        if verbose:
            print("  [FeatSel] No features dropped (all drifted features retain predictive value).")
        return train_df, test_df, []

    keep_cols = [c for c in train_df.columns if c not in dropped]
    keep_test  = [c for c in test_df.columns  if c not in dropped]

    if verbose:
        print(f"  [FeatSel] Total dropped: {len(dropped)} features. Remaining: {len(keep_cols) - 1} features.")

    return train_df[keep_cols], test_df[keep_test], dropped