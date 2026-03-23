"""
Label Shift Correction via ELSA (P22: Tian, Zhang & Zhao, ICML 2023)

When label shift is detected as the dominant shift type:
  - Estimates the target label distribution from unlabeled test data
  - Reweights training samples to match the estimated target distribution
  - Does NOT require test labels — only soft predictions (predict_proba)

ELSA (Efficient Label Shift Adaptation):
  Uses semiparametric moment matching: finds weights w_k for each class k
  such that the weighted class-conditional predictions match the observed
  marginal test predictions.

  Specifically: solves for q = (q_1, ..., q_K) (target class priors) via:
    E_test[p(y|x)] ≈ sum_k q_k * E_train[p(y|x) | y=k]

  This is a linear system that can be solved efficiently.
  Training sample weights are then: w(x,y) = q_{y} / p_{train}(y)
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict
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


def _estimate_target_priors_elsa(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    n_classes: int,
) -> np.ndarray:
    """
    ELSA moment-matching estimation of target class priors.

    Returns estimated q: shape (n_classes,) summing to 1.
    """
    X_train = _prepare_X(train_df, feature_cols)
    X_test  = _prepare_X(test_df, feature_cols)
    y_train = train_df["target"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    # Cross-validated soft predictions on train (avoids overfitting bias)
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1, n_jobs=-1)
    train_probs = cross_val_predict(clf, X_train, y_enc, cv=5, method="predict_proba")

    # Fit on all train to get test predictions
    clf.fit(X_train, y_enc)
    test_probs = clf.predict_proba(X_test)   # shape: (n_test, n_classes)

    # ── ELSA moment matching ──────────────────────────────────────────────────
    # Build confusion-like matrix M: M[k, j] = E[p(class j | x) | y=k]
    # i.e. average soft prediction for each true class
    M = np.zeros((n_classes, n_classes))
    for k in range(n_classes):
        mask = y_enc == k
        if mask.sum() > 0:
            M[k, :] = train_probs[mask].mean(axis=0)

    # Target marginal: average soft predictions on test set
    mu_test = test_probs.mean(axis=0)  # shape: (n_classes,)

    # Solve: M^T q = mu_test  for q (target prior vector)
    # Constrained: q >= 0, sum(q) = 1
    # Use non-negative least squares
    from scipy.optimize import nnls
    q_raw, _ = nnls(M.T, mu_test)

    # Normalise
    if q_raw.sum() > 1e-8:
        q = q_raw / q_raw.sum()
    else:
        # Fallback: uniform prior
        q = np.ones(n_classes) / n_classes

    return q, le


def apply_label_shift_correction(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Applies ELSA label shift correction to the training dataframe.

    Computes per-sample weights: w(x, y) = q_hat[y] / p_train[y]
    where q_hat is the estimated target class prior.

    These weights upweight classes underrepresented in training relative
    to the (estimated) test distribution.

    Parameters
    ----------
    train_df : training dataframe WITH target column
    test_df  : test dataframe WITHOUT target column

    Returns
    -------
    train_df with '__sample_weight__' column (or updated if already exists)
    """
    if "target" not in train_df.columns:
        if verbose:
            print("  [LabelShift] No target column found. Skipping label shift correction.")
        return train_df

    y = train_df["target"]
    if y.nunique() > 50 or pd.api.types.is_float_dtype(y):
        if verbose:
            print("  [LabelShift] Target appears continuous. Label shift correction requires classification target. Skipping.")
        return train_df

    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    test_feature_cols = [c for c in feature_cols if c in test_df.columns]

    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y.values)
    n_classes = len(le_target.classes_)

    # Train source priors p_train(y)
    train_priors = np.bincount(y_enc, minlength=n_classes) / len(y_enc)

    # Estimate target priors via ELSA
    q_hat, _ = _estimate_target_priors_elsa(
        train_df=train_df,
        test_df=test_df[test_feature_cols],
        feature_cols=test_feature_cols,
        n_classes=n_classes,
    )

    if verbose:
        print("  [LabelShift] Estimated class prior shift:")
        for k, cls in enumerate(le_target.classes_):
            print(f"    Class {cls}: train={train_priors[k]:.3f} -> estimated test={q_hat[k]:.3f}")

    # Sanity check: if estimated shift is implausibly large, the proxy model
    # is likely corrupted (e.g. by split-construction artifacts in domain classifier).
    # TV distance > 0.15 between estimated test prior and train prior triggers skip.
    tv_distance = float(np.sum(np.abs(q_hat - train_priors)) / 2)
    if tv_distance > 0.15:
        if verbose:
            print(f"  [LabelShift] WARNING: estimated prior shift TV={tv_distance:.3f} > 0.15. "
                  f"Proxy model predictions may be unreliable. Skipping label shift correction.")
        return train_df

    # Per-sample weights: w(y) = q_hat[y] / p_train[y]
    eps = 1e-8
    class_weights = q_hat / np.clip(train_priors, eps, None)
    sample_weights = class_weights[y_enc]

    # Clip and normalise
    clip_val = np.percentile(sample_weights, 99)
    sample_weights = np.clip(sample_weights, 0, clip_val)
    sample_weights = sample_weights / sample_weights.mean()

    result = train_df.copy()

    # If importance weights already exist (from covariate correction), multiply them
    if "__sample_weight__" in result.columns:
        result["__sample_weight__"] = result["__sample_weight__"].values * sample_weights
        result["__sample_weight__"] /= result["__sample_weight__"].mean()
        if verbose:
            print("  [LabelShift] Combined with existing importance weights (multiplicative).")
    else:
        result["__sample_weight__"] = sample_weights

    if verbose:
        eff_n = float((sample_weights.sum() ** 2) / (sample_weights ** 2).sum())
        print(f"  [LabelShift] Effective sample size after label shift correction: {eff_n:.0f} / {len(train_df)}")

    return result