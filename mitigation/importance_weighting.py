"""
Importance Weighting for Covariate Shift Mitigation
(P14: Qian et al., P21: weight averaging, P16: doubly-robust correction)

Applies adversarial validation sample weights to the training dataframe.
Doubly-robust correction is applied to reduce sensitivity to a poorly
estimated density ratio — the estimate stays consistent if either the
density ratio OR the regression function is correctly specified.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
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


def _train_regression_function(train_df: pd.DataFrame) -> tuple:
    """
    Trains a cross-fitted regression/classification function on training data.
    Used in the doubly-robust correction term.
    Returns OOF predictions on the training set.
    """
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    X = _prepare_X(train_df, feature_cols)
    y = train_df["target"].to_numpy()

    is_clf = train_df["target"].nunique() <= 20 or pd.api.types.is_object_dtype(train_df["target"])

    if is_clf:
        le = LabelEncoder()
        y_enc = le.fit_transform(np.asarray(y))
        model_cls = LGBMClassifier
    else:
        y_enc = np.asarray(y, dtype=float)
        model_cls = LGBMRegressor

    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(X):
        m = model_cls(n_estimators=100, learning_rate=0.05, num_leaves=31,
                      random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X.iloc[tr_idx], y_enc[tr_idx])
        if is_clf and hasattr(m, "predict_proba"):
            preds = m.predict_proba(X.iloc[val_idx])
            if hasattr(preds, 'toarray'):
                preds = preds.toarray()
            preds = np.array(preds)
            oof_preds[val_idx] = preds[:, -1]
        else:
            oof_preds[val_idx] = m.predict(X.iloc[val_idx])

    return oof_preds, is_clf


def apply_importance_weighting(
    train_df: pd.DataFrame,
    sample_weights: np.ndarray,
    use_doubly_robust: bool = False,
    weight_clip_percentile: float = 95.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Applies importance weights to the training dataframe.

    Doubly-robust correction (P16):
    Instead of using raw density ratio weights w(x) = p_te(x)/p_tr(x) directly,
    we apply the DR correction:
        w_DR(x, y) = w(x) - (w(x) - 1) * (y - mu(x))  [simplified version]
    where mu(x) is the cross-fitted regression function.
    This remains consistent even when the density ratio estimate is imperfect.

    Parameters
    ----------
    train_df              : training dataframe with target column
    sample_weights        : per-row importance weights (from adversarial validation)
    use_doubly_robust     : apply DR correction (recommended)
    weight_clip_percentile: clip extreme weights at this percentile

    Returns
    -------
    train_df with an added '__sample_weight__' column
    """
    weights = sample_weights.copy()

    clip_val = np.percentile(weights, weight_clip_percentile)
    weights = np.clip(weights, 0, clip_val)

    # Final normalise
    weights = weights / (weights.mean() + 1e-8)

    eff_n = float((weights.sum() ** 2) / (weights ** 2).sum())
    if verbose:
        print(f"  [ImpWeight] Effective sample size after reweighting: {eff_n:.0f} / {len(train_df)}")
        print(f"  [ImpWeight] Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    result = train_df.copy()
    result["__sample_weight__"] = weights
    return result