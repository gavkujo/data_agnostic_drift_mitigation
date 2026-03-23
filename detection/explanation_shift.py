"""
Explanation Shift Detection (P09: Mougan et al. 2022, P39: Mougan et al. 2023)

Key insight: monitoring changes in SHAP value distributions catches
interaction-driven drift that per-feature marginal KS/PSI tests miss.

Pipeline:
  1. Train proxy model on training data
  2. Compute SHAP values for train samples and test samples
  3. Compare SHAP distributions per feature using KS test
  4. Train an Explanation Shift Detector (ESD): a classifier on
     (train SHAP, label=0) vs (test SHAP, label=1)
  5. ESD feature importances rank which features' contribution patterns changed
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


_DROP_COLS = {"target", "CustomerID", "Month"}


def _prepare_X(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Align to feature_cols, fill numeric NaNs, encode categoricals for LightGBM."""
    X = df.reindex(columns=feature_cols).copy()
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(-9999)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X


def _train_proxy_model(train_df: pd.DataFrame) -> object:
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    X = _prepare_X(train_df, feature_cols)
    y = train_df["target"]

    n_unique = y.nunique()
    is_clf = n_unique <= 20 or pd.api.types.is_object_dtype(y)

    if is_clf:
        if pd.api.types.is_object_dtype(y):
            y = LabelEncoder().fit_transform(y)
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    else:
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X, y)
    return model, feature_cols, is_clf


def run_explanation_shift(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    max_shap_samples: int = 2000,
    verbose: bool = True,
) -> dict:
    """
    Computes SHAP-based explanation shift between train and test.

    Parameters
    ----------
    train_df          : training dataframe WITH target column
    test_df           : test dataframe WITHOUT target (same feature columns)
    max_shap_samples  : max rows to compute SHAP on (for speed)

    Returns
    -------
    dict with keys:
        shap_drift_scores    : pd.Series  - KS stat per feature in SHAP space (higher = more shift)
        shap_drifted_cols    : list[str]  - features with significant SHAP distribution shift
        esd_auc              : float      - ESD classifier AUC (higher = more explanation shift)
        esd_feature_scores   : pd.Series  - which features' SHAP values shifted most (ESD importances)
        shap_train           : np.ndarray - SHAP values on train sample (for downstream use)
        shap_test            : np.ndarray - SHAP values on test sample (for downstream use)
    """
    if not SHAP_AVAILABLE:
        if verbose:
            print("  [ExplShift] shap not installed. Skipping explanation shift. Run: pip install shap")
        return {
            "shap_drift_scores":  pd.Series(dtype=float),
            "shap_drifted_cols":  [],
            "esd_auc":            None,
            "esd_feature_scores": pd.Series(dtype=float),
            "shap_train":         None,
            "shap_test":          None,
        }

    model, feature_cols, is_clf = _train_proxy_model(train_df)

    X_train_feat = _prepare_X(train_df, feature_cols)
    X_test_feat  = _prepare_X(test_df, feature_cols)

    # Sample for SHAP (full dataset can be slow)
    n_tr = min(max_shap_samples, len(X_train_feat))
    n_te = min(max_shap_samples, len(X_test_feat))

    X_tr_sample = X_train_feat.sample(n=n_tr, random_state=42)
    X_te_sample = X_test_feat.sample(n=n_te, random_state=42)

    # SHAP TreeExplainer is fast for LightGBM
    explainer = shap.TreeExplainer(model)

    if is_clf and hasattr(model, "classes_") and len(model.classes_) == 2:
        # Binary classification: use SHAP values for positive class
        shap_train = explainer.shap_values(X_tr_sample)
        shap_test  = explainer.shap_values(X_te_sample)
        if isinstance(shap_train, list):
            shap_train = shap_train[1]
            shap_test  = shap_test[1]
    else:
        shap_train = explainer.shap_values(X_tr_sample)
        shap_test  = explainer.shap_values(X_te_sample)
        if isinstance(shap_train, list):
            shap_train = shap_train[0]
            shap_test  = shap_test[0]

    shap_train = np.array(shap_train)
    shap_test  = np.array(shap_test)

    # ── Per-feature KS test in SHAP space ────────────────────────────────────
    shap_ks_records = []
    for i, col in enumerate(feature_cols):
        ks_stat, ks_pval = ks_2samp(shap_train[:, i], shap_test[:, i])
        shap_ks_records.append({"column": col, "shap_ks_stat": ks_stat, "shap_ks_pval": ks_pval})

    shap_df = pd.DataFrame(shap_ks_records).sort_values("shap_ks_stat", ascending=False)
    shap_drift_scores = shap_df.set_index("column")["shap_ks_stat"]
    shap_drifted_cols = shap_df.loc[shap_df["shap_ks_pval"] < 0.05, "column"].tolist()

    # ── Explanation Shift Detector (ESD) ─────────────────────────────────────
    # Train a classifier to distinguish train SHAP vectors from test SHAP vectors
    shap_combined = np.vstack([shap_train, shap_test])
    y_domain      = np.array([0] * len(shap_train) + [1] * len(shap_test))

    esd = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15,
                          random_state=42, verbose=-1, n_jobs=-1)
    esd.fit(shap_combined, y_domain)

    esd_probs = esd.predict_proba(shap_combined)[:, 1]
    esd_auc   = float(roc_auc_score(y_domain, esd_probs))

    esd_feature_scores = pd.Series(
        esd.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    if verbose:
        print(f"  [ExplShift] ESD AUC: {esd_auc:.4f}  (>0.6 = significant explanation shift)")
        print(f"  [ExplShift] {len(shap_drifted_cols)} features with SHAP distribution shift (KS p<0.05)")
        if shap_drifted_cols:
            print(f"  [ExplShift] Top SHAP-shifted: {shap_drifted_cols[:5]}")

    return {
        "shap_drift_scores":  shap_drift_scores,
        "shap_drifted_cols":  shap_drifted_cols,
        "esd_auc":            esd_auc,
        "esd_feature_scores": esd_feature_scores,
        "shap_train":         shap_train,
        "shap_test":          shap_test,
    }