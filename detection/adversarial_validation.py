"""
Adversarial Validation (P14: Qian et al. 2021)
- Trains a LightGBM domain classifier (train=0, test=1)
- Returns AUC as shift severity score
- Returns per-sample weights: P(test | x) for training rows
- Weight averaging over multiple seeds for stability (P21)
- Domain classifier feature importances reveal which features drive shift

Artifact pruning (auto, no hardcoded names):
  Before training, drop near-ID columns (n_unique/n_rows > 0.95).
  If AUC > 0.95 after training, iteratively exclude features that have
  high univariate domain AUC but low KS drift score — these are
  split-construction artifacts, not genuine drift. Repeat up to 3 times.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


_NEAR_ID_THRESHOLD      = 0.95   # n_unique / n_rows above this → likely identifier
_AUC_INFLATION_CAP      = 0.95   # AUC above this triggers artifact pruning
_UNIVARIATE_AUC_MIN     = 0.80   # univariate AUC above this = feature is too separating
_DRIFT_SCORE_MAX        = 0.15   # KS drift score below this = feature has low real drift
_MAX_PRUNE_ITERATIONS   = 3


def _train_domain_clf(X_combined, y_combined, n_seeds, n_folds):
    """Core domain classifier training. Returns (mean_auc, all_probs, feat_importances)."""
    all_probs        = np.zeros(len(X_combined))
    all_aucs         = []
    feat_importances = np.zeros(X_combined.shape[1])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for seed in range(n_seeds):
        fold_probs = np.zeros(len(X_combined))
        for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(X_combined, y_combined)):
            clf = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, random_state=seed * 100 + fold_idx,
                verbose=-1, n_jobs=-1,
            )
            clf.fit(X_combined.iloc[idx_tr], y_combined[idx_tr])
            proba = clf.predict_proba(X_combined.iloc[idx_val])
            if isinstance(proba, list):
                proba = np.array(proba)
            fold_probs[idx_val] = np.array(proba)[:, 1]
            feat_importances += clf.feature_importances_

        all_aucs.append(roc_auc_score(y_combined, fold_probs))
        all_probs += fold_probs

    all_probs /= n_seeds
    feat_importances /= (n_seeds * n_folds)
    return float(np.mean(all_aucs)), all_probs, feat_importances


def _univariate_domain_auc(col_vals_combined, y_combined):
    """AUC of a single feature as a domain classifier (rank-based, no model needed)."""
    return roc_auc_score(y_combined, col_vals_combined)


def _encode_for_domain(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X


def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drift_scores: pd.Series = None,
    n_seeds: int = 5,
    n_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Trains a domain classifier to distinguish train from test.

    Parameters
    ----------
    drift_scores : pd.Series (optional) — KS-based drift score per feature from
                   statistical_tests. Used to cross-reference against univariate
                   domain AUC during artifact pruning. If None, pruning uses
                   univariate AUC alone.

    Returns
    -------
    dict with keys:
        auc              : float
        sample_weights   : np.ndarray
        feature_scores   : pd.Series
        shift_detected   : bool
        excluded_cols    : list — columns removed during artifact pruning
    """
    _drop = {"target"}
    all_feature_cols = [c for c in train_df.columns if c not in _drop]

    n_train = len(train_df)
    n_test  = len(test_df)

    # ── Pre-filter: near-ID columns ───────────────────────────────────────────
    # Columns where almost every value is unique are identifiers, not features.
    # Check against the larger of train/test to be conservative.
    near_id_cols = [
        c for c in all_feature_cols
        if train_df[c].nunique() / max(n_train, 1) > _NEAR_ID_THRESHOLD
        or (c in test_df.columns and test_df[c].nunique() / max(n_test, 1) > _NEAR_ID_THRESHOLD)
    ]
    excluded_cols = list(near_id_cols)
    active_cols   = [c for c in all_feature_cols if c not in excluded_cols and c in test_df.columns]

    if near_id_cols and verbose:
        print(f"  [AdvVal] Pre-filtered {len(near_id_cols)} near-ID columns: {near_id_cols}")

    X_train = train_df[active_cols].copy()
    X_test  = test_df[active_cols].copy()
    X_combined = _encode_for_domain(pd.concat([X_train, X_test], axis=0, ignore_index=True))
    y_combined = np.array([0] * n_train + [1] * n_test)

    mean_auc, all_probs, feat_importances = _train_domain_clf(
        X_combined, y_combined, n_seeds, n_folds
    )

    # ── Iterative artifact pruning ────────────────────────────────────────────
    # If AUC is suspiciously high, find features that are individually powerful
    # domain separators but have low actual drift — these are artifacts.
    for iteration in range(_MAX_PRUNE_ITERATIONS):
        if mean_auc <= _AUC_INFLATION_CAP:
            break

        if verbose:
            print(f"  [AdvVal] AUC={mean_auc:.4f} > {_AUC_INFLATION_CAP} — checking for split-construction artifacts (iter {iteration+1})")

        feat_imp_series = pd.Series(feat_importances, index=active_cols).sort_values(ascending=False)
        top_features    = feat_imp_series.head(10).index.tolist()

        # Compute univariate domain AUC for each top feature
        newly_excluded = []
        for col in top_features:
            col_idx = active_cols.index(col)
            col_vals = X_combined.iloc[:, col_idx]

            # Encode categoricals to numeric for AUC computation
            if hasattr(col_vals, "cat"):
                col_vals_num = col_vals.cat.codes.astype(float)
            else:
                col_vals_num = pd.to_numeric(col_vals, errors="coerce").fillna(0)

            try:
                uni_auc = _univariate_domain_auc(col_vals_num, y_combined)
                # Reflect AUC < 0.5 (inverted separability is still separability)
                uni_auc = max(uni_auc, 1.0 - uni_auc)
            except Exception:
                continue

            # Cross-reference with KS drift score if available
            ks_drift = float(drift_scores.get(col, 1.0)) if drift_scores is not None else 1.0

            is_artifact = uni_auc >= _UNIVARIATE_AUC_MIN and ks_drift < _DRIFT_SCORE_MAX
            if is_artifact:
                newly_excluded.append(col)
                if verbose:
                    print(f"    [AdvVal] Excluding '{col}': univariate AUC={uni_auc:.3f}, KS drift={ks_drift:.3f}")

        if not newly_excluded:
            if verbose:
                print(f"  [AdvVal] No artifact features found. Stopping pruning at AUC={mean_auc:.4f}.")
            break

        excluded_cols.extend(newly_excluded)
        active_cols = [c for c in active_cols if c not in newly_excluded]

        if not active_cols:
            break

        X_train    = train_df[active_cols].copy()
        X_test     = test_df[active_cols].copy()
        X_combined = _encode_for_domain(pd.concat([X_train, X_test], axis=0, ignore_index=True))
        feat_importances = np.zeros(len(active_cols))

        mean_auc, all_probs, feat_importances = _train_domain_clf(
            X_combined, y_combined, n_seeds, n_folds
        )

    if mean_auc > _AUC_INFLATION_CAP and verbose:
        print(f"  [AdvVal] WARNING: AUC={mean_auc:.4f} remains high after pruning. "
              f"Shift may be real or additional artifacts remain.")

    # ── Compute sample weights from final model ───────────────────────────────
    train_domain_probs = all_probs[:n_train]
    eps = 1e-6
    raw_weights = train_domain_probs / np.clip(1.0 - train_domain_probs, eps, None)
    raw_weights = np.clip(raw_weights, eps, np.percentile(raw_weights, 99))
    sample_weights = raw_weights / raw_weights.mean()

    feat_importances_final = feat_importances  # already averaged inside _train_domain_clf
    feature_scores = pd.Series(feat_importances_final, index=active_cols).sort_values(ascending=False)

    shift_detected = mean_auc > 0.6

    if verbose:
        print(f"  [AdvVal] Domain classifier AUC: {mean_auc:.4f}  |  Shift detected: {shift_detected}")
        print(f"  [AdvVal] Top-5 shift-driving features: {list(feature_scores.head(5).index)}")
        if excluded_cols:
            print(f"  [AdvVal] Excluded from domain classifier: {excluded_cols}")

    return {
        "auc":            mean_auc,
        "sample_weights": sample_weights,
        "feature_scores": feature_scores,
        "shift_detected": shift_detected,
        "excluded_cols":  excluded_cols,
    }