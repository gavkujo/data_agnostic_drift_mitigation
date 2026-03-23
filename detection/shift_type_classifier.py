"""
Shift Type Classifier (P03: Polo et al. 2022)
Diagnoses the dominant shift type before applying any mitigation.

Three shift types:
  COVARIATE  — P(X) changed, P(Y|X) stable
  LABEL      — P(Y) changed, P(X|Y) stable
  CONCEPT    — P(Y|X) changed (hardest case)
  MIXED      — multiple types co-occurring

Strategy:
  1. Domain classifier AUC tells us if X-space shifted at all
  2. Marginal label distribution comparison (train labels only, inferred from model)
  3. Conditional prediction stability: if a model trained on train predicts
     similarly on both splits, P(Y|X) is stable -> covariate shift.
     If conditional predictions shift, concept drift is present.

Note: Since we only have unlabeled test data, concept drift can only be
inferred indirectly via prediction confidence and explanation shift signals.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


_DROP_COLS = {"target", "CustomerID", "Month"}


def _is_classification(train_df: pd.DataFrame) -> bool:
    target = train_df["target"]
    n_unique = target.nunique()
    return n_unique <= 20 or pd.api.types.is_object_dtype(target)


def _prepare_X(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Align to feature_cols, fill numeric NaNs, encode categoricals for LightGBM."""
    X = df.reindex(columns=feature_cols).copy()
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(-9999)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X


def _train_proxy_model(train_df: pd.DataFrame) -> object:
    """Trains a quick proxy model on training data to generate predictions."""
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]
    X = _prepare_X(train_df, feature_cols)
    y = train_df["target"]

    if _is_classification(train_df):
        if pd.api.types.is_object_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                              random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X, y)
        return clf
    else:
        reg = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                             random_state=42, verbose=-1, n_jobs=-1)
        reg.fit(X, y)
        return reg


def classify_shift_type(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    adv_val_auc: float,
    verbose: bool = True,
) -> dict:
    """
    Classifies the dominant type of shift between train and test.

    Parameters
    ----------
    train_df      : training dataframe WITH target column
    test_df       : test dataframe WITHOUT target column
    adv_val_auc   : AUC from adversarial validation (pre-computed)

    Returns
    -------
    dict with keys:
        shift_type          : str   - 'none' | 'covariate' | 'label' | 'concept' | 'mixed'
        covariate_detected  : bool
        label_detected      : bool
        concept_detected    : bool
        confidence          : float - confidence in diagnosis (0–1)
        notes               : str   - human-readable explanation
    """
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]

    covariate_detected = adv_val_auc > 0.60
    label_detected     = False
    concept_detected   = False
    notes_parts        = []

    # ── Signal 1: Covariate shift from adversarial validation ────────────────
    if covariate_detected:
        notes_parts.append(f"X-space shift detected (domain AUC={adv_val_auc:.3f}).")
    else:
        notes_parts.append(f"Minimal X-space shift (domain AUC={adv_val_auc:.3f}).")

    # ── Signal 2: Label shift — compare train label distribution to proxy
    #    predictions on test (since test has no labels, we use the proxy model)
    if "target" in train_df.columns:
        proxy_model = _train_proxy_model(train_df)
        X_test = _prepare_X(test_df, [c for c in feature_cols if c in test_df.columns])

        is_clf = _is_classification(train_df)

        if is_clf:
            le = LabelEncoder()
            y_enc = le.fit_transform(train_df["target"])
            train_label_dist = pd.Series(y_enc).value_counts(normalize=True).sort_index()

            if hasattr(proxy_model, "predict"):
                test_preds = proxy_model.predict(X_test)
                test_label_dist = pd.Series(test_preds).value_counts(normalize=True).sort_index()
            else:
                test_label_dist = pd.Series([], dtype=float)

            # Align indices
            all_labels = train_label_dist.index.union(test_label_dist.index)
            train_vec = train_label_dist.reindex(all_labels, fill_value=0).values
            test_vec  = test_label_dist.reindex(all_labels, fill_value=0).values
            # Ensure float dtype for subtraction
            train_vec = np.array(train_vec, dtype=float)
            test_vec = np.array(test_vec, dtype=float)

            # Total variation distance between distributions
            tv_distance = float(np.sum(np.abs(train_vec - test_vec)) / 2)
            label_detected = tv_distance > 0.05
            notes_parts.append(f"Label distribution TV-distance: {tv_distance:.3f} ({'shift' if label_detected else 'stable'}).")

        else:
            # Regression: compare train target mean/std to predicted test values
            train_mean = train_df["target"].mean()
            train_std  = train_df["target"].std()
            if hasattr(proxy_model, "predict"):
                test_preds = proxy_model.predict(X_test)
                pred_mean  = float(np.mean(test_preds))
                pred_std   = float(np.std(test_preds))
            else:
                pred_mean, pred_std = train_mean, train_std

            mean_shift = abs(pred_mean - train_mean) / (train_std + 1e-8)
            label_detected = mean_shift > 0.2
            notes_parts.append(f"Target mean shift (normalised): {mean_shift:.3f} ({'shift' if label_detected else 'stable'}).")

        # ── Signal 3: Concept drift — prediction confidence on test vs train
        #    If the model is systematically less confident on test, the conditional
        #    relationship P(Y|X) may have shifted (concept drift indicator).
        if is_clf and hasattr(proxy_model, "predict_proba"):
            X_train_feat = _prepare_X(train_df, feature_cols)
            train_probs  = proxy_model.predict_proba(X_train_feat)
            test_probs   = proxy_model.predict_proba(X_test)
            # Ensure dense output for max(axis=1)
            if hasattr(train_probs, 'toarray'):
                train_probs = train_probs.toarray()
            if hasattr(test_probs, 'toarray'):
                test_probs = test_probs.toarray()
            train_probs = np.array(train_probs)
            test_probs = np.array(test_probs)
            train_probs = train_probs.max(axis=1)
            test_probs = test_probs.max(axis=1)

            train_conf = float(np.mean(train_probs))
            test_conf  = float(np.mean(test_probs))
            conf_drop  = train_conf - test_conf

            # Significant confidence drop on test → model's P(Y|X) assumption may be wrong
            concept_detected = conf_drop > 0.05
            notes_parts.append(
                f"Prediction confidence: train={train_conf:.3f}, test={test_conf:.3f}, "
                f"drop={conf_drop:.3f} ({'concept drift signal' if concept_detected else 'stable'})."
            )

    # ── Determine dominant shift type ────────────────────────────────────────
    n_types = int(covariate_detected) + int(label_detected) + int(concept_detected)

    if n_types == 0:
        shift_type = "none"
        confidence = 1.0 - adv_val_auc  # high confidence in no-shift when AUC ≈ 0.5
    elif covariate_detected and not label_detected and not concept_detected:
        shift_type = "covariate"
        confidence = min((adv_val_auc - 0.5) * 2, 1.0)
    elif label_detected and not covariate_detected and not concept_detected:
        shift_type = "label"
        confidence = 0.7
    elif concept_detected and not covariate_detected:
        shift_type = "concept"
        confidence = 0.6
    elif n_types >= 2:
        shift_type = "mixed"
        confidence = 0.5
    else:
        shift_type = "covariate"  # default to covariate if unclear
        confidence = 0.5

    notes = " ".join(notes_parts)

    if verbose:
        print(f"  [ShiftType] Dominant shift type: {shift_type.upper()} (confidence={confidence:.2f})")
        print(f"  [ShiftType] {notes}")

    return {
        "shift_type":         shift_type,
        "covariate_detected": covariate_detected,
        "label_detected":     label_detected,
        "concept_detected":   concept_detected,
        "confidence":         confidence,
        "notes":              notes,
    }