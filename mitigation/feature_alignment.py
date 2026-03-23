"""
Feature Alignment — per-feature targeted correction.

Each drifted feature is routed based on its own `feature_shift_type` label
produced by statistical_tests.py:

  covariate_shape      → full quantile normalisation (shape + location)
  covariate_location   → median centering only
  covariate_categorical→ skip (LightGBM handles categoricals natively)
  none                 → skip
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


_DROP_COLS = {"target", "CustomerID", "Month", "__sample_weight__"}


def _align_quantile(train_vals: np.ndarray, test_vals: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    qt_test = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(test_vals)),
        output_distribution="uniform", random_state=42,
    )
    qt_test.fit(test_vals.reshape(-1, 1))
    qt_train = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(train_vals)),
        output_distribution="uniform", random_state=42,
    )
    qt_train.fit(train_vals.reshape(-1, 1))
    return qt_test.inverse_transform(qt_train.transform(train_vals.reshape(-1, 1))).ravel()


def _align_location(train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    return train_vals + (np.median(test_vals) - np.median(train_vals))


def apply_feature_alignment(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    stat_results: pd.DataFrame,
    drifted_columns: list,
    apply_decorrelation: bool = False,  # retained in signature for compatibility, unused
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-feature targeted alignment. Routing is driven by `feature_shift_type`
    in stat_results — each feature's own diagnostic signals determine the correction.
    Categoricals are skipped (LightGBM handles them natively).
    """
    result = train_df.copy()
    feature_cols = [c for c in train_df.columns if c not in _DROP_COLS]

    shift_type_lookup = {}
    if stat_results is not None and not stat_results.empty and "feature_shift_type" in stat_results.columns:
        shift_type_lookup = stat_results.set_index("column")["feature_shift_type"].to_dict()

    valid_drift_cols = [c for c in drifted_columns if c in feature_cols and c in test_df.columns]

    quantile_aligned = []
    location_aligned = []

    for col in valid_drift_cols:
        is_cat = not pd.api.types.is_numeric_dtype(train_df[col])
        default_type = "covariate_categorical" if is_cat else "covariate_shape"
        ftype = shift_type_lookup.get(col, default_type)

        if ftype in ("none", "covariate_categorical"):
            continue

        train_vals  = train_df[col].values.astype(float)
        test_vals   = test_df[col].values.astype(float)
        train_valid = train_vals[~np.isnan(train_vals)]
        test_valid  = test_vals[~np.isnan(test_vals)]
        if len(train_valid) < 10 or len(test_valid) < 10:
            continue

        if ftype == "covariate_location":
            aligned = _align_location(train_valid, test_valid)
            location_aligned.append(col)
        else:
            aligned = _align_quantile(train_valid, test_valid)
            quantile_aligned.append(col)

        result_col = result[col].values.astype(float)
        result_col[~np.isnan(result_col)] = aligned
        result[col] = result_col

    if verbose:
        if quantile_aligned:
            print(f"  [FeatAlign] Quantile-normalised {len(quantile_aligned)}: {quantile_aligned}")
        if location_aligned:
            print(f"  [FeatAlign] Location-shifted {len(location_aligned)}: {location_aligned}")
        if not quantile_aligned and not location_aligned:
            print("  [FeatAlign] No numeric features required alignment.")

    return result
