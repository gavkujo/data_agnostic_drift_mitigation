"""
Statistical Tests for Per-Feature Drift Detection
- Kolmogorov-Smirnov test (nonparametric, per-column)
- Population Stability Index (PSI)
- Wasserstein / Earth Mover's Distance (magnitude of drift)

Each test returns a score and a flag per feature.
Results are aggregated into a ranked drift summary.

References: Rabanser et al. 2019 (Failing Loudly), standard industry practice.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance, chi2_contingency


# ── PSI ───────────────────────────────────────────────────────────────────────

def _psi_single(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index for a single numerical feature.
    PSI < 0.1  → stable
    PSI 0.1–0.25 → moderate drift
    PSI > 0.25 → major drift
    """
    # Bin edges from expected (train) distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # deduplicate identical percentiles

    if len(breakpoints) < 2:
        return 0.0

    eps = 1e-6
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + eps

    # Normalise to sum to 1
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct   = actual_pct   / actual_pct.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_statistical_tests(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Runs KS test, PSI, and Wasserstein distance for every numerical feature.

    Parameters
    ----------
    train_df       : training dataframe (must not include target column)
    test_df        : test dataframe
    ks_alpha       : p-value threshold for KS test (default 0.05)
    psi_threshold  : PSI threshold for drift flag (default 0.1)

    Returns
    -------
    dict with keys:
        results          : pd.DataFrame - per-feature scores for all three tests
        drifted_columns  : list[str]    - columns flagged by at least 2 of 3 tests
        drift_scores     : pd.Series    - composite drift score per feature (0–1, higher = more drift)
    """
    feature_cols = [c for c in train_df.columns if c != "target"]
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]

    records = []
    for col in numeric_cols:
        # Ensure float dtype for all calculations
        tr_vals = train_df[col].dropna().to_numpy(dtype=float)
        te_vals = test_df[col].dropna().to_numpy(dtype=float)

        if len(tr_vals) == 0 or len(te_vals) == 0:
            continue


        # KS test: handle both namedtuple and tuple return types
        ks_result = stats.ks_2samp(tr_vals, te_vals)
        if isinstance(ks_result, tuple):
            ks_stat, ks_pval = ks_result
        else:
            ks_stat = getattr(ks_result, 'statistic', None)
            ks_pval = getattr(ks_result, 'pvalue', None)
        # Ensure both are floats, fallback to safe defaults if not
        # Only convert to float if not a tuple (should never be tuple here)
        if isinstance(ks_stat, (int, float, str)):
            try:
                ks_stat = float(ks_stat)
            except Exception:
                ks_stat = 0.0
        else:
            ks_stat = 0.0
        if isinstance(ks_pval, (int, float, str)):
            try:
                ks_pval = float(ks_pval)
            except Exception:
                ks_pval = 1.0
        else:
            ks_pval = 1.0
        ks_flagged = ks_pval < ks_alpha

        # PSI
        psi_score = _psi_single(tr_vals, te_vals)
        psi_flagged = psi_score > psi_threshold

        # Wasserstein (normalised by train std so it's scale-invariant)
        tr_std = np.std(tr_vals)
        emd = wasserstein_distance(tr_vals, te_vals) / (tr_std + 1e-8)
        emd_flagged = emd > 0.2  # >0.2 normalised std units = meaningful shift

        # How many tests flagged this column
        n_flagged = int(ks_flagged) + int(psi_flagged) + int(emd_flagged)

        records.append({
            "column":       col,
            "ks_stat":      round(float(ks_stat), 4),
            "ks_pval":      round(float(ks_pval), 4),
            "ks_flagged":   ks_flagged,
            "psi":          round(float(psi_score), 4),
            "psi_flagged":  psi_flagged,
            "emd":          round(float(emd), 4),
            "emd_flagged":  emd_flagged,
            "n_tests_flagged": n_flagged,
        })

    # ── Categorical features: chi-squared + categorical PSI ─────────────────────
    cat_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(train_df[c])
        and c in test_df.columns
    ]
    for col in cat_cols:
        tr_vals = train_df[col].dropna().astype(str)
        te_vals = test_df[col].dropna().astype(str)
        if len(tr_vals) == 0 or len(te_vals) == 0:
            continue

        all_cats = sorted(set(tr_vals.unique()) | set(te_vals.unique()))
        eps = 1e-6

        tr_counts = tr_vals.value_counts()
        te_counts = te_vals.value_counts()
        tr_pct = np.array([tr_counts.get(c, 0) for c in all_cats], dtype=float) / len(tr_vals) + eps
        te_pct = np.array([te_counts.get(c, 0) for c in all_cats], dtype=float) / len(te_vals) + eps
        tr_pct /= tr_pct.sum()
        te_pct /= te_pct.sum()

        # Chi-squared on raw counts
        tr_cnt = np.array([tr_counts.get(c, 0) for c in all_cats], dtype=float)
        te_cnt = np.array([te_counts.get(c, 0) for c in all_cats], dtype=float)
        contingency = np.vstack([tr_cnt, te_cnt])
        # Suppress chi2 if any expected cell is zero
        try:
            _, chi2_pval, _, _ = chi2_contingency(contingency)
        except Exception:
            chi2_pval = 1.0
        chi2_flagged = chi2_pval < ks_alpha

        # Categorical PSI (bin by unique values)
        cat_psi = float(np.sum((te_pct - tr_pct) * np.log(te_pct / tr_pct)))
        cat_psi_flagged = cat_psi > psi_threshold

        # TV distance as a third signal (analogous to EMD for categoricals)
        tv_dist = float(np.sum(np.abs(tr_pct - te_pct)) / 2)
        tv_flagged = tv_dist > 0.05

        n_flagged = int(chi2_flagged) + int(cat_psi_flagged) + int(tv_flagged)

        records.append({
            "column":          col,
            "ks_stat":         tv_dist,      # reuse ks_stat slot with TV distance
            "ks_pval":         chi2_pval,    # reuse ks_pval slot with chi2 p-value
            "ks_flagged":      chi2_flagged,
            "psi":             round(cat_psi, 4),
            "psi_flagged":     cat_psi_flagged,
            "emd":             round(tv_dist, 4),
            "emd_flagged":     tv_flagged,
            "n_tests_flagged": n_flagged,
        })

    if not records:
        return {"results": pd.DataFrame(), "drifted_columns": [], "drift_scores": pd.Series(dtype=float)}

    results = pd.DataFrame(records).sort_values("n_tests_flagged", ascending=False)

    # Composite drift score: normalised sum of the three statistics
    # Each stat is min-max scaled to [0,1] then averaged
    def _minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-8)

    composite = (
        _minmax(results["ks_stat"]) +
        _minmax(results["psi"])     +
        _minmax(results["emd"])
    ) / 3.0

    results["drift_score"] = composite.values
    results = results.sort_values("drift_score", ascending=False).reset_index(drop=True)
    drift_scores = results.set_index("column")["drift_score"]

    # Flag column if at least 2 of 3 tests agree
    drifted_columns = results.loc[results["n_tests_flagged"] >= 2, "column"].tolist()

    # ── Per-feature shift type classification ─────────────────────────────────
    # Each feature gets its own shift_type label based on its own signals.
    # This is the contract for per-feature targeted mitigation.
    #
    # Numeric signals:
    #   emd > 0.3                        → covariate_shape  (full distribution shift)
    #   ks_flagged and emd < 0.1         → covariate_location (mean/median shift only)
    #   default numeric                  → covariate_shape
    # Categorical signals:
    #   any flagged categorical          → covariate_categorical
    # Not flagged (n_tests_flagged < 2)  → none
    def _feature_shift_type(row):
        if row["n_tests_flagged"] < 2:
            return "none"
        # Detect if categorical: ks_pval slot holds chi2 p-value and emd == ks_stat for cats
        # We use the original dtype from train_df to distinguish
        col = row["column"]
        if col in train_df.columns and not pd.api.types.is_numeric_dtype(train_df[col]):
            return "covariate_categorical"
        emd = float(row["emd"])
        ks_flagged = bool(row["ks_flagged"])
        if emd > 0.3:
            return "covariate_shape"
        elif ks_flagged and emd < 0.1:
            return "covariate_location"
        return "covariate_shape"

    results["feature_shift_type"] = results.apply(_feature_shift_type, axis=1)

    if verbose:
        n_total = len(numeric_cols) + len(cat_cols)
        print(f"  [StatTests] {len(drifted_columns)}/{n_total} features flagged (≥2 tests agree, {len(numeric_cols)} numeric + {len(cat_cols)} categorical)")
        if drifted_columns:
            top = results.head(5)[["column", "ks_pval", "psi", "emd", "drift_score"]]
            print(f"  [StatTests] Top drifted features:\n{top.to_string(index=False)}")

    return {
        "results":         results,
        "drifted_columns": drifted_columns,
        "drift_scores":    drift_scores,
    }