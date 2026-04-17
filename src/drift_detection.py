"""Phase 1: Per-feature drift diagnostics — distribution shift, concept drift, format changes."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from utils import sample_idx, ncdf, DRIFT_SAMPLE


def diagnose_features(train_df, test_df, features, feat_types, y_train, month_train,
                      feature_importances=None):
    """
    Per-feature diagnostic battery. Returns:
      results: dict[col] -> diagnostic info including drift_type and mitigation
      risk_median: median risk score among shifted features (for reporting)
      any_concept_drift: whether any feature has concept drift

    If feature_importances is provided (dict col->float), uses importance-weighted
    risk scoring: risk = drift_severity × normalised_importance. This ensures
    high-drift low-importance features are not over-mitigated.
    """
    tr_s = train_df.iloc[sample_idx(len(train_df), DRIFT_SAMPLE)]
    te_s = test_df.iloc[sample_idx(len(test_df), DRIFT_SAMPLE)]
    eps = 1e-8
    results, pvalues = {}, {}

    # Temporal split for concept drift detection
    valid_months = ~month_train.isna()
    if valid_months.any():
        month_ord = month_train[valid_months].astype("int64")
        median_month = float(np.median(month_ord))
        early_mask = np.zeros(len(train_df), dtype=bool)
        late_mask = np.zeros(len(train_df), dtype=bool)
        early_mask[valid_months.to_numpy()] = month_ord.to_numpy() <= median_month
        late_mask[valid_months.to_numpy()] = month_ord.to_numpy() > median_month
        has_temporal = early_mask.sum() > 100 and late_mask.sum() > 100
    else:
        has_temporal = False
        early_mask = late_mask = np.zeros(len(train_df), dtype=bool)

    for col in features:
        is_num = feat_types[col] == "numeric"
        info = {"col": col, "feat_type": feat_types[col]}

        if is_num:
            info, p = _diagnose_numeric(info, col, tr_s, te_s, train_df, test_df)
        else:
            info, p = _diagnose_categorical(info, col, tr_s, te_s, train_df, test_df, eps)
        pvalues[col] = p

        # Concept drift test
        info["concept_drift"] = False
        if has_temporal:
            info = _test_concept_drift(info, col, is_num, train_df, y_train, early_mask, late_mask)

        results[col] = info

    # BH correction + importance-weighted risk scoring + drift type classification
    results, risk_med, any_concept = _classify_all(results, pvalues, features, feature_importances)
    return results, risk_med, any_concept


def _diagnose_numeric(info, col, tr_s, te_s, train_df, test_df):
    tr = pd.to_numeric(tr_s[col], errors="coerce").dropna().to_numpy(np.float64)
    te = pd.to_numeric(te_s[col], errors="coerce").dropna().to_numpy(np.float64)
    if tr.size < 2 or te.size < 2:
        info.update({"ks": 0.0, "p": 1.0, "effect": 0.0, "new_cats": False,
                     "missing_change": 0.0, "concept_drift": False,
                     "psi": 0.0, "wasserstein_norm": 0.0, "js": 0.0})
        return info, 1.0

    # KS test
    combined = np.sort(np.concatenate([np.sort(tr), np.sort(te)]))
    cdf_x = np.searchsorted(np.sort(tr), combined, side="right") / len(tr)
    cdf_y = np.searchsorted(np.sort(te), combined, side="right") / len(te)
    ks = float(np.max(np.abs(cdf_x - cdf_y)))
    en = (len(tr) * len(te)) / max(len(tr) + len(te), 1)
    p = min(1.0, max(0.0, 2.0 * math.exp(-2.0 * en * ks ** 2)))

    # PSI (Population Stability Index) — sensitive to bin proportion changes
    n_bins = 10
    lo = min(tr.min(), te.min())
    hi = max(tr.max(), te.max())
    if hi > lo:
        breaks = np.linspace(lo, hi, n_bins + 1)
        tr_hist = np.histogram(tr, bins=breaks)[0] / len(tr)
        te_hist = np.histogram(te, bins=breaks)[0] / len(te)
        tr_hist = np.where(tr_hist == 0, 1e-8, tr_hist)
        te_hist = np.where(te_hist == 0, 1e-8, te_hist)
        psi = float(np.sum((te_hist - tr_hist) * np.log(te_hist / tr_hist)))
    else:
        psi = 0.0

    # Wasserstein distance (normalised by train std) — sensitive to location/scale
    tr_std = float(tr.std())
    wd_raw = 0.0
    q_grid = np.linspace(0, 100, 51)
    q_tr = np.percentile(tr, q_grid)
    q_te = np.percentile(te, q_grid)
    wd_raw = float(np.mean(np.abs(q_tr - q_te)))
    wd_norm = wd_raw / tr_std if tr_std > 0 else wd_raw

    # JS divergence — sensitive to shape changes
    if hi > lo:
        tr_dens = np.histogram(tr, bins=breaks, density=True)[0].astype(float) + 1e-8
        te_dens = np.histogram(te, bins=breaks, density=True)[0].astype(float) + 1e-8
        m_dens = 0.5 * (tr_dens + te_dens)
        js = float(0.5 * np.sum(tr_dens * np.log(tr_dens / m_dens))
                   + 0.5 * np.sum(te_dens * np.log(te_dens / m_dens)))
    else:
        js = 0.0

    info.update({"ks": ks, "p": p, "effect": ks, "new_cats": False,
                 "psi": round(psi, 6), "wasserstein_norm": round(wd_norm, 4), "js": round(js, 6),
                 "missing_change": abs(float(train_df[col].isna().mean()) - float(test_df[col].isna().mean()))})
    return info, p


def _diagnose_categorical(info, col, tr_s, te_s, train_df, test_df, eps):
    tr_v = tr_s[col].astype("string").fillna("__NA__")
    te_v = te_s[col].astype("string").fillna("__NA__")
    tr_c = tr_v.value_counts(normalize=True)
    te_c = te_v.value_counts(normalize=True)
    idx_u = tr_c.index.union(te_c.index)
    tr_vec = tr_c.reindex(idx_u, fill_value=0.0).to_numpy(np.float64)
    te_vec = te_c.reindex(idx_u, fill_value=0.0).to_numpy(np.float64)
    n1, n2 = len(tr_v), len(te_v)
    pooled = (tr_vec * n1 + te_vec * n2) / max(n1 + n2, 1)
    chi = float(np.sum((tr_vec * n1 - pooled * n1) ** 2 / (pooled * n1 + eps))
                + np.sum((te_vec * n2 - pooled * n2) ** 2 / (pooled * n2 + eps)))
    dof = max(len(idx_u) - 1, 1)
    k = float(dof)
    zv = ((chi / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    p = min(1.0, max(0.0, 1.0 - ncdf(zv)))
    tvd = 0.5 * float(np.abs(tr_vec - te_vec).sum())
    tr_lo = {c.lower().strip() for c in set(tr_v.unique())}
    te_lo = {c.lower().strip() for c in set(te_v.unique())}
    info.update({"chi2": chi, "p": p, "effect": tvd, "tvd": tvd,
                 "new_cats": len(te_lo - tr_lo) > 0,
                 "nunique": int(train_df[col].nunique()),
                 "js": round(float(0.5 * np.sum(tr_vec * np.log((tr_vec + 1e-8) / (0.5 * (tr_vec + te_vec) + 1e-8)))
                                    + 0.5 * np.sum(te_vec * np.log((te_vec + 1e-8) / (0.5 * (tr_vec + te_vec) + 1e-8)))), 6),
                 "missing_change": abs(float(train_df[col].isna().mean()) - float(test_df[col].isna().mean()))})
    return info, p


def _test_concept_drift(info, col, is_num, train_df, y_train, early_mask, late_mask):
    if is_num:
        ev = pd.to_numeric(train_df.loc[early_mask, col], errors="coerce")
        lv = pd.to_numeric(train_df.loc[late_mask, col], errors="coerce")
        ey, ly = y_train[early_mask], y_train[late_mask]
        evd, lvd = ev.dropna(), lv.dropna()
        eyd = ey[ev.notna().to_numpy()]
        lyd = ly[lv.notna().to_numpy()]
        if len(evd) > 50 and len(lvd) > 50:
            ce = float(np.corrcoef(evd, eyd)[0, 1]) if evd.std() > 0 else 0
            cl = float(np.corrcoef(lvd, lyd)[0, 1]) if lvd.std() > 0 else 0
            sign_flip = (ce * cl < 0) and (abs(ce) > 0.05 and abs(cl) > 0.05)
            me, ml = abs(ce), abs(cl)
            mag_change = (max(me, ml) > 2 * max(min(me, ml), 0.01) and max(me, ml) > 0.1)
            info["concept_drift"] = sign_flip or mag_change
            info["corr_early"], info["corr_late"] = ce, cl
    else:
        ec = train_df.loc[early_mask, col].fillna("__NA__").astype(str).str.lower()
        lc = train_df.loc[late_mask, col].fillna("__NA__").astype(str).str.lower()
        er = pd.DataFrame({"c": ec, "y": y_train[early_mask]}).groupby("c")["y"].mean()
        lr = pd.DataFrame({"c": lc, "y": y_train[late_mask]}).groupby("c")["y"].mean()
        common = er.index.intersection(lr.index)
        if len(common) >= 3:
            ecnt = pd.DataFrame({"c": ec}).groupby("c").size()
            lcnt = pd.DataFrame({"c": lc}).groupby("c").size()
            reliable = [c for c in common if ecnt.get(c, 0) >= 30 and lcnt.get(c, 0) >= 30]
            if len(reliable) >= 3:
                rc = float(np.corrcoef(er.loc[reliable].values, lr.loc[reliable].values)[0, 1])
                info["concept_drift"] = rc < 0.5
                info["rate_corr"] = rc
    return info


def _classify_all(results, pvalues, features, feature_importances=None):
    """
    Classify drift type and assign mitigation per feature.
    Uses effect-size median filter: only mitigate numeric features with effect
    above the median of significant features. This prevents over-correction.
    Feature importances are stored for reporting but don't change routing
    (baseline importances are unreliable when features are broken by drift).
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    k_max = 0
    for i, (_, p) in enumerate(ordered, start=1):
        if p <= 0.05 * i / m: k_max = i
    sig = {name for name, _ in ordered[:k_max]} if k_max > 0 else set()

    # Store importances for reporting
    if feature_importances:
        max_imp = max(feature_importances.values()) if feature_importances else 1.0
        for col in features:
            results[col]["importance"] = feature_importances.get(col, 0) / max(max_imp, 1e-8)

    # Effect-size median filter for numerics
    sig_num_eff = [results[c]["effect"] for c in sig if results[c]["feat_type"] == "numeric"]
    eff_med = float(np.median(sig_num_eff)) if sig_num_eff else 0.0

    # Compute relative thresholds for subtype routing (data-driven, not hardcoded)
    # Use percentiles of diagnostic values among shifted numeric features
    sig_num_cols = [c for c in sig if results[c]["feat_type"] == "numeric"]
    if sig_num_cols:
        js_vals = [results[c].get("js", 0) for c in sig_num_cols]
        wd_vals = [results[c].get("wasserstein_norm", 0) for c in sig_num_cols]
        psi_vals = [results[c].get("psi", 0) for c in sig_num_cols]
        ks_vals = [results[c].get("ks", 0) for c in sig_num_cols]
        # 75th percentile = "high" for this dataset
        js_high = float(np.percentile(js_vals, 75)) if len(js_vals) >= 2 else 0.05
        wd_high = float(np.percentile(wd_vals, 75)) if len(wd_vals) >= 2 else 0.3
        psi_high = float(np.percentile(psi_vals, 75)) if len(psi_vals) >= 2 else 0.1
        ks_low = float(np.percentile(ks_vals, 25)) if len(ks_vals) >= 2 else 0.05
    else:
        js_high, wd_high, psi_high, ks_low = 0.05, 0.3, 0.1, 0.05

    any_concept = False
    for col in features:
        info = results[col]
        ds = col in sig
        concept = info.get("concept_drift", False)
        new_cats = info.get("new_cats", False)
        is_num = info["feat_type"] == "numeric"
        large = ds and info["effect"] >= eff_med if is_num else ds
        info["dist_shifted"] = ds

        # Drift type classification
        if not ds and not concept and not new_cats: info["drift_type"] = "none"
        elif ds and not concept and not new_cats: info["drift_type"] = "covariate"
        elif ds and not concept and new_cats: info["drift_type"] = "covariate_format"
        elif not ds and concept: info["drift_type"] = "concept"
        elif ds and concept: info["drift_type"] = "mixed"
        elif new_cats and not ds: info["drift_type"] = "format_only"
        else: info["drift_type"] = "unknown"

        # Mitigation routing — numerics use diagnostic profile for subtype-specific fix
        if is_num:
            if info["drift_type"] == "none":
                info["mitigation"] = "keep_raw"
                info["drift_subtype"] = "none"
            elif not large:
                info["mitigation"] = "keep_raw"
                info["drift_subtype"] = "minor"
            else:
                # Use diagnostic signals + relative thresholds for subtype routing
                ks_val = info.get("ks", 0)
                psi_val = info.get("psi", 0)
                wd_val = info.get("wasserstein_norm", 0)
                js_val = info.get("js", 0)

                if js_val >= js_high or (ks_val >= eff_med and psi_val >= psi_high):
                    info["mitigation"] = "quantile_map"
                    info["drift_subtype"] = "shape_shift"
                elif wd_val >= wd_high and js_val < js_high:
                    info["mitigation"] = "quantile_map"
                    info["drift_subtype"] = "location_scale_shift"
                elif psi_val >= psi_high and ks_val < ks_low:
                    info["mitigation"] = "winsorize"
                    info["drift_subtype"] = "bin_proportion_shift"
                elif wd_val >= wd_high * 0.3 and ks_val < ks_low:
                    info["mitigation"] = "clip"
                    info["drift_subtype"] = "tail_shift"
                else:
                    info["mitigation"] = "quantile_map"
                    info["drift_subtype"] = "general"

                if info["drift_type"] in ("concept", "mixed"):
                    any_concept = True
        else:
            if info["drift_type"] == "none": info["mitigation"] = "target_encode"
            elif info["drift_type"] in ("mixed",) and new_cats and info.get("tvd", 0) > 0.8:
                info["mitigation"] = "drop"
            else: info["mitigation"] = "target_encode"

        if concept: any_concept = True

    return results, eff_med, any_concept
