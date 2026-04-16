"""Phase 1: Per-feature drift diagnostics — distribution shift, concept drift, format changes."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from utils import sample_idx, ncdf, DRIFT_SAMPLE


def diagnose_features(train_df, test_df, features, feat_types, y_train, month_train):
    """
    Per-feature diagnostic battery. Returns:
      results: dict[col] -> diagnostic info including drift_type and mitigation
      eff_med: median effect size among significant numeric features
      any_concept_drift: whether any feature has concept drift
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

    # BH correction + effect size filter + drift type classification
    results, eff_med, any_concept = _classify_all(results, pvalues, features)
    return results, eff_med, any_concept


def _diagnose_numeric(info, col, tr_s, te_s, train_df, test_df):
    tr = pd.to_numeric(tr_s[col], errors="coerce").dropna().to_numpy(np.float64)
    te = pd.to_numeric(te_s[col], errors="coerce").dropna().to_numpy(np.float64)
    if tr.size < 2 or te.size < 2:
        info.update({"ks": 0.0, "p": 1.0, "effect": 0.0, "new_cats": False,
                     "missing_change": 0.0, "concept_drift": False})
        return info, 1.0
    combined = np.sort(np.concatenate([np.sort(tr), np.sort(te)]))
    cdf_x = np.searchsorted(np.sort(tr), combined, side="right") / len(tr)
    cdf_y = np.searchsorted(np.sort(te), combined, side="right") / len(te)
    ks = float(np.max(np.abs(cdf_x - cdf_y)))
    en = (len(tr) * len(te)) / max(len(tr) + len(te), 1)
    p = min(1.0, max(0.0, 2.0 * math.exp(-2.0 * en * ks ** 2)))
    info.update({"ks": ks, "p": p, "effect": ks, "new_cats": False,
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


def _classify_all(results, pvalues, features):
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    k_max = 0
    for i, (_, p) in enumerate(ordered, start=1):
        if p <= 0.05 * i / m: k_max = i
    sig = {name for name, _ in ordered[:k_max]} if k_max > 0 else set()

    sig_num_eff = [results[c]["effect"] for c in sig if results[c]["feat_type"] == "numeric"]
    eff_med = float(np.median(sig_num_eff)) if sig_num_eff else 0.0

    any_concept = False
    for col in features:
        info = results[col]
        ds = col in sig
        concept = info.get("concept_drift", False)
        new_cats = info.get("new_cats", False)
        is_num = info["feat_type"] == "numeric"
        large = ds and info["effect"] >= eff_med if is_num else ds
        info["dist_shifted"] = ds

        # Drift type
        if not ds and not concept and not new_cats: info["drift_type"] = "none"
        elif ds and not concept and not new_cats: info["drift_type"] = "covariate"
        elif ds and not concept and new_cats: info["drift_type"] = "covariate_format"
        elif not ds and concept: info["drift_type"] = "concept"
        elif ds and concept: info["drift_type"] = "mixed"
        elif new_cats and not ds: info["drift_type"] = "format_only"
        else: info["drift_type"] = "unknown"

        # Mitigation
        if is_num:
            if info["drift_type"] == "none": info["mitigation"] = "keep_raw"
            elif info["drift_type"] in ("covariate", "covariate_format") and large: info["mitigation"] = "quantile_map"
            elif info["drift_type"] in ("concept", "mixed"):
                info["mitigation"] = "quantile_map" if large else "keep_raw"
                any_concept = True
            else: info["mitigation"] = "keep_raw"
        else:
            if info["drift_type"] == "none": info["mitigation"] = "target_encode"
            elif info["drift_type"] in ("mixed",) and new_cats and info.get("tvd", 0) > 0.8:
                info["mitigation"] = "drop"
            else: info["mitigation"] = "target_encode"

        if concept: any_concept = True

    return results, eff_med, any_concept
