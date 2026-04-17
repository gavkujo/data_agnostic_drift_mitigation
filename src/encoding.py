"""Phase 2: Per-feature encoding and mitigation — quantile mapping, target encoding, dropping."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import sample_idx, ENCODE_SAMPLE, DRIFT_SAMPLE


def encode_and_mitigate(train_df, test_df, features, y_train, diag_results, feat_types):
    """Apply per-feature mitigation based on drift diagnosis. Returns encoded train/test + drift rows."""
    x_tr = pd.DataFrame(index=train_df.index)
    x_te = pd.DataFrame(index=test_df.index)
    drift_rows = []
    n_train = len(train_df)
    use_kfold = n_train < 100_000
    enc_idx = sample_idx(n_train, ENCODE_SAMPLE)
    te_samp_n = min(DRIFT_SAMPLE, len(test_df))
    gm = float(y_train.mean())
    alpha = 10.0

    for col in features:
        info = diag_results[col]
        mitigation = info["mitigation"]

        if mitigation == "drop":
            drift_rows.append({"Column Name": col, "Column Type": info["feat_type"],
                "Drift Description": f"Severe mixed drift (type={info['drift_type']}, effect={info['effect']:.4f})",
                "Drift Mitigation": "Dropped: too unstable to mitigate"})
            continue

        if info["feat_type"] == "numeric":
            x_tr, x_te, drift_rows = _encode_numeric(
                x_tr, x_te, drift_rows, train_df, test_df, col, info, mitigation, enc_idx, te_samp_n)
        else:
            x_tr, x_te, drift_rows = _encode_categorical(
                x_tr, x_te, drift_rows, train_df, test_df, col, info, y_train, enc_idx, gm, alpha, use_kfold)

    return x_tr, x_te, drift_rows


def _encode_numeric(x_tr, x_te, drift_rows, train_df, test_df, col, info, mitigation, enc_idx, te_samp_n):
    tr = pd.to_numeric(train_df[col], errors="coerce")
    te = pd.to_numeric(test_df[col], errors="coerce")
    fill = float(tr.median()) if tr.notna().any() else 0.0
    tr_f = tr.fillna(fill).to_numpy(np.float32)
    te_f = te.fillna(fill).to_numpy(np.float32)

    subtype = info.get("drift_subtype", "general")
    applied_mitigation = "No transform"

    if mitigation == "quantile_map":
        tr_samp = tr_f[enc_idx].astype(np.float64)
        te_samp = te_f[:te_samp_n].astype(np.float64)
        tr_v = tr_samp[np.isfinite(tr_samp)]
        te_v = te_samp[np.isfinite(te_samp)]
        if tr_v.size >= 2 and te_v.size >= 2:
            g = np.linspace(0, 100, 101)
            qt, qe = np.percentile(tr_v, g), np.percentile(te_v, g)
            mask = np.concatenate(([True], np.diff(qt) > 1e-12))
            qx, qy = qt[mask], qe[mask]
            if qx.size >= 2:
                tr_f = np.interp(tr_f.astype(np.float64), qx, qy).astype(np.float32)
        applied_mitigation = f"Quantile mapping ({subtype})"

    elif mitigation == "winsorize":
        # Clip test values to train's 1st/99th percentile range
        tr_valid = tr_f[np.isfinite(tr_f)]
        if tr_valid.size >= 2:
            lo = float(np.percentile(tr_valid, 1))
            hi = float(np.percentile(tr_valid, 99))
            te_f = np.clip(te_f, lo, hi).astype(np.float32)
        applied_mitigation = f"Winsorization p1/p99 ({subtype})"

    elif mitigation == "clip":
        # Clip test values to train min/max
        tr_valid = tr_f[np.isfinite(tr_f)]
        if tr_valid.size >= 2:
            te_f = np.clip(te_f, tr_valid.min(), tr_valid.max()).astype(np.float32)
        applied_mitigation = f"Clipping to train range ({subtype})"

    if mitigation in ("quantile_map", "winsorize", "clip"):
        ks_s = f"KS={info.get('ks',0):.4f}"
        psi_s = f"PSI={info.get('psi',0):.4f}"
        wd_s = f"WD={info.get('wasserstein_norm',0):.3f}"
        js_s = f"JS={info.get('js',0):.4f}"
        drift_rows.append({"Column Name": col, "Column Type": "numeric",
            "Drift Description": f"{info['drift_type']} drift: {subtype} ({ks_s}, {psi_s}, {wd_s}, {js_s})",
            "Drift Mitigation": applied_mitigation})
    elif info.get("dist_shifted"):
        drift_rows.append({"Column Name": col, "Column Type": "numeric",
            "Drift Description": f"{info['drift_type']} drift (KS={info.get('ks',0):.4f}, PSI={info.get('psi',0):.4f}, small effect)",
            "Drift Mitigation": "No transform (effect below median)"})

    x_tr[col] = tr_f
    x_te[col] = te_f
    return x_tr, x_te, drift_rows


def _encode_categorical(x_tr, x_te, drift_rows, train_df, test_df, col, info, y_train, enc_idx, gm, alpha, use_kfold):
    tr_cat = train_df[col].fillna("__MISSING__").astype(str).str.lower().str.strip()
    te_cat = test_df[col].fillna("__MISSING__").astype(str).str.lower().str.strip()
    samp_cats = tr_cat.values[enc_idx]
    samp_y = y_train[enc_idx]
    df_t = pd.DataFrame({"c": samp_cats, "y": samp_y})
    st = df_t.groupby("c")["y"].agg(["mean", "count"])
    sm = (st["count"] * st["mean"] + alpha * gm) / (st["count"] + alpha)

    if use_kfold:
        tr_enc = np.full(len(tr_cat), gm, dtype=np.float64)
        fi = np.arange(len(tr_cat)) % 5
        for fold in range(5):
            fm = fi == fold
            oof = pd.DataFrame({"c": tr_cat.values[~fm], "y": y_train[~fm]})
            os = oof.groupby("c")["y"].agg(["mean", "count"])
            osm = (os["count"] * os["mean"] + alpha * gm) / (os["count"] + alpha)
            tr_enc[fm] = pd.Series(tr_cat.values[fm]).map(osm).fillna(gm).to_numpy()
    else:
        tr_enc = tr_cat.map(sm).fillna(gm).to_numpy(np.float64)

    te_enc = te_cat.map(sm).fillna(gm).to_numpy(np.float64)
    x_tr[col] = tr_enc.astype(np.float32)
    x_te[col] = te_enc.astype(np.float32)

    if info.get("dist_shifted") or info.get("new_cats") or info.get("concept_drift"):
        reasons = []
        if info.get("new_cats"): reasons.append("new categories")
        if info.get("dist_shifted"): reasons.append(f"proportion shift (TVD={info.get('tvd',0):.4f}, JS={info.get('js',0):.4f})")
        if info.get("concept_drift"): reasons.append("concept drift (target relationship changed)")
        drift_rows.append({"Column Name": col, "Column Type": "categorical",
            "Drift Description": f"{info['drift_type']} drift: {'; '.join(reasons)}",
            "Drift Mitigation": "Target encoding with case normalization"})

    return x_tr, x_te, drift_rows
