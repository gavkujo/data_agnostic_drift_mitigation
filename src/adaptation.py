"""Phase 3+4: Temporal weighting, domain AUC, iterative self-training."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import sample_idx, stratified_sample_idx, FIXED_PARAMS, TRAIN_CAP, DOMAIN_AUC_SAMPLE


def compute_temporal_weights(month_train, y_train):
    """Recency weights with data-driven decay. Only used when concept drift detected."""
    valid = ~month_train.isna()
    if not valid.any():
        return np.ones(len(month_train), dtype=np.float64)

    ordinals = month_train[valid].astype("int64").astype(np.float64)
    max_ord = ordinals.max()
    delta = np.zeros(len(month_train), dtype=np.float64)
    delta[valid.to_numpy()] = max_ord - ordinals.to_numpy()
    delta[~valid.to_numpy()] = float(np.median(delta[valid.to_numpy()]))

    uniq_months = sorted(month_train[valid].unique())
    if len(uniq_months) < 3:
        return np.ones(len(month_train), dtype=np.float64)

    val_months = set(uniq_months[-2:])
    val_mask = month_train.isin(val_months).to_numpy()
    core_mask = ~val_mask & valid.to_numpy()
    if core_mask.sum() < 100 or val_mask.sum() < 100:
        return np.ones(len(month_train), dtype=np.float64)

    best_decay, best_score = 0.0, -1.0
    for decay in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        w = np.exp(-decay * delta)
        w = w / (w.mean() + 1e-8)
        weighted_pos = np.average(y_train[core_mask], weights=w[core_mask])
        val_pos = y_train[val_mask].mean()
        score = -abs(weighted_pos - val_pos)
        if score > best_score:
            best_score, best_decay = score, decay

    w = np.exp(-best_decay * delta)
    return w / (w.mean() + 1e-8)


def compute_domain_auc(train_df, test_df, features, feat_types):
    """Domain AUC on case-normalised features for drift gating."""
    n_tr = min(len(train_df), DOMAIN_AUC_SAMPLE // 2)
    n_te = min(len(test_df), DOMAIN_AUC_SAMPLE - n_tr)
    parts = []
    for df, n in [(train_df, n_tr), (test_df, n_te)]:
        idx = sample_idx(len(df), n)
        x = pd.DataFrame(index=range(len(idx)))
        for col in features:
            vals = df.iloc[idx][col]
            if feat_types[col] == "numeric":
                x[col] = pd.to_numeric(vals, errors="coerce").fillna(0).astype(np.float32).values
            else:
                x[col] = (vals.fillna("__MISSING__").astype(str).str.lower().str.strip()
                          .astype("category").cat.codes.astype(np.float32).values)
        parts.append(x)
    x_adv = pd.concat(parts, ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(parts[0])), np.ones(len(parts[1]))])
    m = lgb.LGBMClassifier(**FIXED_PARAMS)
    m.fit(x_adv, y_adv)
    return float(roc_auc_score(y_adv, m.predict_proba(x_adv)[:, 1]))


def _est_fit_time(x, y):
    n = min(5000, len(x))
    idx = sample_idx(len(x), n)
    t0 = time.time()
    lgb.LGBMClassifier(**FIXED_PARAMS).fit(x.iloc[idx], y[idx])
    return (time.time() - t0) * (min(len(x), TRAIN_CAP) / n) * 1.5


def train_and_adapt(x_tr, x_te, y_train, budget, drift_detected, sample_weights=None):
    """Single LightGBM + iterative self-training if drift detected."""
    n = len(x_tr)
    if n > TRAIN_CAP:
        tidx = stratified_sample_idx(y_train, TRAIN_CAP)
        x_fit, y_fit = x_tr.iloc[tidx], y_train[tidx]
        w_fit = sample_weights[tidx] if sample_weights is not None else None
    else:
        x_fit, y_fit = x_tr, y_train
        w_fit = sample_weights
        tidx = None

    model = lgb.LGBMClassifier(**FIXED_PARAMS)
    model.fit(x_fit, y_fit, sample_weight=w_fit)
    scores = model.predict_proba(x_te)[:, 1].astype(np.float64)

    if not drift_detected:
        return scores, model

    baseline_train = float(average_precision_score(y_fit, model.predict_proba(x_fit)[:, 1]))
    threshold = 0.85
    est = _est_fit_time(x_fit, y_fit)
    max_rounds = min(15, max(1, int(budget.remaining() / max(est, 1)) - 1))
    all_scores = [scores.copy()]

    for r in range(max_rounds):
        if not budget.can_afford(est):
            break
        pp = scores >= threshold
        pn = scores <= (1 - threshold)
        pm = pp | pn
        if pm.sum() < 10:
            break

        pseudo_idx = pm.nonzero()[0]
        max_pseudo = TRAIN_CAP - min(n, TRAIN_CAP)
        if len(pseudo_idx) > max_pseudo > 0:
            pseudo_idx = pseudo_idx[sample_idx(len(pseudo_idx), max_pseudo)]

        x_pseudo = x_te.iloc[pseudo_idx]
        y_pseudo = (scores[pseudo_idx] >= threshold).astype(np.int32)

        if n > TRAIN_CAP:
            train_budget = TRAIN_CAP - len(pseudo_idx)
            tr_idx = stratified_sample_idx(y_train, max(1000, train_budget))
            xf = pd.concat([x_tr.iloc[tr_idx], x_pseudo], ignore_index=True)
            yf = np.concatenate([y_train[tr_idx], y_pseudo])
            wf = np.concatenate([sample_weights[tr_idx], np.ones(len(y_pseudo))]) if sample_weights is not None else None
        else:
            xf = pd.concat([x_tr, x_pseudo], ignore_index=True)
            yf = np.concatenate([y_train, y_pseudo])
            wf = np.concatenate([sample_weights, np.ones(len(y_pseudo))]) if sample_weights is not None else None
            if len(xf) > TRAIN_CAP:
                cidx = stratified_sample_idx(yf, TRAIN_CAP)
                xf, yf = xf.iloc[cidx], yf[cidx]
                wf = wf[cidx] if wf is not None else None

        model = lgb.LGBMClassifier(**FIXED_PARAMS)
        model.fit(xf, yf, sample_weight=wf)
        scores = model.predict_proba(x_te)[:, 1].astype(np.float64)
        all_scores.append(scores.copy())

        # Safety check
        if tidx is not None:
            check = model.predict_proba(x_tr.iloc[tidx])[:, 1]
            train_auprc = float(average_precision_score(y_train[tidx], check))
        else:
            train_auprc = float(average_precision_score(y_train, model.predict_proba(x_tr)[:, 1]))
        if baseline_train - train_auprc > 0.10:
            break

    n_avg = min(5, len(all_scores))
    return np.mean(all_scores[-n_avg:], axis=0), model
