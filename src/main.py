"""
NAISC 2026 — Drift Detection & Mitigation Pipeline

Entry point. Orchestrates: diagnose → encode → adapt → output.
See individual modules for implementation details:
  utils.py           — constants, Budget, sampling, stats
  drift_detection.py — Phase 1: per-feature diagnostics
  encoding.py        — Phase 2: feature encoding & mitigation
  adaptation.py      — Phase 3+4: temporal weighting, domain AUC, self-training
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score

from utils import (ID_COL, TIME_COL, TARGET_COL, FIXED_PARAMS, TRAIN_CAP,
                   Budget, sample_idx, parse_months, classify_features)
from drift_detection import diagnose_features
from encoding import encode_and_mitigate
from adaptation import compute_domain_auc, compute_temporal_weights, train_and_adapt


# ---------------------------------------------------------------------------
# Console output helpers (competition-compliant ASCII tables)
# ---------------------------------------------------------------------------

def _print_drift_table(all_drift_rows, n_no_drift):
    if all_drift_rows:
        headers = ["Columns with Drift", "Column Type", "Drift Description", "Drift Mitigation"]
        rows = [(r["Column Name"], r["Column Type"], r["Drift Description"], r["Drift Mitigation"])
                for r in all_drift_rows]
        widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
        print(f"\n{sep}\n{hdr}\n{sep}")
        for row in rows:
            print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
        print(sep)
    else:
        print("\n+--------------------+-------------+-------------------------------+----------------------+")
        print("| Columns with Drift | Column Type | Drift Description             | Drift Mitigation     |")
        print("+--------------------+-------------+-------------------------------+----------------------+")
        print("| NA                 | NA          | No significant drift detected | No mitigation applied|")
        print("+--------------------+-------------+-------------------------------+----------------------+")
    if n_no_drift > 0:
        print(f"\n({n_no_drift} additional features showed no significant drift "
              f"— BH-corrected alpha=0.05, tested via KS/chi-square)")


def _print_boxed(label, value):
    w = max(len(label), len(value))
    sep = f"+{'-' * (w + 2)}+"
    print(f"\n{sep}\n| {label.ljust(w)} |\n{sep}\n| {value.ljust(w)} |\n{sep}")


def _print_auprc(atr_s, ate_s):
    w1 = max(len("Train Set"), len("Test Set"))
    w2 = max(len("AU-PRC"), len(atr_s), len(ate_s))
    sep = f"+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+"
    print(f"\n{sep}")
    print(f"| {''.ljust(w1)} | {'AU-PRC'.ljust(w2)} |")
    print(sep)
    print(f"| {'Train Set'.ljust(w1)} | {atr_s.ljust(w2)} |")
    print(sep)
    print(f"| {'Test Set'.ljust(w1)} | {ate_s.ljust(w2)} |")
    print(sep)


def _print_predictions(pred_df):
    head = pred_df.head(5)
    w_id = max(len(ID_COL), head[ID_COL].astype(str).str.len().max())
    w_ps = max(len("probability_score"), 5)
    sep = f"+{'-' * (w_id + 2)}+{'-' * (w_ps + 2)}+"
    print(f"\n{sep}")
    print(f"| {ID_COL.ljust(w_id)} | {'probability_score'.ljust(w_ps)} |")
    print(sep)
    for _, row in head.iterrows():
        ps = f"{row['probability_score']:.3f}"
        print(f"| {str(row[ID_COL]).ljust(w_id)} | {ps.ljust(w_ps)} |")
        print(sep)


# ---------------------------------------------------------------------------
# Fallback (must always produce outputs)
# ---------------------------------------------------------------------------

def _fallback(train_path, test_path, start_time, err):
    import lightgbm as lgb
    print(f"[FALLBACK] {err}")
    try:
        train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
        feats = [c for c in train_df.columns if c not in {ID_COL, TIME_COL, TARGET_COL} and c in test_df.columns]
        x_tr, x_te = pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)
        for col in feats:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                tr = pd.to_numeric(train_df[col], errors="coerce")
                fill = float(tr.median()) if tr.notna().any() else 0.0
                x_tr[col] = tr.fillna(fill).astype(np.float32)
                x_te[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(fill).astype(np.float32)
            else:
                av = pd.concat([train_df[col].fillna("__NA__").astype(str),
                                test_df[col].fillna("__NA__").astype(str)], ignore_index=True).astype("category")
                cd = av.cat.codes.to_numpy(np.int32)
                x_tr[col], x_te[col] = cd[:len(train_df)].astype(np.float32), cd[len(train_df):].astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(train_df[TARGET_COL].astype(str))
        m = lgb.LGBMClassifier(**FIXED_PARAMS); m.fit(x_tr, y)
        te_s = m.predict_proba(x_te)[:, 1]
        pd.DataFrame({ID_COL: test_df[ID_COL], "probability_score": te_s}).to_csv("prediction.csv", index=False)
        joblib.dump(m, "model.joblib")
        _print_boxed("Time Taken (s)", f"{time.time()-start_time:.1f}")
    except Exception:
        pd.DataFrame({ID_COL: ["unknown"], "probability_score": [0.5]}).to_csv("prediction.csv", index=False)
        joblib.dump({"fallback": True}, "model.joblib")
        _print_boxed("Time Taken (s)", f"{time.time()-start_time:.1f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(train_path: str, test_path: str) -> None:
    start_time = time.time()
    budget = Budget(limit=600.0)

    try:
        # Load & validate
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        for c in [ID_COL, TIME_COL, TARGET_COL]:
            if c not in train_df.columns: raise KeyError(f"Missing in train: {c}")
        for c in [ID_COL, TIME_COL]:
            if c not in test_df.columns: raise KeyError(f"Missing in test: {c}")
        features = [c for c in train_df.columns
                    if c not in {ID_COL, TIME_COL, TARGET_COL} and c in test_df.columns]
        if not features: raise ValueError("No common features")

        le = LabelEncoder()
        y_train = le.fit_transform(train_df[TARGET_COL].astype(str))
        month_train = parse_months(train_df[TIME_COL])
        print(f"[DATA] Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(features)}")

        # Phase 1: Per-feature diagnostics
        feat_types = classify_features(train_df, features)

        # Quick baseline model for feature importances (used for risk-weighted drift routing)
        import lightgbm as lgb
        x_quick = pd.DataFrame(index=train_df.index)
        for col in features:
            if feat_types[col] == "numeric":
                x_quick[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).astype(np.float32)
            else:
                x_quick[col] = (train_df[col].fillna("__MISSING__").astype(str).str.lower().str.strip()
                                .astype("category").cat.codes.astype(np.float32))
        quick_model = lgb.LGBMClassifier(**FIXED_PARAMS)
        quick_model.fit(x_quick, y_train)
        feat_importances = dict(zip(features, quick_model.feature_importances_.astype(float)))

        diag, risk_med, any_concept = diagnose_features(
            train_df, test_df, features, feat_types, y_train, month_train,
            feature_importances=feat_importances)
        n_shifted = sum(1 for v in diag.values() if v.get("dist_shifted"))
        n_drop = sum(1 for v in diag.values() if v.get("mitigation") == "drop")
        print(f"[DIAG] Shifted: {n_shifted}/{len(features)}, "
              f"Concept: {sum(1 for v in diag.values() if v.get('concept_drift'))}, "
              f"Drop: {n_drop}")

        # Domain AUC for pseudo-labeling gate
        domain_auc = compute_domain_auc(train_df, test_df, features, feat_types)
        drift_on = domain_auc > 0.6
        print(f"[DRIFT] Domain AUC: {domain_auc:.4f} -> pseudo={'ON' if drift_on else 'OFF'}")

        # Phase 2: Encode & mitigate
        active = [c for c in features if diag[c]["mitigation"] != "drop"]
        x_tr, x_te, drift_rows = encode_and_mitigate(
            train_df, test_df, active, y_train, diag, feat_types)
        print(f"[ENCODE] {x_tr.shape[1]} features, {budget.elapsed():.1f}s elapsed")

        # Phase 3: Temporal weighting if concept drift
        temporal_w = compute_temporal_weights(month_train, y_train) if any_concept else None

        # Phase 4: Train + self-training
        np.random.seed(42)
        final_scores, model = train_and_adapt(
            x_tr, x_te, y_train, budget, drift_on, sample_weights=temporal_w)
        print(f"[MODEL] Done, {budget.elapsed():.1f}s elapsed")

        # --- Outputs ---
        pred_df = pd.DataFrame({ID_COL: test_df[ID_COL], "probability_score": final_scores})
        pred_df.to_csv("prediction.csv", index=False)
        joblib.dump(model, "model.joblib")

        # Drift table
        drop_rows = [{"Column Name": c, "Column Type": diag[c]["feat_type"],
                       "Drift Description": f"Severe {diag[c]['drift_type']} drift",
                       "Drift Mitigation": "Dropped"} for c in features if diag[c]["mitigation"] == "drop"]
        all_rows = drift_rows + drop_rows
        _print_drift_table(all_rows, len(features) - len(all_rows))

        # Runtime
        _print_boxed("Time Taken (s)", f"{time.time() - start_time:.1f}")

        # AU-PRC
        if len(train_df) > TRAIN_CAP:
            ei = sample_idx(len(train_df), TRAIN_CAP)
            atr = float(average_precision_score(y_train[ei], model.predict_proba(x_tr.iloc[ei])[:, 1]))
        else:
            atr = float(average_precision_score(y_train, model.predict_proba(x_tr)[:, 1]))
        if TARGET_COL in test_df.columns:
            yt = np.array([le.transform([str(v)])[0] if str(v) in le.classes_ else 0
                           for v in test_df[TARGET_COL]], dtype=np.int32)
            ate = float(average_precision_score(yt, final_scores))
        else:
            ate = float("nan")
        _print_auprc(f"{atr:.3f}", f"{ate:.3f}" if not np.isnan(ate) else "nan")

        # Predictions preview
        _print_predictions(pred_df)

    except Exception as exc:
        import traceback; traceback.print_exc()
        _fallback(train_path, test_path, start_time, str(exc))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data_filepath", type=str, required=True)
    p.add_argument("--test_data_filepath", type=str, required=True)
    a = p.parse_args()
    run_pipeline(a.train_data_filepath, a.test_data_filepath)


if __name__ == "__main__":
    main()
