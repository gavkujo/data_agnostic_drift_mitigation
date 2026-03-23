"""
Mitigation Main — per-feature drift mitigation.

Routing is driven by per-feature `feature_shift_type` labels from stat_results,
not a single global shift type verdict. Each feature gets only what it needs.

Dataset-level techniques (importance weighting, ELSA) are triggered by whether
any feature has the relevant shift type, or by dataset-level signals:
  - Importance weighting : any feature has covariate_* shift type
  - ELSA                 : dataset-level label shift detected (shift_diagnosis)
  - Feature alignment    : per-feature, driven by feature_shift_type per column
  - Feature selection    : always last

The output is a modified training CSV saved to disk, with a '__sample_weight__'
column that models.py passes to LightGBM as sample_weight.
"""

import os
import pandas as pd
import numpy as np

from .importance_weighting   import apply_importance_weighting
from .feature_alignment      import apply_feature_alignment
from .feature_selection      import apply_feature_selection
from .label_shift_correction import apply_label_shift_correction


def run_mitigation(train_path: str, drift_info: dict, verbose: bool = True) -> str:
    if "_train_df" not in drift_info or "_test_df" not in drift_info:
        raise KeyError("drift_info missing '_train_df' or '_test_df'.")

    train_df = drift_info["_train_df"].copy()
    test_df  = drift_info["_test_df"].copy()

    stat_results    = drift_info.get("stat_results", pd.DataFrame())
    drifted_columns = drift_info.get("columns_with_drift", [])
    drift_scores    = drift_info.get("drift_scores", pd.Series(dtype=float))
    sample_weights  = drift_info.get("sample_weights", None)
    shift_diagnosis = drift_info.get("shift_diagnosis", {})
    global_shift    = drift_info.get("shift_type", "none")

    _drop = {"target", "CustomerID", "Month", "__sample_weight__", "ChurnStatus"}
    test_feature_cols = [c for c in test_df.columns if c not in _drop]
    test_df_features  = test_df[test_feature_cols]

    # ── Per-feature shift type lookup ─────────────────────────────────────────
    # feature_shift_type is set per feature by statistical_tests.py
    feat_shift_types = {}
    if not stat_results.empty and "feature_shift_type" in stat_results.columns:
        feat_shift_types = stat_results.set_index("column")["feature_shift_type"].to_dict()

    has_covariate = any(
        v.startswith("covariate") for v in feat_shift_types.values()
    )
    has_label     = shift_diagnosis.get("label_detected", False)

    applied = []
    print(f"\n[MITIGATION] Global shift type: {global_shift.upper()}")
    if feat_shift_types:
        type_counts = {}
        for v in feat_shift_types.values():
            type_counts[v] = type_counts.get(v, 0) + 1
        print(f"  [MITIGATION] Per-feature shift types: {type_counts}")

    if global_shift == "none" and not has_covariate and not has_label:
        print("[MITIGATION] No drift detected. Returning original training data.")
        out_path = _save(train_df, train_path)
        _print_summary(global_shift, applied)
        return out_path

    # ── Importance weighting: runs if any feature has covariate shift ─────────
    if has_covariate and sample_weights is not None and len(sample_weights) == len(train_df):
        print("\n[MITIGATION] Applying importance weighting (adversarial reweighting)...")
        train_df = apply_importance_weighting(
            train_df=train_df,
            sample_weights=sample_weights,
            use_doubly_robust=True,
            verbose=verbose,
        )
        applied.append("Importance weighting (adversarial validation + doubly-robust)")

    # ── ELSA: runs if dataset-level label shift detected ──────────────────────
    if has_label and "target" in train_df.columns:
        print("\n[MITIGATION] Applying label shift correction (ELSA)...")
        train_df_after_elsa = apply_label_shift_correction(
            train_df=train_df,
            test_df=test_df_features,
            verbose=verbose,
        )
        elsa_ran = (
            "__sample_weight__" in train_df_after_elsa.columns and (
                "__sample_weight__" not in train_df.columns or
                not train_df_after_elsa["__sample_weight__"].equals(
                    train_df.get("__sample_weight__", pd.Series())
                )
            )
        )
        if elsa_ran:
            applied.append("Label shift correction (ELSA moment matching)")
        train_df = train_df_after_elsa

    # ── Feature alignment: per-feature, driven by feature_shift_type ─────────
    if drifted_columns:
        print("\n[MITIGATION] Applying feature alignment (per-feature targeted)...")
        apply_decorrelation = global_shift in ("mixed", "concept")
        train_df = apply_feature_alignment(
            train_df=train_df,
            test_df=test_df_features,
            stat_results=stat_results,
            drifted_columns=drifted_columns,
            apply_decorrelation=apply_decorrelation,
            verbose=verbose,
        )
        applied.append(f"Feature alignment (per-feature targeted, {len(drifted_columns)} features)")

    # ── Feature selection: always last ────────────────────────────────────────
    if drifted_columns:
        print("\n[MITIGATION] Applying feature selection (drop high-drift/low-importance)...")
        train_df, test_df_features, dropped = apply_feature_selection(
            train_df=train_df,
            test_df=test_df_features,
            drifted_columns=drifted_columns,
            drift_scores=drift_scores,
            verbose=verbose,
        )
        if dropped:
            applied.append(f"Feature selection (dropped {len(dropped)}: {dropped})")

    out_path = _save(train_df, train_path)
    _print_summary(global_shift, applied)
    print(f"[MITIGATION] Mitigated training data saved to: {out_path}\n")
    return out_path


def _save(df: pd.DataFrame, original_path: str) -> str:
    base, ext = os.path.splitext(original_path)
    out_path = f"{base}_mitigated{ext}"
    df.to_csv(out_path, index=False)
    return out_path


def _print_summary(global_shift: str, applied: list) -> None:
    print("\n" + "=" * 60)
    print("  DRIFT MITIGATION SUMMARY")
    print("=" * 60)
    print(f"  Global shift type  : {global_shift.upper()}")
    print(f"  Techniques applied : {len(applied)}")
    if not applied:
        print("  (none)")
    else:
        for i, a in enumerate(applied, 1):
            print(f"    {i}. {a}")
    print("=" * 60 + "\n")