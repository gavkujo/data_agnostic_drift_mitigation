"""Shared utilities: constants, Budget, sampling, stats, month parsing, feature classification."""
from __future__ import annotations
import math, time
import numpy as np
import pandas as pd

ID_COL = "CustomerID"
TIME_COL = "Month"
TARGET_COL = "ChurnStatus"
FIXED_PARAMS = {"verbosity": -1, "objective": "binary", "is_unbalance": True,
                "random_state": 42, "importance_type": "gain"}
DRIFT_SAMPLE = 200_000
ENCODE_SAMPLE = 1_000_000
TRAIN_CAP = 2_000_000
DOMAIN_AUC_SAMPLE = 100_000


class Budget:
    def __init__(self, limit=600.0):
        self.start = time.time()
        self.limit = limit
    def elapsed(self): return time.time() - self.start
    def remaining(self): return self.limit - self.elapsed()
    def can_afford(self, s): return self.remaining() > s + 30


def sample_idx(n, max_n):
    if n <= max_n: return np.arange(n)
    return np.linspace(0, n - 1, max_n, dtype=int)


def stratified_sample_idx(y, max_n):
    if len(y) <= max_n: return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    parts = []
    for cls, cnt in zip(classes, counts):
        ci = np.where(y == cls)[0]
        parts.append(ci[sample_idx(len(ci), max(1, int(max_n * cnt / len(y))))])
    return np.sort(np.concatenate(parts))


def ncdf(z): return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def parse_months(month_series):
    s = month_series.astype(str).str.strip()
    for fmt in ["%y-%b", "%d-%b", "%Y-%b", "%Y-%m", "%b-%y", "%b-%d"]:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed.dt.to_period("M")
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().mean() > 0.5:
        return parsed.dt.to_period("M")
    return pd.Series(pd.NaT, index=month_series.index)


def classify_features(train_df, features):
    """Robustly classify numeric vs categorical (handles string-encoded numbers)."""
    classified = {}
    si = sample_idx(len(train_df), min(10_000, len(train_df)))
    for col in features:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            classified[col] = "numeric"
        else:
            coerced = pd.to_numeric(train_df.iloc[si][col], errors="coerce")
            classified[col] = "numeric" if coerced.notna().mean() > 0.5 else "categorical"
    return classified
