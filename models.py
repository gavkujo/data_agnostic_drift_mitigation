import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

TARGET_COL   = "target"
ID_COLS      = {"CustomerID", "Month"}
WEIGHT_COL   = "__sample_weight__"


def _get_feature_cols(df: pd.DataFrame, exclude: set) -> list:
    return [c for c in df.columns if c not in exclude]


def _prepare_X(df: pd.DataFrame, feature_cols: list, dtype_ref: pd.DataFrame = None) -> pd.DataFrame:
    """Align to feature_cols, fill numeric NaNs, encode categoricals for LightGBM.
    If dtype_ref is provided, coerce each column to match its dtype in dtype_ref
    (handles cases where train had a column freq-encoded to float but test still has strings).
    """
    X = df.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        if dtype_ref is not None and col in dtype_ref.columns:
            ref_dtype = dtype_ref[col].dtype
            if pd.api.types.is_numeric_dtype(ref_dtype):
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-9999)
            else:
                X[col] = X[col].astype(str).astype("category")
        else:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(-9999)
            elif X[col].dtype == object:
                X[col] = X[col].astype("category")
    return X


def train_and_evaluate_model(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # Normalise target column name
    if "ChurnStatus" in train.columns:
        train = train.rename(columns={"ChurnStatus": "target"})
    if "ChurnStatus" in test.columns:
        test = test.rename(columns={"ChurnStatus": "target"})

    sample_weight = train[WEIGHT_COL].values if WEIGHT_COL in train.columns else None

    drop_train = ID_COLS | {TARGET_COL, WEIGHT_COL}
    drop_test  = ID_COLS | {TARGET_COL}

    train_feature_cols = _get_feature_cols(train, drop_train)

    X_train = _prepare_X(train, train_feature_cols)
    y_train = train[TARGET_COL]
    X_test  = _prepare_X(test, train_feature_cols, dtype_ref=X_train)

    y_test = test[TARGET_COL] if TARGET_COL in test.columns else None

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    if y_test is not None:
        y_test = le.transform(y_test)

    model = lgb.LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type='gain'
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    auprc_train = average_precision_score(y_train, y_train_pred)
    auprc_test = average_precision_score(y_test, y_test_pred)

    model_path = os.path.join(os.path.dirname(train_path), "model.joblib")
    joblib.dump(model, model_path)
    print(f"[MODEL] Saved to: {model_path}")

    return {
        'auprc_train': auprc_train,
        'auprc_test': auprc_test
    }
