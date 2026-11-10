"""
مدل یادگیری ماشین «احتمال نادرست بودن هر ادعا».

داده: نتیجه کارشناسی هر ادعا (درست بود / نادرست بود) که در صفحه گزارش ثبت می‌شود.
ورودی: نوع ادعا، وضعیت آن هنگام تحلیل، سن و کارکرد خودرو، اختلاف قیمت، نوع فروشنده و متن جمله ادعا.
با حداقل ۱۰۰ ادعای برچسب‌خورده:  python cli.py train
"""
from __future__ import annotations

import time

import joblib
import numpy as np

from .. import config, db

MODEL_PATH = config.MODELS_DIR / "claim_model.joblib"
MIN_LABELS = 100
NUMERIC = ["age_years", "km_per_year", "price_diff_pct", "image_count", "is_dealer", "has_seller_eval",
           "n_claims"]
CATEGORICAL = ["code", "status"]
_cache: dict = {}


def _frame(rows: list[dict]):
    import pandas as pd

    f = pd.DataFrame([{k: r.get(k) for k in NUMERIC + CATEGORICAL + ["claim_text"]} for r in rows])
    f[NUMERIC] = f[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for k in CATEGORICAL + ["claim_text"]:
        f[k] = f[k].fillna("").astype(str)
    return f


def train() -> dict:
    rows = db.claim_training_rows()
    y = np.array([r["label"] for r in rows])
    if len(rows) < MIN_LABELS or len(set(y)) < 2:
        return {"trained": False,
                "reason": f"{len(rows)} ادعای برچسب‌خورده؛ حداقل {MIN_LABELS} (شامل درست و نادرست) لازم است."}

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("txt", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=3000), "claim_text"),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))])
    X = _frame(rows)
    n_splits = min(5, int(np.bincount(y).min()))
    auc = None
    if n_splits >= 2:
        auc = float(cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                                    scoring="roc_auc").mean())
    pipe.fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "trained_at": time.time(), "n": len(rows), "cv_auc": auc}, MODEL_PATH)
    _cache.clear()
    return {"trained": True, "n": len(rows), "false_claims": int(y.sum()), "cv_auc": auc}


def predict(features: dict) -> float | None:
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    bundle = _cache["bundle"]
    if not bundle:
        return None
    return float(bundle["pipeline"].predict_proba(_frame([features]))[0, 1])
