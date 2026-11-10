"""
مدل یادگیری ماشین ریسک: از «نتیجه کارشناسی‌های واقعی شما» یاد می‌گیرد.

چرخه:
  ۱) هر آگهی تحلیل می‌شود → بردار ویژگی در دیتابیس ذخیره می‌شود.
  ۲) بعد از کارشناسی حضوری، نتیجه ثبت می‌شود:  python cli.py label <توکن> --hidden-issue yes --severity 2
  ۳) با حداقل ۳۰ برچسب:  python cli.py train
  ۴) از آن به بعد امتیاز نهایی = ترکیب امتیاز قاعده‌محور و احتمال پیش‌بینی‌شده مدل.

ورودی مدل: ویژگی‌های عددی + متن توضیحات (TF-IDF با n-gram کاراکتری که برای فارسی خوب کار می‌کند).
"""
from __future__ import annotations

import time

import joblib
import numpy as np

from .. import config, db
from ..models import StageResult

MODEL_PATH = config.MODELS_DIR / "risk_model.joblib"
_cache: dict = {}

NUMERIC_FEATURES = [
    "body_severity", "form_declared_areas", "unspecified_paint_spots", "n_contradictions",
    "contradiction_points", "doc_points", "pressure_points", "mechanical_points", "positive_points",
    "chassis_issue", "engine_repair", "gearbox_repair", "price_diff_pct", "n_comparables",
    "image_count", "is_dealer", "not_owner", "has_seller_eval", "desc_len", "km", "age_years",
    "km_per_year", "kb_base_risk", "promoted", "ad_age_days", "behavior_points", "image_points",
    "rule_score", "honesty_score", "honesty_consistency", "honesty_plausibility", "honesty_specificity",
    "honesty_transparency",
]


def build_features(ad, stages: dict[str, StageResult], rule_score: float) -> dict:
    text = stages["text"]
    td = text.data
    codes = [f.code for f in text.findings]

    def pts(prefixes: tuple[str, ...]) -> float:
        return float(sum(f.points for f in text.findings if f.code.startswith(prefixes)))

    pricing = stages["pricing"].data
    behavior = stages["behavior"]
    images = stages.get("images")
    hon = stages["honesty"].data if "honesty" in stages else {}
    return {
        "body_severity": td.get("body_severity", 0),
        "form_declared_areas": td.get("form_declared_areas", 0),
        "unspecified_paint_spots": td.get("unspecified_paint_spots", 0),
        "n_contradictions": sum(c.startswith("contradiction") for c in codes),
        "contradiction_points": pts(("contradiction",)),
        "doc_points": pts(("doc_",)),
        "pressure_points": pts(("pressure", "remote")),
        "mechanical_points": pts(("engine", "head_", "oil_", "overheat", "gearbox", "needs_", "odometer",
                                  "heavy", "form_")),
        "positive_points": pts(("pos_",)),
        "chassis_issue": int(td.get("flags", {}).get("chassis_issue", False)),
        "engine_repair": int(td.get("flags", {}).get("engine_repair", False)),
        "gearbox_repair": int(td.get("flags", {}).get("gearbox_repair", False)),
        "price_diff_pct": pricing.get("diff_pct"),
        "n_comparables": pricing.get("n_comparables", 0),
        "image_count": ad.image_count,
        "is_dealer": int(ad.business_type not in ("", "personal")),
        "not_owner": int(bool(ad.ownership) and "مالک" not in ad.ownership),
        "has_seller_eval": int(ad.has_seller_eval),
        "desc_len": len(ad.description or ""),
        "km": ad.km,
        "age_years": td.get("age_years"),
        "km_per_year": td.get("km_per_year"),
        "kb_base_risk": stages["knowledge"].data.get("base_risk"),
        "promoted": int(bool(behavior.data.get("promotion"))),
        "ad_age_days": behavior.data.get("ad_age_days"),
        "behavior_points": behavior.points,
        "image_points": images.points if images and not images.skipped else 0.0,
        "rule_score": rule_score,
        "honesty_score": hon.get("score"),
        **{f"honesty_{k}": v for k, v in hon.get("scores", {}).items()},
        "description": ad.description or "",
    }


def train() -> dict:
    rows = db.labeled_training_rows()
    labels = [r["label"] for r in rows]
    if len(rows) < config.ML_MIN_LABELS or len(set(labels)) < 2:
        return {"trained": False,
                "reason": f"{len(rows)} نتیجه کارشناسی ثبت شده؛ حداقل {config.ML_MIN_LABELS} "
                          f"(شامل هر دو حالت سالم و دارای ایراد پنهان) لازم است."}

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = _to_matrix(rows)
    y = np.array(labels)
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
        ("txt", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=5000),
         "description"),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))])
    n_splits = min(5, int(min(np.bincount(y))))
    auc = None
    if n_splits >= 2:
        auc = float(cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                                    scoring="roc_auc").mean())
    pipe.fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "trained_at": time.time(), "n": len(rows), "cv_auc": auc}, MODEL_PATH)
    _cache.clear()
    return {"trained": True, "n": len(rows), "positives": int(y.sum()), "cv_auc": auc}


def _to_matrix(rows: list[dict]):
    import pandas as pd

    frame = pd.DataFrame([{k: r.get(k) for k in NUMERIC_FEATURES + ["description"]} for r in rows])
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    frame["description"] = frame["description"].fillna("").astype(str)
    return frame


def load():
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    return _cache["bundle"]


def predict_proba(features: dict) -> float | None:
    bundle = load()
    if not bundle:
        return None
    return float(bundle["pipeline"].predict_proba(_to_matrix([features]))[0, 1])
