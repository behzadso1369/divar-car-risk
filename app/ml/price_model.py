"""
مدل سراسری قیمت (یادگیری ماشین) که از همه آگهی‌های مشابه جمع‌شده در دیتابیس یاد می‌گیرد.
وقتی آگهی هم‌مدل کافی در لحظه پیدا نشود، از این مدل استفاده می‌شود.

آموزش:  python cli.py train
"""
from __future__ import annotations

import time

import joblib
import numpy as np

from .. import config, db

MODEL_PATH = config.MODELS_DIR / "price_model.joblib"
_cache: dict = {}


def train() -> dict:
    rows = db.recent_comparables(config.COMPARABLE_MAX_AGE_DAYS)
    if len(rows) < config.PRICE_MODEL_MIN_ROWS:
        return {"trained": False,
                "reason": f"{len(rows)} آگهی در دیتابیس؛ حداقل {config.PRICE_MODEL_MIN_ROWS} لازم است."}

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score

    models = sorted({r["brand_model"] for r in rows})
    cities = sorted({r["city_id"] for r in rows})
    m_idx = {m: i for i, m in enumerate(models)}
    c_idx = {c: i for i, c in enumerate(cities)}
    X = np.array([[m_idx[r["brand_model"]], c_idx[r["city_id"]], r["year"], r["km"] / 10_000] for r in rows],
                 dtype=float)
    y = np.log([r["price"] for r in rows])
    est = HistGradientBoostingRegressor(categorical_features=[0, 1], max_iter=300, learning_rate=0.08)
    cv = KFold(min(5, max(2, len(rows) // 50)), shuffle=True, random_state=0)
    mae = -cross_val_score(est, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
    est.fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": est, "models": m_idx, "cities": c_idx, "trained_at": time.time(),
                 "n": len(rows), "cv_mae_log": mae}, MODEL_PATH)
    _cache.clear()
    return {"trained": True, "n": len(rows), "n_models": len(models),
            "cv_mean_abs_pct_error": round(float(np.expm1(mae)) * 100, 1)}


def _load():
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    return _cache["bundle"]


def predict(brand_model: str, year: int, km: int | None, city_id: str) -> float | None:
    bundle = _load()
    if not bundle or km is None or brand_model not in bundle["models"]:
        return None
    # مدل قدیمی‌تر از یک ماه به‌خاطر تورم قابل اعتماد نیست
    if time.time() - bundle["trained_at"] > config.COMPARABLE_MAX_AGE_DAYS * 86400:
        return None
    city = bundle["cities"].get(str(city_id))
    if city is None:
        return None
    x = [[bundle["models"][brand_model], city, year, km / 10_000]]
    return float(np.exp(bundle["model"].predict(x)[0]))
