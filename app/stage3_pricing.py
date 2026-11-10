"""
مرحله ۳: قیمت‌گذاری اختصاصی.

۱) آگهی‌های هم‌مدل و هم‌سال همان شهر از دیوار جمع می‌شوند (و در دیتابیس ذخیره می‌شوند).
۲) داده پرت با IQR حذف می‌شود.
۳) رگرسیون مقاوم (HuberRegressor) روی log(قیمت) ~ کارکرد (+ سال) → قیمت مورد انتظار بازار برای این کارکرد.
۴) تعدیل بر اساس شدت خسارت همین خودرو → «قیمت منصفانه».
۵) اگر داده محلی کم بود: مدل سراسری یادگرفته‌شده از دیتابیس (ml/train_price.py).
"""
from __future__ import annotations

import numpy as np

from . import config
from .divar_client import DivarError, search_comparables
from .ml import price_model
from .models import Ad, Finding, StageResult
from .text_utils import fmt_toman


def _clean_outliers(rows: list[dict]) -> list[dict]:
    if len(rows) < 6:
        return rows
    logp = np.log([r["price"] for r in rows])
    q1, q3 = np.percentile(logp, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [r for r, lp in zip(rows, logp) if lo <= lp <= hi]


def _fit_local_model(rows: list[dict], target_km: int | None, target_year: int) -> tuple[float, float, str]:
    """خروجی: (قیمت مورد انتظار، انحراف معیار نسبی باقیمانده، روش)"""
    prices = np.array([r["price"] for r in rows], dtype=float)
    kms = [r.get("km") for r in rows]
    usable = [i for i, k in enumerate(kms) if k is not None]
    if target_km is None or len(usable) < 6:
        med = float(np.median(prices))
        spread = float(np.std(np.log(prices))) if len(prices) > 1 else 0.15
        return med, spread, "median"

    from sklearn.linear_model import HuberRegressor

    years = np.array([rows[i].get("year") or target_year for i in usable], dtype=float)
    X = np.column_stack([
        np.array([kms[i] for i in usable], dtype=float) / 10_000,
        years - target_year,
    ])
    if np.ptp(years) == 0:
        X = X[:, :1]
    y = np.log(prices[usable])
    model = HuberRegressor(epsilon=1.5, max_iter=500).fit(X, y)
    x0 = [target_km / 10_000] + ([0.0] if X.shape[1] == 2 else [])
    pred = float(np.exp(model.predict([x0])[0]))
    resid = y - model.predict(X)
    spread = float(np.std(resid)) if len(resid) > 2 else 0.15
    # شیب منفی منطقی است؛ اگر مثبت درآمد (داده نویزی) به میانه برگرد
    if model.coef_[0] > 0.02:
        return float(np.median(prices)), spread, "median"
    return pred, spread, "huber_regression"


def condition_discount(text_data: dict) -> float:
    flags = text_data.get("flags", {})
    d = text_data.get("body_severity", 0) * config.DAMAGE_DISCOUNT_PER_SEVERITY
    if flags.get("chassis_issue"):
        d += config.CHASSIS_DISCOUNT
    if flags.get("engine_repair"):
        d += config.ENGINE_REPAIR_DISCOUNT
    if flags.get("gearbox_repair"):
        d += config.GEARBOX_REPAIR_DISCOUNT
    d = min(d, config.MAX_DAMAGE_DISCOUNT) - config.MARKET_AVG_DAMAGE_DISCOUNT
    return max(d, -config.MARKET_AVG_DAMAGE_DISCOUNT)


def run(ad: Ad, text_result: StageResult, offline: bool = False) -> StageResult:
    res = StageResult("pricing", "۳. فیلتر قیمت‌گذاری اختصاصی")
    if not ad.price:
        res.add(Finding("pricing", "price_negotiable",
                        "قیمت واقعی اعلام نشده (توافقی / معاوضه)؛ مقایسه قیمت ممکن نیست",
                        points=3, level="low", question="قیمت نقدی نهایی خودرو چقدر است؟"))
        res.data = {"available": False}
        return res
    if text_result.data.get("presale"):
        res.skipped = True
        res.skip_reason = "آگهی پیش‌فروش/حواله است و با قیمت خودروهای موجود قابل مقایسه نیست."
        res.data = {"available": False}
        return res
    if not (ad.brand_model_slug and ad.year_jalali and ad.city_id):
        res.skipped, res.skip_reason = True, "مدل، سال یا شهر آگهی مشخص نیست."
        res.data = {"available": False}
        return res

    rows: list[dict] = []
    own_listing = None
    years_used = [ad.year_jalali]
    error = ""
    if not offline:
        try:
            found = search_comparables(ad.brand_model_slug, ad.city_id, ad.year_jalali)
            own_listing = next((r for r in found if r["token"] == ad.token), None)
            rows = [r for r in found if r["token"] != ad.token]
            if len(rows) < config.MIN_COMPARABLES:
                for y in (ad.year_jalali - 1, ad.year_jalali + 1):
                    more = search_comparables(ad.brand_model_slug, ad.city_id, y, max_pages=2)
                    rows += [r for r in more if r["token"] != ad.token]
                    years_used.append(y)
        except DivarError as exc:
            error = str(exc)

    clean = _clean_outliers(rows)
    method = ""
    expected = spread = None
    if len(clean) >= 4:
        expected, spread, method = _fit_local_model(clean, ad.km, ad.year_jalali)
    else:
        pred = price_model.predict(ad.brand_model_slug, ad.year_jalali, ad.km, ad.city_id)
        if pred:
            expected, spread, method = pred, 0.15, "global_model"

    if expected is None:
        res.skipped = True
        res.skip_reason = error or "آگهی مشابه کافی برای قیمت‌گذاری پیدا نشد."
        res.data = {"available": False, "n_comparables": len(clean)}
        return res

    discount = condition_discount(text_result.data)
    fair = expected * (1 - discount)
    diff = (ad.price - fair) / fair
    prices = sorted(r["price"] for r in clean)
    percentile = (sum(p <= ad.price for p in prices) / len(prices) * 100) if prices else None

    if diff <= config.VERY_SUSPICIOUS_LOW_PRICE:
        label = "مشکوک — خیلی ارزان‌تر از بازار"
        res.add(Finding("pricing", "price_very_low",
                        f"قیمت {abs(diff) * 100:.0f}٪ پایین‌تر از قیمت منصفانه است",
                        detail="قیمت خیلی پایین یا نشانه ایراد پنهان است یا الگوی کلاهبرداری (درخواست بیعانه).",
                        points=20, level="high",
                        question="چرا قیمت این‌قدر پایین‌تر از بازار است؟"))
    elif diff <= config.SUSPICIOUS_LOW_PRICE:
        label = "ارزان — نیازمند بررسی دقیق"
        res.add(Finding("pricing", "price_low",
                        f"قیمت {abs(diff) * 100:.0f}٪ پایین‌تر از قیمت منصفانه است",
                        detail="ممکن است فرصت خوبی باشد، ولی احتمال ایراد اعلام‌نشده هم بالاست.",
                        points=10, level="medium",
                        question="دلیل قیمت پایین‌تر از بازار چیست؟"))
    elif diff <= -0.07:
        label = "معامله خوب"
    elif diff <= 0.07:
        label = "منصفانه"
    elif diff <= 0.15:
        label = "کمی گران"
    else:
        label = "گران"
    if diff > 0.07:
        res.add(Finding("pricing", "price_high",
                        f"قیمت {diff * 100:.0f}٪ بالاتر از قیمت منصفانه (با احتساب خسارت‌ها) است",
                        detail="ریسک فنی نیست، اما جای چانه‌زنی دارد.", points=0, level="info"))
    if text_result.data.get("body_severity", 0) > 0 and diff > 0.0:
        res.add(Finding("pricing", "price_ignores_damage",
                        "قیمت پیشنهادی خسارت‌های اعلام‌شده را لحاظ نکرده است", points=0, level="info",
                        question=f"با توجه به خسارت‌ها، قیمت منصفانه حدود {fmt_toman(fair)} است؛ تخفیف می‌دهید؟"))

    if own_listing and own_listing.get("red_text"):
        res.data["promotion"] = own_listing["red_text"]

    res.data.update({
        "available": True,
        "method": method,
        "n_comparables": len(clean),
        "n_raw": len(rows),
        "years_used": sorted(set(years_used)),
        "market_median": float(np.median(prices)) if prices else None,
        "expected_at_km": expected,
        "condition_discount": discount,
        "fair_price": fair,
        "ask": ad.price,
        "diff_pct": diff,
        "percentile": percentile,
        "fair_low": fair * float(np.exp(-spread)) if spread else None,
        "fair_high": fair * float(np.exp(spread)) if spread else None,
        "deal_label": label,
        "points": [{"km": r["km"], "price": r["price"]} for r in clean if r.get("km") is not None][:150],
        "error": error,
    })
    return res
