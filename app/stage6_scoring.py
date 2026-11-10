"""مرحله ۶: امتیاز نهایی ریسک، سطح‌بندی (کم/متوسط/بالا)، توضیح و توصیه."""
from __future__ import annotations

from . import config
from .models import LEVEL_ORDER, StageResult
from .text_utils import fmt_toman

TIER_FA = {"low": "ریسک پایین", "medium": "ریسک متوسط", "high": "ریسک بالا"}


def _capped(stage: StageResult, cap: float) -> tuple[float, float]:
    """(امتیاز ریسک با سقف، امتیاز مثبت)"""
    risk = sum(f.points for f in stage.findings if f.points > 0)
    pos = sum(f.points for f in stage.findings if f.points < 0)
    return min(risk, cap), pos


def tier_for(score: float) -> str:
    if score < config.TIER_LOW_MAX:
        return "low"
    if score < config.TIER_MEDIUM_MAX:
        return "medium"
    return "high"


def run(stages: dict[str, StageResult], ml_probability: float | None) -> StageResult:
    res = StageResult("scoring", "۶. رتبه‌بندی نهایی و توضیح")
    breakdown = {}
    total_pos = 0.0
    for name, cap in config.STAGE_CAPS.items():
        st = stages.get(name)
        if not st or st.skipped:
            continue
        risk, pos = _capped(st, cap)
        breakdown[name] = {"title": st.title, "risk": round(risk, 1), "cap": cap}
        total_pos += pos
    rule_score = sum(b["risk"] for b in breakdown.values()) + max(total_pos, config.POSITIVE_FLOOR)
    rule_score = max(0.0, min(100.0, rule_score))

    final = rule_score
    if ml_probability is not None:
        final = (1 - config.ML_BLEND_WEIGHT) * rule_score + config.ML_BLEND_WEIGHT * ml_probability * 100
    tier = tier_for(final)

    # --- اطمینان تحلیل: چقدر اطلاعات داشتیم ---
    confidence = 100
    missing = []
    text = stages["text"].data
    if not stages["text"].data.get("seller_eval"):
        confidence -= 25
        missing.append("فرم ارزیابی فروشنده")
    if text.get("desc_len", 0) < 60:
        confidence -= 15
        missing.append("توضیحات کافی")
    pricing = stages["pricing"]
    if pricing.skipped or not pricing.data.get("available"):
        confidence -= 20
        missing.append("قیمت‌گذاری")
    elif pricing.data.get("n_comparables", 0) < config.MIN_COMPARABLES:
        confidence -= 10
        missing.append("آگهی مشابه کافی")
    if stages["images"].skipped:
        confidence -= 5
    confidence = max(confidence, 20)

    # --- مهم‌ترین دلایل ---
    all_findings = [f for st in stages.values() for f in st.findings if st.name not in ("needs",)]
    top_risks = sorted((f for f in all_findings if f.points > 0), key=lambda f: -f.points)[:6]
    positives = [f for f in all_findings if f.points < 0]
    questions = []
    for f in sorted(all_findings, key=lambda f: -f.points):
        if f.question and f.question not in questions:
            questions.append(f.question)

    # --- توصیه ---
    deal = pricing.data.get("deal_label", "") if pricing.data.get("available") else ""
    diff = pricing.data.get("diff_pct")
    if tier == "low":
        rec = "گزینه کم‌ریسک. برای اطمینان قبل از پرداخت، کارشناسی حضوری (رنگ‌سنج + دیاگ) انجام شود."
    elif tier == "medium":
        rec = ("فقط با کارشناسی حضوری کامل و گرفتن پاسخ سؤالات زیر از فروشنده. "
               "ابهام‌ها و تناقض‌های این آگهی باید قبل از معامله روشن شود.")
    else:
        rec = ("خرید توصیه نمی‌شود، مگر کارشناسی دقیق ریسک‌ها را رد کند و قیمت متناسب با خسارت‌ها کاهش یابد.")
    if diff is not None and diff <= config.SUSPICIOUS_LOW_PRICE:
        rec += " ⚠ قیمت به‌شکل مشکوکی پایین است؛ به‌هیچ‌وجه قبل از دیدن خودرو و سند بیعانه واریز نکنید."
    if pricing.data.get("available") and diff is not None and diff > 0.07:
        rec += f" قیمت منصفانه با احتساب وضعیت حدود {fmt_toman(pricing.data['fair_price'])} است؛ جای چانه‌زنی دارد."

    hon = stages.get("honesty")
    if hon and hon.data.get("score", 100) < 50:
        rec += " به ادعاهای متن آگهی اعتماد نکنید؛ هر ادعا باید در کارشناسی جداگانه تأیید شود."

    needs = stages.get("needs")
    tolerance = needs.data.get("risk_tolerance") if needs and not needs.data.get("empty") else None
    if tolerance and LEVEL_ORDER.get(tier, 0) > LEVEL_ORDER.get(tolerance, 3):
        rec += f" این سطح ریسک از ریسک‌پذیری اعلام‌شده مشتری ({TIER_FA[tolerance].replace('ریسک ', '')}) بیشتر است."

    res.data = {
        "rule_score": round(rule_score, 1),
        "ml_probability": ml_probability,
        "final_score": round(final, 1),
        "tier": tier,
        "tier_fa": TIER_FA[tier],
        "confidence": confidence,
        "missing": missing,
        "breakdown": breakdown,
        "positive_total": round(max(total_pos, config.POSITIVE_FLOOR), 1),
        "top_risks": [f.to_dict() for f in top_risks],
        "positives": [f.to_dict() for f in positives],
        "questions": questions[:10],
        "deal_label": deal,
        "recommendation": rec,
        "honesty_score": hon.data.get("score") if hon else None,
        "honesty_label": hon.data.get("label") if hon else None,
        "honesty_level": hon.data.get("level") if hon else None,
    }
    return res
