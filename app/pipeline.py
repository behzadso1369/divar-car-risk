"""اجرای کامل تحلیل یک آگهی، به ترتیب مراحل."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from . import (db, honesty, stage1_needs, stage2_text, stage3_pricing, stage4_behavior, stage5_images,
               stage6_scoring, stage7_knowledge)
from .divar_client import DivarError, extract_token, fetch_post
from .ml import risk_model
from .parser import parse_post
from .stage1_needs import CustomerNeeds


def analyze(url_or_token: str, needs: CustomerNeeds | None = None, analyze_images: bool = False,
            use_cache: bool = True, offline_pricing: bool = False, raw: dict | None = None,
            save: bool = True) -> dict:
    needs = needs or CustomerNeeds()
    token = extract_token(url_or_token)
    raw = raw or fetch_post(token, use_cache=use_cache)
    ad = parse_post(token, raw)
    if ad.category and ad.category != "cars":
        raise DivarError("این آگهی در دسته خودرو نیست.")

    kb_entry = stage7_knowledge.find_entry(ad)

    # ترتیب منطقی اجرا (نمایش گزارش به ترتیب ۱ تا ۷ است)
    text = stage2_text.run(ad)
    body = stage2_text.body_stage(text)
    pricing = stage3_pricing.run(ad, text, offline=offline_pricing)
    needs_res = stage1_needs.run(ad, needs, kb_entry, pricing.data.get("fair_price"))
    behavior = stage4_behavior.run(ad, pricing)
    images = stage5_images.run(ad, analyze_images)
    knowledge = stage7_knowledge.run(ad, kb_entry)
    honesty_res = honesty.run(ad, text, pricing)

    stages = {"honesty": honesty_res, "needs": needs_res, "text": text, "body": body, "pricing": pricing,
              "behavior": behavior, "images": images, "knowledge": knowledge}

    # امتیاز قاعده‌محور → ویژگی‌ها → (اگر مدل آموزش دیده) احتمال ML → امتیاز نهایی
    preliminary = stage6_scoring.run(stages, None)
    features = risk_model.build_features(ad, stages, preliminary.data["rule_score"])
    ml_prob = risk_model.predict_proba(features)
    scoring = stage6_scoring.run(stages, ml_prob) if ml_prob is not None else preliminary
    stages["scoring"] = scoring

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ad": {k: v for k, v in asdict(ad).items() if k not in ("rows",)},
        "needs": asdict(needs),
        "recommended_models": stage1_needs.recommend_models(needs) if not needs.is_empty else [],
        "stages": {name: st.to_dict() for name, st in stages.items()},
        "summary": scoring.data,
    }
    if save:
        db.save_analysis(token, ad.brand_model_slug, ad.year_jalali, ad.price,
                         scoring.data["final_score"], scoring.data["tier"], ad.description, features, report,
                         honesty_score=honesty_res.data["score"])
    return report
