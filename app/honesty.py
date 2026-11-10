"""
صداقت فروشنده: چقدر می‌شود به حرف‌های این آگهی اعتماد کرد؟ (جدا از ریسک خود خودرو)

۱) هر ادعای آگهی استخراج و یکی از چهار وضعیت را می‌گیرد:
   consistent   ✅ با بقیه اطلاعات آگهی (فرم، فیلدها، قیمت) سازگار است
   contradicted ❌ با اطلاعات دیگر همین آگهی تناقض دارد
   implausible  ⚠️ با سن، کارکرد، قیمت یا نوع فروشنده جور درنمی‌آید
   unverifiable ❔ فقط حرف است و از روی آگهی قابل بررسی نیست
۲) چهار شاخص: سازگاری، باورپذیری، جزئیات مشخص (در برابر تعریف کلی)، شفافیت
۳) یادگیری: نتیجه کارشناسی هر ادعا ثبت می‌شود → نرخ دروغ هر نوع ادعا + مدل ML
"""
from __future__ import annotations

import re

from . import config, db
from . import lexicon as L
from .models import Ad, Finding, StageResult
from .text_utils import normalize, split_clauses, word_or_digit_to_int

STATUS_FA = {
    "consistent": "سازگار", "contradicted": "متناقض", "implausible": "بعید", "unverifiable": "غیرقابل بررسی",
}
STATUS_ICON = {"consistent": "✅", "contradicted": "❌", "implausible": "⚠️", "unverifiable": "❔"}
# ادعاهای «همه‌چیز عالی است»: اگر قیمت مشکوک‌پایین باشد، باورپذیری‌شان زیر سؤال می‌رود
PERFECT_CLAIMS = {"body_no_paint", "no_scratch", "no_accident", "no_issues", "like_new", "engine_ok",
                  "technical_ok", "engine_sealed"}
# سؤالی که یک ادعای غیرقابل بررسی را به ادعای قابل بررسی تبدیل می‌کند
CLAIM_QUESTIONS = {
    "km_real": "برای کارکرد، سوابق سرویس یا برگه معاینه فنی با کیلومتر ثبت‌شده دارید؟",
    "service_done": "فاکتور سرویس‌ها را می‌توانم ببینم؟",
    "tech_inspection": "برگه معاینه فنی را می‌توانم ببینم؟ (کیلومتر ثبت‌شده در آن را هم چک می‌کنم)",
    "first_owner": "سند از چه تاریخی به نام شماست؟",
    "body_no_paint": "اجازه می‌دهید بدنه با رنگ‌سنج بررسی شود؟",
    "no_accident": "اجازه می‌دهید سابقه خسارت بیمه خودرو استعلام شود؟",
    "engine_ok": "اجازه دیاگ و تست موتور در حالت سرد را می‌دهید؟",
    "technical_ok": "اجازه دیاگ و تست جاده را می‌دهید؟",
    "doc_clean": "اجازه استعلام خلافی و وضعیت سند از پلیس راهور را می‌دهید؟",
}
WEIGHTS = {"consistency": 0.35, "plausibility": 0.25, "specificity": 0.20, "transparency": 0.20}
_CLAIM_RES = [(code, cat, fa, re.compile(p)) for code, cat, fa, p in L.CLAIMS]
_PARTS_RE = re.compile("|".join(p for _, _, _, p in L.BODY_PARTS))


def extract_claims(title: str, desc: str) -> list[dict]:
    claims, seen = [], set()
    for clause in split_clauses(f"{title}\n{desc}"):
        c = normalize(clause)
        for code, cat, fa, rx in _CLAIM_RES:
            m = rx.search(c)
            if not m:
                continue
            # «سقف و کاپوت بی‌رنگ» = ادعای قطعه‌ای، نه ادعای کل بدنه
            if code == "body_no_paint" and _PARTS_RE.search(c):
                code, fa = "parts_no_paint", "قطعات مشخص بدون رنگ"
            if code in seen:
                continue
            seen.add(code)
            claims.append({"id": f"{code}:{len(claims)}", "code": code, "category": cat, "label": fa,
                           "evidence": clause.strip(), "match": m.group(0)})
    return claims


def _insurance_months(text: str) -> int | None:
    t = normalize(text)
    if re.search(r"(?:یک|یه|1) ?سال|یکسال", t):
        return 12
    m = re.search(r"(\d+|یک|یه|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|یازده|دوازده) ?ماه", t)
    if m:
        extra = {"یازده": 11, "دوازده": 12}
        return extra.get(m.group(1)) or word_or_digit_to_int(m.group(1))
    return None


def _verify(claim: dict, ad: Ad, td: dict, text_codes: set[str], pricing: dict) -> tuple[str, str]:
    code = claim["code"]
    age = td.get("age_years") or 0
    km = ad.km or 0
    damages = [d for d in td.get("merged_damages", []) if d["dtype"] not in ("scratch", "pdr")]
    dmg_txt = "، ".join(f"{d['label']} ({d['dtype_fa']})" for d in damages[:3])
    flags = td.get("flags", {})
    car_level = set(td.get("car_level_damage", []))
    ev = {t: i.get("value", "") for t, i in td.get("seller_eval", {}).items()}

    def form(keyword: str) -> str | None:
        vals = [v for t, v in ev.items() if keyword in t and "تعیین" not in v]
        bad = [v for v in vals if not normalize(v).startswith("سالم")]
        return bad[0] if bad else (vals[0] if vals else None)

    mech_issue = flags.get("engine_repair") or text_codes & {"oil_burning", "overheat", "needs_repair",
                                                             "head_gasket", "engine_rebuild", "engine_replaced"}

    if code == "body_no_paint":
        if damages or td.get("unspecified_paint_spots") or car_level & {"full_paint", "accident_car"}:
            return "contradicted", f"در همین آگهی خسارت بدنه اعلام شده: {dmg_txt or 'رنگ‌شدگی'}"
        if td.get("form_body_clean"):
            return "consistent", "با فرم ارزیابی (بدنه سالم) سازگار است؛ با رنگ‌سنج تأیید شود"
        if age > 12 or km > 250_000:
            return "implausible", f"خودروی {age:.0f} ساله با {km:,} کیلومتر کاملاً بی‌رنگ کم پیش می‌آید"
        return "unverifiable", "فرم بدنه پر نشده؛ فقط با رنگ‌سنج قابل بررسی است"

    if code == "parts_no_paint":
        named = {k for k, _, _, p in L.BODY_PARTS if re.search(p, normalize(claim["evidence"]))}
        clash = [d for d in damages if d["part"] in named and d["source"] == "form"]
        if clash:
            return "contradicted", f"فرم ارزیابی برای همین قطعه خسارت ثبت کرده: {clash[0]['label']}"
        return "consistent", "ادعای مشخص درباره قطعات؛ با بقیه آگهی تناقض ندارد"

    if code == "no_scratch":
        body = form("بدنه") or ""
        if damages or re.search(r"خط|خش", normalize(body)):
            return "contradicted", f"در همین آگهی خسارت بدنه اعلام شده: {dmg_txt or body}"
        if age > 8:
            return "implausible", f"خودروی {age:.0f} ساله معمولاً خط و خش جزئی دارد"
        return ("consistent", "با فرم ارزیابی سازگار است") if td.get("form_body_clean") else \
            ("unverifiable", "فقط با بازدید قابل بررسی است")

    if code == "no_accident":
        if flags.get("chassis_issue") or car_level & {"accident_car", "airbag", "rollover"}:
            return "contradicted", "در همین آگهی تصادف، آسیب شاسی یا عمل کردن کیسه هوا اعلام شده"
        replaced = [d for d in damages if d["dtype"] in ("replaced", "accident", "welded")]
        if replaced:
            return "implausible", f"تعویض یا ضربه قطعه ({replaced[0]['label']}) معمولاً نتیجه تصادف است"
        return ("consistent", "با فرم ارزیابی سازگار است") if td.get("form_body_clean") else \
            ("unverifiable", "فقط با کارشناسی بدنه قابل بررسی است")

    if code == "chassis_ok":
        if flags.get("chassis_issue"):
            return "contradicted", "در همین آگهی آسیب شاسی اعلام شده"
        if form("شاسی"):
            return "consistent", f"با فرم ارزیابی (شاسی: {form('شاسی')}) سازگار است"
        return "unverifiable", "فرم شاسی پر نشده"

    if code in ("engine_sealed", "engine_ok", "technical_ok"):
        if mech_issue or (code == "engine_sealed" and re.search(r"تعمیر|تعویض", normalize(form("موتور") or ""))):
            return "contradicted", "در همین آگهی به تعمیر یا ایراد موتور اشاره شده"
        if code == "technical_ok" and flags.get("gearbox_repair"):
            return "contradicted", "در همین آگهی به ایراد یا تعمیر گیربکس اشاره شده"
        engine = form("موتور")
        if engine and normalize(engine).startswith("سالم"):
            return "consistent", "با فرم ارزیابی (موتور سالم) سازگار است؛ با دیاگ تأیید شود"
        return "unverifiable", "فقط با تست و دیاگ قابل بررسی است"

    if code == "gearbox_ok":
        if flags.get("gearbox_repair"):
            return "contradicted", "در همین آگهی به ایراد یا تعمیر گیربکس اشاره شده"
        gb = form("گیربکس")
        if gb and normalize(gb).startswith("سالم"):
            return "consistent", "با فرم ارزیابی سازگار است"
        return "unverifiable", "فقط با تست جاده قابل بررسی است"

    if code == "no_issues":
        if damages or mech_issue or flags.get("chassis_issue") or flags.get("gearbox_repair"):
            return "contradicted", f"در همین آگهی ایراد اعلام شده: {dmg_txt or 'ایراد فنی'}"
        if age > 10 or km > 200_000:
            return "implausible", f"خودروی {age:.0f} ساله با {km:,} کیلومتر بدون هیچ ایرادی بعید است"
        return "unverifiable", "ادعای کلی؛ قابل بررسی نیست"

    if code == "like_new":
        if damages:
            return "contradicted", f"با خسارت‌های اعلام‌شده نمی‌خواند: {dmg_txt}"
        if age > 3 or km > 50_000:
            return "implausible", f"«در حد صفر» برای خودروی {age:.0f} ساله با {km:,} کیلومتر اغراق است"
        return "unverifiable", "ادعای تبلیغاتی"

    if code == "km_real":
        if "contradiction_km" in text_codes:
            return "contradicted", "کارکرد متن با فیلد کارکرد فرق دارد"
        kpy = td.get("km_per_year")
        if kpy is not None and age >= 3 and kpy < 4000:
            return "implausible", f"حدود {kpy:,} کیلومتر در سال برای این سن غیرعادی کم است"
        return "unverifiable", "فقط با سوابق سرویس یا معاینه فنی قابل بررسی است"

    if code == "first_owner":
        if ad.business_type and ad.business_type != "personal":
            return "implausible", "آگهی‌دهنده نمایشگاه است، نه مالک اول"
        if ad.ownership and "مالک" not in ad.ownership:
            return "implausible", f"آگهی‌دهنده مالک نیست («{ad.ownership}»)"
        return "unverifiable", "با تاریخ سند قابل بررسی است"

    if code == "female_owned":
        return "unverifiable", "ادعای رایج تبلیغاتی؛ قابل بررسی نیست و ربطی به سلامت خودرو ندارد"

    if code == "insurance":
        months = _insurance_months(claim["evidence"])
        if months is not None and ad.insurance_months is not None:
            if abs(months - ad.insurance_months) > 2:
                return "contradicted", f"متن {months} ماه می‌گوید، فیلد بیمه {ad.insurance_months} ماه"
            return "consistent", f"با فیلد بیمه ({ad.insurance_months} ماه) سازگار است"
        return "unverifiable", "با برگه بیمه قابل بررسی است"

    if code == "doc_clean":
        if text_codes & {"doc_mortgage", "doc_installment", "doc_power_of_attorney", "doc_seized", "doc_fines"}:
            return "contradicted", "در همین آگهی به رهن، قسط، وکالت یا بدهی اشاره شده"
        return "unverifiable", "با استعلام پلیس راهور قابل بررسی است"

    if code == "tech_inspection":
        return "unverifiable", "با برگه معاینه فنی قابل بررسی است"

    if code == "service_done":
        if re.search(r"فاکتور|رسید|دفترچه", normalize(claim["evidence"])):
            return "unverifiable", "فروشنده مدرک دارد؛ فاکتورها را حتماً ببینید"
        return "unverifiable", "بدون فاکتور فقط حرف است"

    if code == "price_fair":
        diff = pricing.get("diff_pct") if pricing.get("available") else None
        if diff is None:
            return "unverifiable", "قیمت‌گذاری انجام نشد"
        if diff > 0.05:
            return "contradicted", f"قیمت {diff * 100:.0f}٪ بالاتر از قیمت منصفانه ماست"
        return "consistent", f"قیمت در محدوده منصفانه است ({diff * 100:+.0f}٪)"

    return "unverifiable", ""


def _learned_rate(stats: dict, code: str, status: str) -> tuple[float, int] | None:
    """نرخ نادرستی این نوع ادعا در کارشناسی‌های قبلی (با هموارسازی بیزی)."""
    s = stats.get(code)
    if not s:
        return None
    by = s["by_status"].get(status)
    n, nf = (by["n"], by["n_false"]) if by and by["n"] >= 5 else (s["n"], s["n_false"])
    if n < 5:
        return None
    return (nf + 1) / (n + 2), n


def run(ad: Ad, text: StageResult, pricing: StageResult) -> StageResult:
    res = StageResult("honesty", "صداقت فروشنده")
    td = text.data
    text_codes = {f.code for f in text.findings}
    desc = ad.description or ""
    ndesc = normalize(f"{ad.title}\n{desc}")

    claims = extract_claims(ad.title, desc)
    for c in claims:
        c["status"], c["reason"] = _verify(c, ad, td, text_codes, pricing.data)

    # ادعای «همه‌چیز عالی» + قیمت مشکوک‌پایین = چرا این‌قدر ارزان؟
    diff = pricing.data.get("diff_pct") if pricing.data.get("available") else None
    perfect = [c for c in claims if c["code"] in PERFECT_CLAIMS]
    cheap_perfect = bool(perfect) and diff is not None and diff <= config.SUSPICIOUS_LOW_PRICE
    if cheap_perfect:
        for c in perfect:
            if c["status"] in ("consistent", "unverifiable"):
                c["status"] = "implausible"
                c["reason"] = f"اگر خودرو این‌قدر سالم است، چرا {abs(diff) * 100:.0f}٪ زیر قیمت منصفانه است؟"

    # --- یادگیری از کارشناسی‌های قبلی ---
    stats = db.claim_stats()
    from .ml import claim_model
    context = {
        "age_years": td.get("age_years"), "km_per_year": td.get("km_per_year"), "price_diff_pct": diff,
        "image_count": ad.image_count, "is_dealer": int(ad.business_type not in ("", "personal")),
        "has_seller_eval": int(ad.has_seller_eval), "n_claims": len(claims),
    }
    for c in claims:
        c["features"] = {**context, "code": c["code"], "status": c["status"], "claim_text": c["evidence"]}
        rate = _learned_rate(stats, c["code"], c["status"])
        c["learned_false_rate"], c["learned_n"] = (rate if rate else (None, 0))
        c["p_false"] = claim_model.predict(c["features"])
        c["status_fa"], c["icon"] = STATUS_FA[c["status"]], STATUS_ICON[c["status"]]

    # خود قیمت هم یک ادعاست: قیمت غیرواقعی پایین = قیمت طعمه یا ایراد پنهان
    if diff is not None and diff <= config.VERY_SUSPICIOUS_LOW_PRICE:
        claims.append({
            "id": f"price_itself:{len(claims)}", "code": "price_itself", "category": "price",
            "label": "قیمت اعلام‌شده", "evidence": ad.price_text or str(ad.price), "match": ad.price_text,
            "status": "implausible",
            "reason": f"{abs(diff) * 100:.0f}٪ زیر قیمت منصفانه؛ احتمال قیمت طعمه، آگهی جعلی یا ایراد پنهان",
            "features": {**context, "code": "price_itself", "status": "implausible", "claim_text": ""},
            "learned_false_rate": None, "learned_n": 0, "p_false": None,
            "status_fa": STATUS_FA["implausible"], "icon": STATUS_ICON["implausible"],
        })

    counts = {s: sum(c["status"] == s for c in claims) for s in STATUS_FA}

    # ---------- ۱) سازگاری ----------
    contradiction_findings = [f for f in text.findings if f.code.startswith("contradiction")]
    consistency = 100 - 20 * len(contradiction_findings) - 15 * counts["contradicted"]
    consistency = max(0, consistency)
    checked = counts["consistent"] + counts["contradicted"]
    if ad.has_seller_eval and not contradiction_findings:
        checked += len(td.get("text_damages", []))   # خسارتی که در متن گفته و با فرم می‌خواند
    if not checked and not contradiction_findings and not ad.has_seller_eval:
        consistency = 50   # هیچ ادعایی قابل مقایسه نبود: نه خوب، نه بد
    elif not checked and not contradiction_findings:
        consistency = min(consistency, 80)

    # ---------- ۲) باورپذیری ----------
    bait_price = any(c["code"] == "price_itself" for c in claims)
    plausibility = 100 - 20 * counts["implausible"] - (25 if cheap_perfect else 0) - (20 if bait_price else 0)
    if ad.image_count < 4 and len(re.findall(L.PUFFERY, ndesc)) >= 3:
        plausibility -= 15
    for c in claims:   # ادعاهای غیرقابل بررسی که در تجربه شما اغلب دروغ بوده‌اند
        p = c["p_false"] if c["p_false"] is not None else c["learned_false_rate"]
        if c["status"] == "unverifiable" and p is not None and p > 0.3:
            plausibility -= round(30 * (p - 0.3))
    plausibility = max(0, plausibility)

    # ---------- ۳) جزئیات مشخص در برابر تعریف کلی ----------
    puffery = re.findall(L.PUFFERY, ndesc)
    specifics = set(re.findall(L.SPECIFICS, ndesc))
    admitted = len(td.get("text_damages", [])) + td.get("unspecified_paint_spots", 0) + \
        sum(1 for f in text.findings if f.code in ("needs_repair", "engine_rebuild", "gearbox_repair",
                                                    "head_gasket", "oil_burning", "overheat"))
    n_spec = len(specifics) + admitted
    if len(desc.strip()) < 30:
        specificity = 30
    else:
        specificity = round(100 * (n_spec + 1) / (n_spec + len(puffery) + 2))

    # ---------- ۴) شفافیت ----------
    transparency = 50
    notes_t = []
    if admitted:
        transparency += 20
        notes_t.append(f"خودش {admitted} ایراد/خسارت را اعلام کرده (+)")
    if "pos_inspection_ok" in text_codes:
        transparency += 15
        notes_t.append("آماده کارشناسی است (+)")
    if ad.has_seller_eval:
        transparency += 10
        notes_t.append("فرم ارزیابی را پر کرده (+)")
    else:
        transparency -= 10
        notes_t.append("فرم ارزیابی را پر نکرده (−)")
    if ad.image_count >= 8:
        transparency += 5
    elif ad.image_count < 3:
        transparency -= 10
        notes_t.append("عکس کم (−)")
    if ad.ownership and "مالک" not in ad.ownership:
        transparency -= 10
        notes_t.append("مالک نیست (−)")
    for code, fa, penalty, risk_pts, pattern in L.AVOIDANCE:
        m = re.search(pattern, ndesc)
        if m:
            transparency -= penalty
            notes_t.append(f"{fa} (−)")
            if risk_pts:
                res.add(Finding("honesty", code, fa, points=risk_pts, evidence=m.group(0),
                                detail="فروشنده‌ای که از بررسی فرار می‌کند معمولاً چیزی برای پنهان کردن دارد.",
                                question="اجازه می‌دهید خودرو را کارشناس خودم بررسی کند؟"))
    transparency = max(0, min(100, transparency))

    scores = {"consistency": consistency, "plausibility": plausibility,
              "specificity": specificity, "transparency": transparency}
    total = round(sum(scores[k] * w for k, w in WEIGHTS.items()))
    if consistency < 100 and (contradiction_findings or counts["contradicted"]):
        total = min(total, 74)   # آگهی‌ای که با خودش تناقض دارد «قابل اعتماد» نیست
    if bait_price:
        total = min(total, 60)   # قیمت غیرواقعی خودش یک ادعای نادرست است
    if total >= 75:
        label, level = "قابل اعتماد", "low"
    elif total >= 50:
        label, level = "متوسط — بخشی از ادعاها باید بررسی شود", "medium"
    else:
        label, level = "کم — ادعاهای آگهی قابل اتکا نیست", "high"

    # یافته‌های ریسک (تناقض‌ها قبلاً در مرحله ۲ امتیاز گرفته‌اند؛ اینجا فقط موارد جدید)
    if cheap_perfect:
        res.add(Finding("honesty", "perfect_but_cheap",
                        f"ادعای سالم بودن کامل، اما قیمت {abs(diff) * 100:.0f}٪ زیر بازار",
                        detail="ترکیب «همه‌چیز عالی» و «خیلی ارزان» قوی‌ترین نشانه پنهان‌کاری یا کلاهبرداری است.",
                        points=10, level="high", question="اگر خودرو سالم است، چرا زیر قیمت بازار می‌فروشید؟"))
    for c in claims:
        if c["status"] == "implausible" and c["code"] in PERFECT_CLAIMS | {"km_real", "first_owner"}:
            res.add(Finding("honesty", f"implausible_{c['code']}", f"ادعای بعید: «{c['match']}»",
                            detail=c["reason"], points=3, level="low", evidence=c["evidence"]))
    for c in claims:
        q = CLAIM_QUESTIONS.get(c["code"])
        if q and c["status"] in ("unverifiable", "implausible"):
            res.add(Finding("honesty", f"ask_{c['code']}", f"ادعای «{c['match']}» باید با مدرک تأیید شود",
                            points=0, level="info", question=q, evidence=c["evidence"]))
    if total < 50:
        res.add(Finding("honesty", "low_honesty", f"امتیاز صداقت آگهی پایین است ({total} از ۱۰۰)",
                        points=6, level="medium"))

    res.data = {
        "score": total, "label": label, "level": level, "scores": scores,
        "score_labels": {"consistency": "سازگاری اطلاعات", "plausibility": "باورپذیری ادعاها",
                         "specificity": "جزئیات مشخص به‌جای تعریف کلی", "transparency": "شفافیت"},
        "claims": claims, "counts": counts,
        "puffery": sorted(set(puffery)), "n_puffery": len(puffery),
        "specifics": sorted(specifics), "n_specific": n_spec, "admitted_flaws": admitted,
        "transparency_notes": notes_t, "cheap_perfect": cheap_perfect,
        "n_contradiction_findings": len(contradiction_findings),
    }
    return res
