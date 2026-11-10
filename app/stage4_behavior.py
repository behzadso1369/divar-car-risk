"""مرحله ۴: نشانه‌های ریسک از رفتار آگهی و فروشنده."""
from __future__ import annotations

from datetime import date

from . import db
from .models import Ad, Finding, StageResult
from .text_utils import fmt_toman


def run(ad: Ad, pricing: StageResult) -> StageResult:
    res = StageResult("behavior", "۴. نشانه‌های ریسک از رفتار آگهی")

    # --- عکس‌ها ---
    if ad.image_count == 0:
        res.add(Finding("behavior", "no_images", "آگهی هیچ عکسی ندارد", points=12, level="high",
                        question="می‌توانید عکس از همه زوایای خودرو در نور روز بفرستید؟"))
    elif ad.image_count < 4:
        res.add(Finding("behavior", "few_images", f"فقط {ad.image_count} عکس", points=5, level="low",
                        question="عکس از زوایای دیگر (به‌خصوص ستون‌ها و سقف) دارید؟"))
    elif ad.image_count >= 8:
        res.add(Finding("behavior", "many_images", f"{ad.image_count} عکس (پوشش خوب)", points=-1,
                        level="positive"))

    # --- نوع فروشنده ---
    if ad.business_type and ad.business_type != "personal":
        res.add(Finding("behavior", "dealer", "فروشنده نمایشگاه یا کسب‌وکار است",
                        detail="نمایشگاه‌ها معمولاً خودرو را برای فروش آماده (رنگ و تعمیر جزئی) می‌کنند و "
                               "اطلاعات تاریخچه را کمتر دارند.", points=4, level="low",
                        question="خودرو از کجا خریداری شده و چند دست چرخیده؟"))
    if ad.ownership and "مالک" not in ad.ownership:
        res.add(Finding("behavior", "not_owner", f"آگهی‌دهنده مالک نیست («{ad.ownership}»)",
                        detail="معامله با واسطه احتمال اطلاعات ناقص یا قیمت اضافه را بالا می‌برد.",
                        points=6, level="medium", question="مالک خودرو کیست و در زمان معامله حضور دارد؟"))
    elif ad.ownership:
        res.add(Finding("behavior", "owner", "آگهی‌دهنده خودش مالک خودرو است", points=-1, level="positive"))

    # --- کیفیت توضیحات ---
    desc_len = len((ad.description or "").strip())
    if desc_len < 30:
        res.add(Finding("behavior", "desc_very_short", "توضیحات بسیار کوتاه یا خالی",
                        detail="فروشنده اطلاعات وضعیت خودرو را ارائه نکرده؛ ابهام = ریسک.", points=6, level="medium"))
    elif desc_len < 80:
        res.add(Finding("behavior", "desc_short", "توضیحات کوتاه", points=2, level="low"))

    # --- زمان و تبلیغ آگهی ---
    age_days = None
    if ad.published_at:
        age_days = (date.today() - ad.published_at).days
        if age_days > 21:
            res.add(Finding("behavior", "ad_old", f"آگهی {age_days} روز است که فروش نرفته",
                            detail="آگهی قدیمی در بازار فعال معمولاً یعنی قیمت بالا یا ایرادی که خریداران قبلی دیده‌اند.",
                            points=4, level="low", question="خریداران قبلی چرا منصرف شدند؟"))
    promotion = pricing.data.get("promotion")
    if promotion:
        res.add(Finding("behavior", "promoted", f"آگهی با پرداخت هزینه بالا آورده شده («{promotion}»)",
                        detail="نشانه تلاش برای فروش؛ به‌تنهایی مشکل نیست.", points=1, level="low"))

    # --- تاریخچه در دیتابیس خودمان ---
    history = db.price_history(ad.token)
    if len(history) >= 2:
        first, last = history[0][1], history[-1][1]
        if last < first:
            drop = (first - last) / first * 100
            res.add(Finding("behavior", "price_dropped",
                            f"قیمت از {fmt_toman(first)} به {fmt_toman(last)} کاهش یافته ({drop:.0f}٪)",
                            detail="فروشنده برای فروش انعطاف دارد؛ فرصت چانه‌زنی.", points=1, level="low"))
    reposts = db.find_reposts(ad.token, db.desc_hash(ad.description or ""))
    if reposts:
        res.add(Finding("behavior", "reposted",
                        f"همین متن قبلاً در {len(reposts)} آگهی دیگر دیده شده",
                        detail="حذف و ثبت دوباره آگهی برای پنهان کردن سابقه یا آگهی تکراری نمایشگاه.",
                        points=5, level="low", evidence="، ".join(reposts[:3])))

    # --- بیمه ---
    if ad.insurance_months == 0:
        res.add(Finding("behavior", "no_insurance", "بیمه شخص ثالث ندارد", detail="هزینه بیمه را به قیمت اضافه کنید.",
                        points=1, level="low"))

    res.data = {"ad_age_days": age_days, "promotion": promotion, "image_count": ad.image_count,
                "business_type": ad.business_type, "ownership": ad.ownership}
    return res
