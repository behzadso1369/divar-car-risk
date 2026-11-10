"""تبدیل JSON ویجت‌های دیوار به ساختار Ad."""
import re

from .models import Ad
from .text_utils import normalize, parse_int, parse_jalali_datetime_text

# ویجت‌هایی که متن هشدار عمومی دیوار دارند و نباید با متن فروشنده قاطی شوند
_IGNORED_WIDGETS = {"INFO_BOX", "BREADCRUMB", "NOTE", "MAP_ROW"}


def _walk_lists(obj, visit):
    """همه لیست‌های ویجت را (حتی داخل مودال‌ها) به ترتیب پیمایش می‌کند."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("widgets", "widget_list") and isinstance(value, list):
                visit(value)
            _walk_lists(value, visit)
    elif isinstance(obj, list):
        for item in obj:
            _walk_lists(item, visit)


def parse_post(token: str, raw: dict) -> Ad:
    ad = Ad(token=token, url=f"https://divar.ir/v/{token}")
    we = raw.get("webengage", {}) or {}
    seo = raw.get("seo", {}) or {}
    ad.category = we.get("cat_2", "") or raw.get("analytics", {}).get("cat2", "")
    ad.brand_model_slug = we.get("brand_model", "")
    ad.business_type = we.get("business_type", "")
    ad.image_count = int(we.get("image_count") or 0)
    ad.city = (raw.get("city") or {}).get("name", "") or seo.get("web_info", {}).get("city_persian", "")
    ad.city_id = str((raw.get("city") or {}).get("city_id", ""))
    ad.district = seo.get("web_info", {}).get("district_persian", "")

    # --- بخش‌های اصلی ---
    for section in raw.get("sections", []):
        name = section.get("section_name")
        for w in section.get("widgets", []):
            wt, data = w.get("widget_type"), w.get("data", {})
            if name == "DESCRIPTION" and wt == "DESCRIPTION_ROW":
                ad.description = data.get("text", "")
            elif wt == "LEGEND_TITLE_ROW":
                ad.title = data.get("title", "")
            elif wt == "EXPANDABLE_SECTION" and name == "TITLE":
                ad.location_text = data.get("title", "")
                for inner in data.get("widget_list", []):
                    text = inner.get("data", {}).get("text", "")
                    for line in text.splitlines():
                        if "انتشار" in line:
                            ad.published_at = parse_jalali_datetime_text(line)
                        elif "روز" in line and "رسانی" in line:
                            ad.updated_at = parse_jalali_datetime_text(line)
            elif wt == "IMAGE_CAROUSEL":
                ad.image_urls = [it["image"]["url"] for it in data.get("items", [])
                                 if it.get("image", {}).get("url")]

    # --- همه ردیف‌ها (شامل مودال «سایر ویژگی‌ها») ---
    def visit(widgets: list) -> None:
        last_score_title = None
        for w in widgets:
            wt, data = w.get("widget_type"), w.get("data", {})
            if wt in _IGNORED_WIDGETS:
                continue
            if wt == "GROUP_INFO_ROW":
                for item in data.get("items", []):
                    ad.rows[item.get("title", "")] = item.get("value", "")
            elif wt == "UNEXPANDABLE_ROW":
                ad.rows[data.get("title", "")] = data.get("value", "")
            elif wt == "SCORE_ROW":
                last_score_title = data.get("title", "")
                ad.seller_eval[last_score_title] = {
                    "value": data.get("descriptive_score", "") or str(data.get("percentage_score", "")),
                    "tags": [],
                }
                continue
            elif wt == "TAG_LIST_ROW" and last_score_title:
                ad.seller_eval[last_score_title]["tags"] = [
                    re.sub(r"^\s*[\d۰-۹]+\s*[.\-)]\s*", "", t.get("text", ""))
                    for t in data.get("tags", [])
                ]
            elif wt == "WRAPPER_ROW":
                chips = data.get("chip_list", {}).get("chips", [])
                if not any(c.get("action") for c in chips):   # چیپ‌های لینک دسته‌بندی، امکانات نیستند
                    ad.options.extend(c.get("text", "") for c in chips)
            last_score_title = None

    _walk_lists(raw.get("sections", []), visit)

    rows = {normalize(k): v for k, v in ad.rows.items()}

    def row(*keys: str) -> str:
        for k in keys:
            for rk, rv in rows.items():
                if normalize(k) in rk:
                    return rv
        return ""

    ad.km = parse_int(row("کارکرد"))
    year_text = normalize(row("مدل (سال تولید)", "سال تولید"))
    years = [int(y) for y in re.findall(r"\d{4}", year_text)]
    jalali = [y for y in years if 1300 < y < 1500]
    if jalali:
        ad.year_jalali = jalali[0]
    elif years:
        ad.year_jalali = years[0] - 621
    ad.color = row("رنگ")
    ad.brand_model_fa = row("برند و مدل")
    ad.gearbox = row("گیربکس")
    ad.fuel = row("نوع سوخت")
    ins = row("بیمه")
    ad.insurance_months = parse_int(ins) if ins else None
    ad.ownership = row("مالکیت")
    ad.price_text = row("قیمت پایه", "قیمت")
    if "توافقی" in ad.price_text:
        ad.negotiable = True
    else:
        ad.price = parse_int(ad.price_text) or (we.get("price") or None)
    if not ad.price and we.get("price"):
        ad.price = int(we["price"])
    if ad.price is not None and ad.price < 10_000_000:   # «۱۰۰۰ تومان» = قیمت واقعی اعلام نشده (معاوضه)
        ad.price, ad.negotiable = None, True
    if not ad.image_count:
        ad.image_count = len(ad.image_urls)
    return ad
