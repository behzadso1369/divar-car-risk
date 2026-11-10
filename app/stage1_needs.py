"""مرحله ۱: نیاز مشتری — پروفایل خریدار و میزان تطابق این آگهی با آن."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Ad, Finding, StageResult
from .stage7_knowledge import load_kb
from .text_utils import fmt_toman, normalize

USAGE_FA = {
    "city": "شهری", "highway": "جاده‌ای / سفر", "mixed": "ترکیبی", "ride_hailing": "تاکسی اینترنتی / کار",
}
PRIORITY_FA = {
    "resale": "نقدشوندگی و حفظ ارزش", "low_maintenance": "هزینه نگهداری پایین",
    "fuel_economy": "مصرف سوخت پایین", "safety": "ایمنی", "comfort": "راحتی", "durability": "دوام فنی",
}
RISK_TOLERANCE_FA = {"low": "کم", "medium": "متوسط", "high": "زیاد"}
# اولویت‌های ضمنی هر نوع استفاده
USAGE_PRIORITIES = {
    "city": ["fuel_economy", "low_maintenance"],
    "highway": ["safety", "comfort"],
    "ride_hailing": ["durability", "fuel_economy", "low_maintenance"],
    "mixed": [],
}


@dataclass
class CustomerNeeds:
    budget_min: int | None = None        # تومان
    budget_max: int | None = None        # تومان
    usage: str = "mixed"
    family_size: int = 4
    gearbox: str = "any"                 # manual / automatic / any
    max_km: int | None = None
    min_year: int | None = None
    priorities: list[str] = field(default_factory=list)
    risk_tolerance: str = "low"

    @property
    def is_empty(self) -> bool:
        return not (self.budget_max or self.budget_min or self.priorities or self.max_km or self.min_year
                    or self.gearbox != "any" or self.usage != "mixed")

    def summary(self) -> list[str]:
        out = []
        if self.budget_min or self.budget_max:
            out.append(f"بودجه: {fmt_toman(self.budget_min) if self.budget_min else '—'} تا "
                       f"{fmt_toman(self.budget_max) if self.budget_max else '—'}")
        out.append(f"نوع استفاده: {USAGE_FA.get(self.usage, self.usage)}")
        out.append(f"تعداد سرنشین معمول: {self.family_size}")
        if self.gearbox != "any":
            out.append(f"گیربکس: {'اتوماتیک' if self.gearbox == 'automatic' else 'دنده‌ای'}")
        if self.max_km:
            out.append(f"حداکثر کارکرد: {self.max_km:,} کیلومتر")
        if self.min_year:
            out.append(f"حداقل سال: {self.min_year}")
        if self.priorities:
            out.append("اولویت‌ها: " + "، ".join(PRIORITY_FA.get(p, p) for p in self.priorities))
        out.append(f"ریسک‌پذیری: {RISK_TOLERANCE_FA.get(self.risk_tolerance, self.risk_tolerance)}")
        return out


def run(ad: Ad, needs: CustomerNeeds, kb_entry: dict, fair_price: float | None) -> StageResult:
    res = StageResult("needs", "۱. نیاز مشتری")
    if needs.is_empty:
        res.data = {"fit_score": None, "summary": needs.summary(), "empty": True}
        res.add(Finding("needs", "needs_empty", "پروفایل نیاز مشتری وارد نشده؛ فقط ریسک آگهی سنجیده شد."))
        return res

    score, weight = 0.0, 0.0

    def check(ok: bool, w: float, good: str, bad: str, code: str, partial: float | None = None) -> None:
        nonlocal score, weight
        weight += w
        if ok:
            score += w
            res.add(Finding("needs", code, good, level="positive"))
        else:
            score += w * (partial or 0)
            res.add(Finding("needs", code, bad, level="medium"))

    # بودجه
    if ad.price and (needs.budget_max or needs.budget_min):
        within_max = not needs.budget_max or ad.price <= needs.budget_max
        if within_max:
            check(True, 3, f"قیمت ({fmt_toman(ad.price)}) در بودجه است", "", "budget")
        elif fair_price and needs.budget_max and fair_price <= needs.budget_max:
            check(False, 3, "", f"قیمت آگهی بالاتر از بودجه است، اما قیمت منصفانه ({fmt_toman(fair_price)}) "
                               f"در بودجه است؛ با چانه‌زنی ممکن است", "budget", partial=0.6)
        else:
            over = (ad.price - needs.budget_max) / needs.budget_max * 100
            check(False, 3, "", f"قیمت {over:.0f}٪ بالاتر از سقف بودجه است", "budget")
        if needs.budget_min and ad.price < needs.budget_min * 0.85:
            res.add(Finding("needs", "budget_below_min",
                            "قیمت خیلی پایین‌تر از حداقل بودجه است؛ ممکن است با بودجه شما خودروی بهتری پیدا شود",
                            level="info"))

    if needs.gearbox != "any" and ad.gearbox:
        wants_auto = needs.gearbox == "automatic"
        check(wants_auto == ad.is_automatic, 2, f"گیربکس ({ad.gearbox}) مطابق خواسته است",
              f"گیربکس ({ad.gearbox}) مطابق خواسته مشتری نیست", "gearbox")

    if needs.max_km and ad.km is not None:
        ok = ad.km <= needs.max_km
        check(ok, 1.5, f"کارکرد ({ad.km:,}) در محدوده مطلوب است",
              f"کارکرد ({ad.km:,}) بیشتر از حداکثر مطلوب ({needs.max_km:,}) است", "max_km",
              partial=0.5 if ad.km <= needs.max_km * 1.2 else 0)

    if needs.min_year and ad.year_jalali:
        check(ad.year_jalali >= needs.min_year, 1.5, f"سال ساخت ({ad.year_jalali}) مطابق خواسته است",
              f"سال ساخت ({ad.year_jalali}) قدیمی‌تر از حداقل مطلوب ({needs.min_year}) است", "min_year",
              partial=0.5 if ad.year_jalali >= needs.min_year - 1 else 0)

    seats = kb_entry.get("seats", 5)
    if needs.family_size > seats:
        check(False, 2, "", f"ظرفیت خودرو ({seats} نفر) برای {needs.family_size} سرنشین کافی نیست", "seats")
    elif needs.family_size >= 5 and kb_entry.get("segment") == "small":
        check(False, 1, "", "خودروی کوچک برای خانواده پنج‌نفره کمی تنگ است", "seats", partial=0.5)

    # اولویت‌ها در برابر امتیاز مدل (پایگاه دانش)
    ratings = kb_entry.get("ratings", {})
    labels = load_kb().get("rating_labels", PRIORITY_FA)
    wanted = list(dict.fromkeys(needs.priorities + USAGE_PRIORITIES.get(needs.usage, [])))
    for p in wanted:
        r = ratings.get(p)
        if r is None:
            continue
        name = labels.get(p, p)
        w = 1.5 if p in needs.priorities else 1.0
        weight += w
        score += w * (r - 1) / 4
        if r >= 4:
            res.add(Finding("needs", f"prio_{p}", f"«{name}» این مدل خوب است ({r} از ۵)", level="positive"))
        elif r <= 2:
            res.add(Finding("needs", f"prio_{p}", f"«{name}» این مدل ضعیف است ({r} از ۵)", level="medium"))

    if needs.usage == "ride_hailing" and ad.is_automatic and ratings.get("low_maintenance", 3) <= 2:
        res.add(Finding("needs", "ride_hailing_auto",
                        "برای کار در تاکسی اینترنتی، گیربکس اتومات این مدل هزینه نگهداری بالایی دارد",
                        level="medium"))

    fit = round(score / weight * 100) if weight else None
    res.data = {"fit_score": fit, "summary": needs.summary(), "empty": False,
                "risk_tolerance": needs.risk_tolerance}
    return res


def recommend_models(needs: CustomerNeeds, top: int = 5) -> list[dict]:
    """پیشنهاد مدل بر اساس اولویت‌ها (بدون قیمت؛ قیمت‌ها با تورم سریع تغییر می‌کنند)."""
    kb = load_kb()
    wanted = list(dict.fromkeys(needs.priorities + USAGE_PRIORITIES.get(needs.usage, []))) or \
        ["resale", "low_maintenance", "durability"]
    scored = []
    seen = set()
    for e in kb["models"]:
        if needs.gearbox == "automatic" and e.get("gearbox") != "automatic":
            continue
        if needs.gearbox == "manual" and e.get("gearbox") == "automatic":
            continue
        if e["name"] in seen:
            continue
        seen.add(e["name"])
        r = e.get("ratings", {})
        s = sum(r.get(p, 3) * (1.5 if p in needs.priorities else 1.0) for p in wanted)
        s -= e.get("base_risk", 4) * 0.3
        if needs.family_size >= 5 and e.get("segment") == "small":
            s -= 1
        scored.append({"name": e["name"], "score": round(s, 1)})
    return sorted(scored, key=lambda x: -x["score"])[:top]


def parse_priorities(value: str | list | None) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else normalize(value).replace("،", ",").split(",")
    return [p.strip() for p in items if p and p.strip() in PRIORITY_FA]
