"""ساختارهای داده مشترک بین مراحل."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

LEVEL_ORDER = {"positive": -1, "info": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class Finding:
    """یک یافته (نشانه ریسک یا نشانه مثبت) با توضیح قابل فهم برای مشتری."""
    stage: str            # needs / text / body / pricing / behavior / images / knowledge
    code: str             # شناسه ماشینی، برای ویژگی‌های یادگیری ماشین
    title: str            # عنوان کوتاه فارسی
    detail: str = ""      # توضیح
    points: float = 0.0   # امتیاز ریسک (منفی = نشانه مثبت)
    level: str = "info"   # positive / info / low / medium / high
    question: str = ""    # سؤالی که خریدار باید از فروشنده بپرسد
    evidence: str = ""    # بخشی از متن آگهی که این یافته از آن آمده

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def level_for_points(points: float) -> str:
    if points < 0:
        return "positive"
    if points == 0:
        return "info"
    if points < 6:
        return "low"
    if points < 12:
        return "medium"
    return "high"


@dataclass
class DamageItem:
    part: str             # کلید قطعه، مثل fender
    part_fa: str
    dtype: str            # نوع خسارت، مثل replaced
    dtype_fa: str
    side: str | None = None       # driver / passenger
    position: str | None = None   # front / rear
    source: str = "text"          # text / form
    evidence: str = ""

    @property
    def label(self) -> str:
        side = {"driver": "سمت راننده", "passenger": "سمت شاگرد"}.get(self.side or "", "")
        pos = {"front": "جلو", "rear": "عقب"}.get(self.position or "", "")
        name = self.part_fa
        if pos and pos not in name:
            name += f" {pos}"
        if side:
            name += f" {side}"
        return name


@dataclass
class Ad:
    token: str
    url: str
    title: str = ""
    description: str = ""
    city: str = ""
    city_id: str = ""
    district: str = ""
    category: str = ""
    brand_model_slug: str = ""     # مثل "Peugeot 206 5"
    brand_model_fa: str = ""       # مثل "پژو 206 تیپ ۵"
    year_jalali: int | None = None
    km: int | None = None
    price: int | None = None
    price_text: str = ""
    negotiable: bool = False
    color: str = ""
    gearbox: str = ""
    fuel: str = ""
    insurance_months: int | None = None
    ownership: str = ""
    options: list[str] = field(default_factory=list)
    seller_eval: dict[str, dict] = field(default_factory=dict)  # عنوان → {value, tags}
    rows: dict[str, str] = field(default_factory=dict)          # همه ردیف‌های عنوان/مقدار
    image_urls: list[str] = field(default_factory=list)
    image_count: int = 0
    business_type: str = ""
    published_at: date | None = None
    updated_at: date | None = None
    location_text: str = ""

    @property
    def is_automatic(self) -> bool:
        return "اتومات" in self.gearbox

    @property
    def has_seller_eval(self) -> bool:
        return bool(self.seller_eval)


@dataclass
class StageResult:
    name: str
    title: str
    findings: list[Finding] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    def add(self, finding: Finding) -> None:
        if finding.level == "info" and finding.points:
            finding.level = level_for_points(finding.points)
        self.findings.append(finding)

    @property
    def points(self) -> float:
        return sum(f.points for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "points": round(self.points, 1),
            "findings": [f.to_dict() for f in self.findings],
            "data": self.data,
        }
