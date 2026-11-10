"""ساخت گزارش HTML از خروجی تحلیل."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config
from .text_utils import fmt_toman

env = Environment(loader=FileSystemLoader(config.BASE_DIR / "templates"), autoescape=select_autoescape())
env.filters["toman"] = fmt_toman
env.filters["pct"] = lambda v: "—" if v is None else f"{v * 100:+.0f}٪"
env.filters["num"] = lambda v: "—" if v is None else f"{int(v):,}"


def price_chart(p: dict, ad: dict) -> dict | None:
    """نمودار پراکندگی قیمت/کارکرد آگهی‌های مشابه (SVG ساده سمت سرور)."""
    pts = p.get("points") or []
    if len(pts) < 3 or ad.get("km") is None:
        return None
    w, h, pad_l, pad_r, pad_t, pad_b = 640, 260, 70, 16, 14, 34
    kms = [x["km"] for x in pts] + [ad["km"]]
    prices = [x["price"] for x in pts] + [ad["price"], p["fair_price"]]
    kmin, kmax = min(kms), max(kms) or 1
    pmin, pmax = min(prices) * 0.97, max(prices) * 1.03
    kspan = (kmax - kmin) or 1
    pspan = (pmax - pmin) or 1

    def sx(km):
        return pad_l + (km - kmin) / kspan * (w - pad_l - pad_r)

    def sy(price):
        return pad_t + (1 - (price - pmin) / pspan) * (h - pad_t - pad_b)

    yticks = [pmin + pspan * i / 4 for i in range(5)]
    xticks = [kmin + kspan * i / 4 for i in range(5)]
    return {
        "w": w, "h": h, "pad_l": pad_l, "pad_b": pad_b, "pad_t": pad_t, "pad_r": pad_r,
        "dots": [(round(sx(x["km"]), 1), round(sy(x["price"]), 1)) for x in pts],
        "ask": (round(sx(ad["km"]), 1), round(sy(ad["price"]), 1)),
        "fair": (round(sx(ad["km"]), 1), round(sy(p["fair_price"]), 1)),
        "yticks": [(round(sy(v), 1), f"{v / 1e9:.2f}") for v in yticks],
        "xticks": [(round(sx(v), 1), f"{v / 1000:.0f}") for v in xticks],
    }


STAGE_ORDER = ["needs", "text", "body", "pricing", "behavior", "images", "scoring", "knowledge"]


def render_report(report: dict, save: bool = False, standalone: bool = True) -> Path | str:
    pricing = report["stages"]["pricing"]["data"]
    chart = price_chart(pricing, report["ad"]) if pricing.get("available") else None
    html = env.get_template("report.html").render(
        r=report, s=report["summary"], ad=report["ad"], st=report["stages"], chart=chart,
        order=STAGE_ORDER, standalone=standalone,
        contras=[f for f in report["stages"]["text"]["findings"] if f["code"].startswith("contradiction")],
    )
    if not save:
        return html
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "").replace("-", "")
    path = config.REPORTS_DIR / f"{report['ad']['token']}_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    return path
