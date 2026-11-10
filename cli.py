"""
رابط خط فرمان.

  python cli.py analyze <لینک دیوار> [--budget-max 1500] [--usage city] [--images] [--json]
  python cli.py label <توکن یا لینک> --hidden-issue yes --severity 2 --notes "گلگیر عقب جوش داشت"
  python cli.py train
  python cli.py serve [--port 8000]

بودجه به «میلیون تومان» وارد می‌شود (1500 = یک‌ونیم میلیارد).
"""
import argparse
import json
import sys
import webbrowser

from app import config
from app.divar_client import DivarError, extract_token
from app.stage1_needs import CustomerNeeds, parse_priorities


def _needs_from_args(a) -> CustomerNeeds:
    return CustomerNeeds(
        budget_min=int(a.budget_min * 1_000_000) if a.budget_min else None,
        budget_max=int(a.budget_max * 1_000_000) if a.budget_max else None,
        usage=a.usage, family_size=a.family, gearbox=a.gearbox, max_km=a.max_km, min_year=a.min_year,
        priorities=parse_priorities(a.priorities), risk_tolerance=a.risk_tolerance,
    )


def cmd_analyze(a) -> int:
    from app.pipeline import analyze
    from app.report import render_report

    try:
        report = analyze(a.url, _needs_from_args(a), analyze_images=a.images, use_cache=not a.no_cache)
    except DivarError as exc:
        print(f"خطا: {exc}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    s = report["summary"]
    ad = report["ad"]
    print(f"\n{ad['title']}  ({ad['brand_model_fa']}، {ad['year_jalali']}، {ad['km'] or 0:,} کیلومتر)")
    h = report["stages"]["honesty"]["data"]
    print(f"صداقت آگهی: {h['score']}/100 — {h['label']}")
    for c in h["claims"]:
        print(f"  {c['icon']} {c['label']} [{c['id']}]: {c['reason']}")
    print(f"سطح ریسک: {s['tier_fa']}   امتیاز: {s['final_score']}/100   اطمینان تحلیل: {s['confidence']}٪")
    if s["deal_label"]:
        print(f"ارزیابی قیمت: {s['deal_label']}")
    print("\nمهم‌ترین دلایل:")
    for f in s["top_risks"]:
        print(f"  [{f['points']:+.0f}] {f['title']}")
    if s["questions"]:
        print("\nسؤال از فروشنده:")
        for q in s["questions"]:
            print(f"  - {q}")
    print(f"\nتوصیه: {s['recommendation']}")
    path = render_report(report, save=True)
    print(f"\nگزارش کامل: {path}")
    if a.open:
        webbrowser.open(path.as_uri())
    return 0


def cmd_label(a) -> int:
    from app import db

    from app.web import save_claim_verdicts

    token = extract_token(a.token)
    db.save_inspection(token, a.hidden_issue == "yes", a.severity, a.notes or "")
    n = 0
    if a.claims:
        report = db.latest_report(token) or {}
        claims = report.get("stages", {}).get("honesty", {}).get("data", {}).get("claims", [])
        by_code = {c["code"]: c["id"] for c in claims} | {c["id"]: c["id"] for c in claims}
        verdicts = {}
        for item in a.claims.split(","):
            key, _, val = item.strip().partition("=")
            if key in by_code and val in ("true", "false", "1", "0"):
                verdicts[by_code[key]] = "1" if val in ("true", "1") else "0"
            else:
                print(f"ادعای ناشناخته یا مقدار نامعتبر: {item}")
        n = save_claim_verdicts(token, verdicts)
    print(f"نتیجه کارشناسی برای {token} ثبت شد ({n} ادعا).")
    return 0


def cmd_train(_a) -> int:
    from app.ml import claim_model, price_model, risk_model

    print("مدل قیمت:", json.dumps(price_model.train(), ensure_ascii=False))
    print("مدل ریسک:", json.dumps(risk_model.train(), ensure_ascii=False))
    print("مدل صداقت ادعاها:", json.dumps(claim_model.train(), ensure_ascii=False))
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    uvicorn.run("app.web:app", host=a.host, port=a.port, reload=False)
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="تحلیل ریسک آگهی خودرو در دیوار")
    sub = p.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="تحلیل یک آگهی")
    an.add_argument("url")
    an.add_argument("--budget-min", type=float, help="میلیون تومان")
    an.add_argument("--budget-max", type=float, help="میلیون تومان")
    an.add_argument("--usage", default="mixed", choices=["city", "highway", "mixed", "ride_hailing"])
    an.add_argument("--family", type=int, default=4)
    an.add_argument("--gearbox", default="any", choices=["any", "manual", "automatic"])
    an.add_argument("--max-km", type=int)
    an.add_argument("--min-year", type=int)
    an.add_argument("--priorities", help="resale,low_maintenance,fuel_economy,safety,comfort,durability")
    an.add_argument("--risk-tolerance", default="low", choices=["low", "medium", "high"])
    an.add_argument("--images", action="store_true", help="تحلیل عکس‌ها (کندتر)")
    an.add_argument("--no-cache", action="store_true")
    an.add_argument("--json", action="store_true")
    an.add_argument("--open", action="store_true", help="باز کردن گزارش در مرورگر")
    an.set_defaults(func=cmd_analyze)

    lb = sub.add_parser("label", help="ثبت نتیجه کارشناسی حضوری (داده آموزشی ML)")
    lb.add_argument("token")
    lb.add_argument("--hidden-issue", required=True, choices=["yes", "no"],
                    help="آیا کارشناسی ایراد اعلام‌نشده پیدا کرد؟")
    lb.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3])
    lb.add_argument("--notes")
    lb.add_argument("--claims", help="نتیجه هر ادعا، مثل: body_no_paint=false,engine_ok=true")
    lb.set_defaults(func=cmd_label)

    tr = sub.add_parser("train", help="آموزش مدل‌های قیمت و ریسک")
    tr.set_defaults(func=cmd_train)

    sv = sub.add_parser("serve", help="اجرای وب‌سایت")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    a = p.parse_args()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
