"""وب‌سایت و API (FastAPI)."""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from . import db
from .divar_client import DivarError, extract_token
from .pipeline import analyze
from .report import env, render_report
from .stage1_needs import PRIORITY_FA, USAGE_FA, CustomerNeeds, parse_priorities
from .stage6_scoring import TIER_FA

app = FastAPI(title="ریسک‌سنج آگهی خودرو دیوار")


def _page(name: str, **ctx) -> HTMLResponse:
    ctx.setdefault("standalone", False)
    return HTMLResponse(env.get_template(name).render(**ctx))


def _million(v: str | None) -> int | None:
    try:
        return int(float(v) * 1_000_000) if v not in (None, "") else None
    except ValueError:
        return None


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


@app.get("/", response_class=HTMLResponse)
def index(msg: str | None = None):
    return _page("index.html", form={}, usage_options=USAGE_FA, priority_options=PRIORITY_FA,
                 error=None, msg=msg)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_form(request: Request):
    form = await request.form()
    data = {k: form.get(k) for k in form.keys()}
    data["priorities"] = form.getlist("priorities")
    needs = CustomerNeeds(
        budget_min=_million(data.get("budget_min")), budget_max=_million(data.get("budget_max")),
        usage=data.get("usage") or "mixed", family_size=_int(data.get("family")) or 4,
        gearbox=data.get("gearbox") or "any", max_km=_int(data.get("max_km")),
        min_year=_int(data.get("min_year")), priorities=parse_priorities(data["priorities"]),
        risk_tolerance=data.get("risk_tolerance") or "low",
    )
    try:
        report = analyze(data.get("url", ""), needs, analyze_images=bool(data.get("images")))
    except DivarError as exc:
        return _page("index.html", form=data, usage_options=USAGE_FA, priority_options=PRIORITY_FA,
                     error=str(exc))
    render_report(report, save=True)
    return HTMLResponse(render_report(report, standalone=False))


class NeedsIn(BaseModel):
    budget_min: int | None = Field(None, description="تومان")
    budget_max: int | None = Field(None, description="تومان")
    usage: str = "mixed"
    family_size: int = 4
    gearbox: str = "any"
    max_km: int | None = None
    min_year: int | None = None
    priorities: list[str] = []
    risk_tolerance: str = "low"


class AnalyzeIn(BaseModel):
    url: str
    needs: NeedsIn = NeedsIn()
    images: bool = False


@app.post("/api/analyze")
def analyze_api(body: AnalyzeIn):
    try:
        report = analyze(body.url, CustomerNeeds(**body.needs.model_dump()), analyze_images=body.images)
    except DivarError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(json.loads(json.dumps(report, ensure_ascii=False, default=str)))


@app.post("/label")
async def label(request: Request):
    form = await request.form()
    token = extract_token(str(form.get("token", "")))
    db.save_inspection(token, form.get("hidden_issue") == "yes", int(form.get("severity") or 0),
                       str(form.get("notes") or ""))
    save_claim_verdicts(token, {k.removeprefix("claim__"): v for k, v in form.items() if k.startswith("claim__")})
    return RedirectResponse("/history?saved=1", status_code=303)


def save_claim_verdicts(token: str, verdicts: dict[str, str]) -> int:
    """verdicts: {claim_id: "1" درست / "0" نادرست / "" بررسی نشد}"""
    report = db.latest_report(token)
    claims = {c["id"]: c for c in (report or {}).get("stages", {}).get("honesty", {}).get("data", {})
              .get("claims", [])}
    saved = 0
    for claim_id, v in verdicts.items():
        if v in ("0", "1") and claim_id in claims:
            db.save_claim_check(token, claims[claim_id], v == "1")
            saved += 1
    return saved


@app.get("/insights", response_class=HTMLResponse)
def insights():
    from .lexicon import CLAIMS

    names = {code: fa for code, _, fa, _ in CLAIMS} | {"parts_no_paint": "قطعات مشخص بدون رنگ",
                                                       "price_itself": "قیمت اعلام‌شده"}
    stats = db.claim_stats()
    rows = sorted(
        ({"code": k, "name": names.get(k, k), "n": s["n"], "n_false": s["n_false"],
          "rate": s["n_false"] / s["n"] if s["n"] else 0} for k, s in stats.items()),
        key=lambda r: (-r["rate"], -r["n"]))
    return _page("insights.html", rows=rows, total=sum(r["n"] for r in rows))


@app.get("/history", response_class=HTMLResponse)
def history(saved: int = 0):
    with db.connect() as c:
        rows = c.execute(
            """SELECT a.token, a.analyzed_at, a.brand_model, a.year, a.price, a.risk_score, a.tier, a.honesty_score,
                      i.hidden_issue, i.severity
               FROM analyses a LEFT JOIN inspections i ON i.token = a.token
               WHERE a.id IN (SELECT MAX(id) FROM analyses GROUP BY token)
               ORDER BY a.analyzed_at DESC LIMIT 200"""
        ).fetchall()
        labeled = c.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    items = [dict(r) | {"when": datetime.fromtimestamp(r["analyzed_at"]).strftime("%Y-%m-%d %H:%M"),
                        "tier_fa": TIER_FA.get(r["tier"], "")} for r in rows]
    return _page("history.html", items=items, labeled=labeled, saved=saved)
