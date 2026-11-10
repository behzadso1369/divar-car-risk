"""ارتباط با API عمومی دیوار: دریافت آگهی و جستجوی آگهی‌های مشابه برای قیمت‌گذاری."""
import json
import re
import time
from urllib.parse import unquote, urlparse

import requests

from . import config, db
from .text_utils import parse_int

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{5,20}$")
_last_request = 0.0
_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json"})


class DivarError(Exception):
    pass


def extract_token(url_or_token: str) -> str:
    """https://divar.ir/v/<slug>/<token> یا https://divar.ir/v/<token> یا خود توکن."""
    value = url_or_token.strip()
    if "/" not in value:
        token = value
    else:
        path = urlparse(value if "://" in value else "https://" + value).path
        segments = [unquote(s) for s in path.split("/") if s]
        if "v" not in segments:
            raise DivarError("این لینک آگهی دیوار نیست. لینک باید شبیه divar.ir/v/... باشد.")
        token = segments[-1]
    if not TOKEN_RE.match(token):
        raise DivarError(f"شناسه آگهی نامعتبر است: {token}")
    return token


def _throttle() -> None:
    global _last_request
    wait = config.REQUEST_DELAY_SEC - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def _request(method: str, url: str, **kwargs) -> dict:
    last_exc = None
    for attempt in range(3):
        _throttle()
        try:
            resp = _session.request(method, url, timeout=config.REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if resp.status_code == 404:
            raise DivarError("آگهی پیدا نشد؛ احتمالاً حذف یا منقضی شده است.")
        if resp.status_code == 429 or resp.status_code >= 500:
            last_exc = DivarError(f"دیوار پاسخ {resp.status_code} داد")
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code != 200:
            raise DivarError(f"خطای دیوار: HTTP {resp.status_code}")
        return resp.json()
    raise DivarError(f"ارتباط با دیوار برقرار نشد: {last_exc}")


def fetch_post(token: str, use_cache: bool = True) -> dict:
    key = f"post:{token}"
    if use_cache and (cached := db.cache_get(key, config.AD_CACHE_HOURS)):
        return cached
    data = _request("GET", config.DIVAR_POST_API.format(token=token))
    db.cache_set(key, data)
    return data


def _parse_post_row(widget: dict) -> dict | None:
    if widget.get("widget_type") != "POST_ROW":
        return None
    d = widget.get("data", {})
    token = d.get("token") or d.get("action", {}).get("payload", {}).get("token")
    price = parse_int(d.get("middle_description_text"))
    km_text = d.get("top_description_text", "")
    km = 0 if "صفر" in km_text else parse_int(km_text)
    if not token or not price or price < 10_000_000:   # توافقی یا قیمت نامعتبر
        return None
    return {
        "token": token,
        "title": d.get("title", ""),
        "price": price,
        "km": km,
        "red_text": d.get("red_text", ""),
        "image_count": d.get("image_count", 0),
    }


def search_comparables(brand_model: str, city_id: str, year: int,
                       max_pages: int = config.COMPARABLE_MAX_PAGES, use_cache: bool = True) -> list[dict]:
    """آگهی‌های هم‌مدل و هم‌سال در همان شهر (قیمت + کارکرد)."""
    key = f"search:{brand_model}:{city_id}:{year}:{max_pages}"
    if use_cache and (cached := db.cache_get(key, config.COMPARABLE_CACHE_HOURS)) is not None:
        return cached

    form = {
        "category": {"str": {"value": "light"}},
        "brand_model": {"repeated_string": {"value": [brand_model]}},
        "production-year": {"number_range": {"minimum": str(year), "maximum": str(year)}},
    }
    body: dict = {"city_ids": [str(city_id)], "search_data": {"form_data": {"data": form}}}
    results: dict[str, dict] = {}
    for _ in range(max_pages):
        data = _request("POST", config.DIVAR_SEARCH_API, data=json.dumps(body),
                        headers={"Content-Type": "application/json"})
        for w in data.get("list_widgets", []):
            row = _parse_post_row(w)
            if row:
                row["year"] = year
                results[row["token"]] = row
        pagination = data.get("pagination", {})
        if not pagination.get("has_next_page"):
            break
        body = {"city_ids": [str(city_id)], "pagination_data": pagination.get("data"),
                "search_data": data.get("search_data", body["search_data"])}

    rows = list(results.values())
    db.cache_set(key, rows)
    db.save_comparables(rows, brand_model, str(city_id))
    return rows
