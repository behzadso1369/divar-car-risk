"""ذخیره‌سازی محلی با SQLite: کش، آگهی‌های مشابه، تحلیل‌ها و نتایج کارشناسی (برچسب‌های ML)."""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comparables (
    token TEXT PRIMARY KEY,
    brand_model TEXT, year INTEGER, city_id TEXT,
    price INTEGER, km INTEGER, red_text TEXT, image_count INTEGER,
    title TEXT, first_seen REAL, last_seen REAL
);
CREATE INDEX IF NOT EXISTS ix_comp_model ON comparables(brand_model, year, city_id);
CREATE TABLE IF NOT EXISTS price_history (
    token TEXT, seen_at REAL, price INTEGER,
    PRIMARY KEY (token, seen_at)
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT, analyzed_at REAL, brand_model TEXT, year INTEGER,
    price INTEGER, risk_score REAL, tier TEXT, desc_hash TEXT,
    description TEXT, features_json TEXT, report_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_an_token ON analyses(token);
CREATE INDEX IF NOT EXISTS ix_an_hash ON analyses(desc_hash);
CREATE TABLE IF NOT EXISTS claim_checks (
    token TEXT, claim_id TEXT,
    code TEXT, claim_text TEXT,
    status TEXT,               -- وضعیت ادعا هنگام تحلیل (consistent / contradicted / ...)
    verdict INTEGER,           -- نتیجه کارشناسی: ۱ = درست بود، ۰ = نادرست بود
    features_json TEXT, labeled_at REAL,
    PRIMARY KEY (token, claim_id)
);
CREATE TABLE IF NOT EXISTS inspections (
    token TEXT PRIMARY KEY,
    labeled_at REAL,
    hidden_issue INTEGER,      -- ۱ = کارشناسی ایراد پنهان/اعلام‌نشده پیدا کرد
    severity INTEGER,          -- ۰ سالم، ۱ جزئی، ۲ متوسط، ۳ شدید
    notes TEXT
);
"""


@contextmanager
def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)")}
        if "honesty_score" not in cols:   # مهاجرت دیتابیس‌های قدیمی
            conn.execute("ALTER TABLE analyses ADD COLUMN honesty_score REAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def cache_get(key: str, max_age_hours: float):
    with connect() as c:
        row = c.execute("SELECT fetched_at, body FROM http_cache WHERE key=?", (key,)).fetchone()
    if row and time.time() - row["fetched_at"] < max_age_hours * 3600:
        return json.loads(row["body"])
    return None


def cache_set(key: str, value) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO http_cache(key, fetched_at, body) VALUES (?,?,?)",
            (key, time.time(), json.dumps(value, ensure_ascii=False)),
        )


def save_comparables(rows: list[dict], brand_model: str, city_id: str) -> None:
    now = time.time()
    with connect() as c:
        for r in rows:
            old = c.execute("SELECT price FROM comparables WHERE token=?", (r["token"],)).fetchone()
            if old is None or old["price"] != r["price"]:
                c.execute("INSERT OR IGNORE INTO price_history VALUES (?,?,?)", (r["token"], now, r["price"]))
            c.execute(
                """INSERT INTO comparables(token, brand_model, year, city_id, price, km, red_text,
                       image_count, title, first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(token) DO UPDATE SET price=excluded.price, km=excluded.km,
                       red_text=excluded.red_text, image_count=excluded.image_count,
                       last_seen=excluded.last_seen""",
                (r["token"], brand_model, r.get("year"), city_id, r["price"], r["km"],
                 r.get("red_text", ""), r.get("image_count", 0), r.get("title", ""), now, now),
            )


def price_history(token: str) -> list[tuple[float, int]]:
    with connect() as c:
        rows = c.execute(
            "SELECT seen_at, price FROM price_history WHERE token=? ORDER BY seen_at", (token,)
        ).fetchall()
    return [(r["seen_at"], r["price"]) for r in rows]


def desc_hash(description: str) -> str:
    from .text_utils import normalize
    norm = " ".join(normalize(description).split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest() if len(norm) >= 40 else ""


def find_reposts(token: str, dhash: str) -> list[str]:
    if not dhash:
        return []
    with connect() as c:
        rows = c.execute(
            "SELECT DISTINCT token FROM analyses WHERE desc_hash=? AND token<>?", (dhash, token)
        ).fetchall()
    return [r["token"] for r in rows]


def previous_analyses(token: str) -> list[sqlite3.Row]:
    with connect() as c:
        return c.execute(
            "SELECT analyzed_at, price, risk_score, tier FROM analyses WHERE token=? ORDER BY analyzed_at",
            (token,),
        ).fetchall()


def save_analysis(token, brand_model, year, price, risk_score, tier, description, features, report,
                  honesty_score=None) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO analyses(token, analyzed_at, brand_model, year, price, risk_score, tier,
                   desc_hash, description, features_json, report_json, honesty_score)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (token, time.time(), brand_model, year, price, risk_score, tier, desc_hash(description),
             description, json.dumps(features, ensure_ascii=False),
             json.dumps(report, ensure_ascii=False, default=str), honesty_score),
        )


def latest_report(token: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT report_json FROM analyses WHERE token=? ORDER BY id DESC LIMIT 1",
                        (token,)).fetchone()
    return json.loads(row["report_json"]) if row else None


def save_claim_check(token: str, claim: dict, verdict: bool) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO claim_checks VALUES (?,?,?,?,?,?,?,?)",
            (token, claim["id"], claim["code"], claim["evidence"], claim["status"], int(verdict),
             json.dumps(claim.get("features", {}), ensure_ascii=False), time.time()),
        )


def claim_stats() -> dict[str, dict]:
    """نرخ نادرستی هر نوع ادعا در کارشناسی‌های واقعی شما."""
    with connect() as c:
        rows = c.execute(
            "SELECT code, status, COUNT(*) n, SUM(1 - verdict) n_false FROM claim_checks GROUP BY code, status"
        ).fetchall()
    stats: dict[str, dict] = {}
    for r in rows:
        s = stats.setdefault(r["code"], {"n": 0, "n_false": 0, "by_status": {}})
        s["n"] += r["n"]
        s["n_false"] += r["n_false"]
        s["by_status"][r["status"]] = {"n": r["n"], "n_false": r["n_false"]}
    return stats


def claim_training_rows() -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT code, claim_text, status, verdict, features_json FROM claim_checks").fetchall()
    return [{**json.loads(r["features_json"] or "{}"), "code": r["code"], "claim_text": r["claim_text"],
             "status": r["status"], "label": 1 - r["verdict"]} for r in rows]


def save_inspection(token: str, hidden_issue: bool, severity: int, notes: str = "") -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO inspections VALUES (?,?,?,?,?)",
            (token, time.time(), int(hidden_issue), int(severity), notes),
        )


def labeled_training_rows() -> list[dict]:
    """آخرین تحلیل هر آگهی + نتیجه کارشناسی آن."""
    with connect() as c:
        rows = c.execute(
            """SELECT a.features_json, i.hidden_issue, i.severity
               FROM inspections i
               JOIN analyses a ON a.id = (SELECT MAX(id) FROM analyses WHERE token = i.token)"""
        ).fetchall()
    return [
        {**json.loads(r["features_json"]), "label": r["hidden_issue"], "severity": r["severity"]}
        for r in rows
    ]


def recent_comparables(max_age_days: int) -> list[sqlite3.Row]:
    cutoff = time.time() - max_age_days * 86400
    with connect() as c:
        return c.execute(
            "SELECT brand_model, year, city_id, price, km FROM comparables "
            "WHERE last_seen >= ? AND price > 0 AND km IS NOT NULL AND year IS NOT NULL",
            (cutoff,),
        ).fetchall()
