"""مرحله ۷: دانش اختصاصی هر مدل (ایرادات رایج، چک‌لیست، ریسک ذاتی، امتیازها)."""
import json
import re
from functools import lru_cache

from . import config
from .models import Ad, Finding, StageResult
from .text_utils import normalize


@lru_cache(maxsize=1)
def load_kb() -> dict:
    with open(config.KNOWLEDGE_FILE, encoding="utf-8") as f:
        return json.load(f)


def _has_word(text: str, word: str) -> bool:
    """تطبیق کلمه کامل («بنز» نباید در «بنزینی» پیدا شود)."""
    return re.search(rf"(?<!\w){re.escape(normalize(word))}(?!\w)", text) is not None


def find_entry(ad: Ad) -> dict:
    kb = load_kb()
    slug = (ad.brand_model_slug or "").lower()
    fa = normalize(f"{ad.brand_model_fa} {ad.title}")
    best, best_score = None, 0.0
    for entry in kb["models"]:
        score = 0.0
        for m in entry.get("match", []):
            if m and slug.startswith(m.lower()):
                score = max(score, 10 + len(m))
        kws = entry.get("fa_keywords", [])
        if kws and all(_has_word(fa, k) for k in kws):
            score = max(score, 5 + sum(len(k) for k in kws) / 10)
        if any(_has_word(fa, k) for k in entry.get("fa_any", [])):
            score = max(score, 5)
        if not score:
            continue
        gb = entry.get("gearbox")
        if gb == "automatic":
            if not ad.is_automatic:
                continue
            score += 3
        elif gb == "manual" and ad.is_automatic:
            continue
        if score > best_score:
            best, best_score = entry, score
    return best or kb["default"]


def run(ad: Ad, entry: dict) -> StageResult:
    result = StageResult("knowledge", "۷. دانش اختصاصی مدل")
    kb = load_kb()
    base = float(entry.get("base_risk", 4))
    result.add(Finding(
        stage="knowledge", code="model_base_risk",
        title=f"ریسک ذاتی مدل «{entry['name']}»: {base:g} از ۱۰",
        detail=entry.get("notes", ""), points=base,
        level="low" if base < 5 else "medium" if base < 7 else "high",
    ))
    result.data = {
        "model_id": entry["id"],
        "model_name": entry["name"],
        "base_risk": base,
        "ratings": entry.get("ratings", {}),
        "rating_labels": kb.get("rating_labels", {}),
        "common_issues": entry.get("common_issues", []),
        "checklist": entry.get("checklist", []) + kb.get("general_checklist", []),
        "is_default": entry["id"] == "default",
    }
    return result
