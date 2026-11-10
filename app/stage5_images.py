"""
مرحله ۵ (اختیاری): تحلیل عکس‌ها با بینایی ماشین کلاسیک (OpenCV، رایگان و محلی).

چیزهایی که الان سنجیده می‌شود:
- عکس در نور کم (اختلاف رنگ بدنه پنهان می‌شود)
- عکس تار
- عکس تکراری (perceptual hash)
- وضوح پایین
نسخه بعدی: مدل تشخیص اختلاف رنگ پنل‌ها (YOLO/سگمنتیشن آموزش‌داده‌شده روی عکس‌های کارشناسی خودتان).
"""
from __future__ import annotations

import io

import numpy as np
import requests
from PIL import Image

from . import config
from .models import Ad, Finding, StageResult


def _dhash(img: Image.Image, size: int = 8) -> int:
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _blur_score(gray: np.ndarray) -> float:
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
        return float(lap.var())


def run(ad: Ad, enabled: bool) -> StageResult:
    res = StageResult("images", "۵. تحلیل عکس‌ها (اختیاری)")
    if not enabled:
        res.skipped, res.skip_reason = True, "تحلیل عکس فعال نشده است."
        return res
    if not ad.image_urls:
        res.skipped, res.skip_reason = True, "آگهی عکسی ندارد."
        return res

    stats = []
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    for url in ad.image_urls[: config.IMAGE_MAX_DOWNLOAD]:
        try:
            r = session.get(url, timeout=config.REQUEST_TIMEOUT)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.load()
        except Exception:  # noqa: BLE001 — یک عکس خراب نباید کل تحلیل را متوقف کند
            continue
        gray_img = img.convert("L")
        w, h = gray_img.size
        scale = 512 / max(w, h)
        if scale < 1:
            gray_img = gray_img.resize((int(w * scale), int(h * scale)))
        gray = np.asarray(gray_img, dtype=np.float64)
        stats.append({
            "url": url, "width": w, "height": h,
            "brightness": float(gray.mean()),
            "sharpness": _blur_score(gray),
            "hash": _dhash(img),
        })

    if not stats:
        res.skipped, res.skip_reason = True, "دانلود عکس‌ها ناموفق بود."
        return res

    n = len(stats)
    dark = [s for s in stats if s["brightness"] < 60]
    blurry = [s for s in stats if s["sharpness"] < 60]
    small = [s for s in stats if max(s["width"], s["height"]) < 500]
    dupes = 0
    for i in range(n):
        for j in range(i + 1, n):
            if bin(stats[i]["hash"] ^ stats[j]["hash"]).count("1") <= 4:
                dupes += 1
    unique = n - min(dupes, n - 1)

    if len(dark) / n >= 0.3:
        res.add(Finding("images", "dark_photos", f"{len(dark)} عکس از {n} عکس در نور کم یا شب گرفته شده",
                        detail="در نور کم اختلاف رنگ قطعات و موج بدنه دیده نمی‌شود؛ بازدید حتماً در روز باشد.",
                        points=5, level="low"))
    if len(blurry) / n >= 0.4:
        res.add(Finding("images", "blurry_photos", f"{len(blurry)} عکس تار است", points=3, level="low"))
    if dupes:
        res.add(Finding("images", "duplicate_photos", f"{dupes} جفت عکس تکراری یا تقریباً یکسان",
                        detail="تعداد واقعی زوایای پوشش‌داده‌شده کمتر از تعداد عکس‌هاست.", points=3, level="low"))
    if unique < 4:
        res.add(Finding("images", "few_angles", f"فقط حدود {unique} زاویه متفاوت از خودرو",
                        points=3, level="low", question="عکس از ستون‌ها، سقف و زیر خودرو می‌فرستید؟"))
    if len(small) / n >= 0.5:
        res.add(Finding("images", "low_res", "وضوح عکس‌ها پایین است", points=2, level="low"))
    if not res.findings:
        res.add(Finding("images", "photos_ok", "کیفیت عکس‌ها مناسب است (نور، وضوح، بدون تکرار)",
                        points=-1, level="positive"))

    res.data = {
        "analyzed": n, "dark": len(dark), "blurry": len(blurry), "duplicates": dupes,
        "avg_brightness": round(float(np.mean([s["brightness"] for s in stats])), 1),
        "images": [{k: v for k, v in s.items() if k != "hash"} for s in stats],
    }
    return res
