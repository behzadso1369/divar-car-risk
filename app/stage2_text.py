"""
مرحله ۲: استخراج اطلاعات از متن آگهی + دروغ‌سنجی.

دروغ‌سنجی = مقایسه سه منبع که فروشنده خودش پر کرده:
  ۱) متن آزاد توضیحات   ۲) فرم «ارزیابی فروشنده» دیوار (موتور/شاسی/بدنه/گیربکس)
  ۳) فیلدهای ساختاریافته (کارکرد، سال)
هر تناقض بین این‌ها یعنی یا بی‌دقتی یا پنهان‌کاری — و یک سؤال مشخص برای فروشنده.
"""
from __future__ import annotations

import re

from . import lexicon as L
from .models import Ad, DamageItem, Finding, StageResult
from .text_utils import current_jalali_year, normalize, split_clauses, word_or_digit_to_int

_PART_RES = [(k, fa, w, re.compile(p)) for k, fa, w, p in L.BODY_PARTS]
_PART_WEIGHT = {k: w for k, _, w, _ in L.BODY_PARTS}
_DMG_RES = [(k, fa, re.compile(p)) for k, fa, _, _, p in L.DAMAGE_TYPES]
_SIDE_RES = [(k, re.compile(p)) for k, p in L.SIDE_PATTERNS]
_POS_RES = [(k, re.compile(p)) for k, p in L.POSITION_PATTERNS]


# ---------------------------------------------------------------- ابزار استخراج
def _find_parts(clause: str) -> list[tuple[int, int, str, str]]:
    """قطعات بدنه در یک جمله، بدون هم‌پوشانی (الگوی خاص‌تر برنده است)."""
    found, taken = [], []
    for key, fa, _, rx in _PART_RES:
        for m in rx.finditer(clause):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.end(), key, fa))
    return sorted(found)


def _find_damage_types(segment: str) -> list[str]:
    seg = segment
    types = []
    for key, _, rx in _DMG_RES:
        for m in rx.finditer(seg):
            negated = (re.search(L.NEGATION_BEFORE, seg[max(0, m.start() - 6):m.start()])
                       or re.search(L.NEGATION_AFTER, seg[m.end():m.end() + 10]))
            if not negated:
                types.append(key)
                break
        seg = rx.sub(lambda m: " " * len(m.group(0)), seg)  # جلوگیری از تطبیق دوباره
    return types


_CONNECTOR = re.compile(r"\b(?:و|هر|دو|سمت|ها|های|هم|نیز)\b|[،,\s]|" + "|".join(
    p for _, p in L.SIDE_PATTERNS + L.POSITION_PATTERNS))


def _is_connector(segment: str) -> bool:
    """«و»، «سمت راننده»، فاصله... — یعنی قطعه با قطعه بعدی/قبلی یک خسارت مشترک دارد."""
    return not _CONNECTOR.sub("", segment).strip()


def _most_severe(types: list[str]) -> str | None:
    return max(types, key=lambda t: L.DAMAGE_RANK[t]) if types else None


def _detect(rxs, segment: str) -> str | None:
    hits = [k for k, rx in rxs if rx.search(segment)]
    return hits[0] if len(hits) == 1 else None


def extract_text_damages(text: str) -> tuple[list[DamageItem], int, list[str]]:
    """خسارت‌های بدنه از متن آزاد. خروجی: (اقلام، تعداد لکه بدون نام قطعه، جمله‌های طفره‌آمیز)"""
    items: list[DamageItem] = []
    unspecified = 0
    hedges: list[str] = []
    for clause in split_clauses(text):
        c = normalize(clause)
        parts = _find_parts(c)
        clause_types = _find_damage_types(c)
        if not clause_types:
            continue
        if re.search(L.HEDGE_WORDS, c):
            hedges.append(clause.strip())
        if not parts:
            if any(t in ("paint", "two_tone", "dent_repair", "pdr") for t in clause_types):
                m = re.search(L.DAMAGE_COUNT, c)
                n = word_or_digit_to_int(m.group(1)) if m else None
                if re.search(r"لکه|لک\b|رنگ", c):
                    unspecified += n or 1
            continue

        # هر قطعه: خسارتِ بعد از خودش («درب راننده تعویض»)؛ اولین قطعه: یا قبل از خودش («تعویض درب»)
        n = len(parts)
        afters = [c[e:(parts[i + 1][0] if i + 1 < n else len(c))] for i, (_, e, _, _) in enumerate(parts)]
        befores = [c[(parts[i - 1][1] if i else 0):s] for i, (s, _, _, _) in enumerate(parts)]
        dtypes = [_most_severe(_find_damage_types(a)) for a in afters]
        if dtypes[0] is None:
            dtypes[0] = _most_severe(_find_damage_types(befores[0]))
        # خسارت مشترک: «درب و گلگیر راننده تعویض» / «رنگ‌شدگی کاپوت و گلگیر»
        for i in range(n - 2, -1, -1):
            if dtypes[i] is None and _is_connector(afters[i]):
                dtypes[i] = dtypes[i + 1]
        for i in range(1, n):
            if dtypes[i] is None and _is_connector(befores[i]):
                dtypes[i] = dtypes[i - 1]

        clause_side = _detect(_SIDE_RES, c)
        for i, (s, e, key, fa) in enumerate(parts):
            dtype = dtypes[i]
            if dtype is None:
                continue
            side = _detect(_SIDE_RES, afters[i]) or _detect(_SIDE_RES, befores[i]) or clause_side
            if key == "rear_fender":
                pos = "rear"
            else:
                pos = _detect(_POS_RES, afters[i]) or (_detect(_POS_RES, befores[i]) if i == 0 else None)
            items.append(DamageItem(
                part=key, part_fa=fa, dtype=dtype, dtype_fa=L.DAMAGE_FA[dtype],
                side=side, position=pos, source="text", evidence=clause.strip(),
            ))
    return items, unspecified, hedges


def _form_dtype(value: str) -> str | None:
    v = normalize(value)
    if "سالم" in v and "خط" in v or v in ("سالم", "بدون رنگ"):
        return None
    for key, pattern in (("replaced", r"تعویض"), ("accident", r"تصادف|ضربه"), ("two_tone", r"دور ?رنگ"),
                         ("pdr", r"صافکاری (?:بی|بدون) ?رنگ"), ("paint", r"رنگ"),
                         ("dent_repair", r"صافکاری"), ("scratch", r"خط|خش")):
        if re.search(pattern, v):
            return key
    return None


def extract_form_damages(ad: Ad) -> tuple[list[DamageItem], int, bool, list[str]]:
    """
    خسارت‌های اعلام‌شده در فرم ارزیابی فروشنده.
    خروجی: (اقلام، تعداد ناحیه اعلام‌شده، آیا فرم بدنه «سالم» است، خسارت‌های کل‌خودرو)
    """
    items: list[DamageItem] = []
    declared = 0
    body_clean = False
    car_level: list[str] = []
    body_rows = {t: v for t, v in ad.seller_eval.items() if "بدنه" in t}
    if not body_rows:  # فرمت قدیمی دیوار: ردیف «وضعیت بدنه»
        legacy = ad.rows.get("وضعیت بدنه") or ad.rows.get("بدنه")
        if legacy:
            body_rows = {"بدنه": {"value": legacy, "tags": []}}
    for title, info in body_rows.items():
        value = normalize(info.get("value", ""))
        if re.search(r"سالم", value) and not re.search(r"رنگ|تعویض|صافکاری|تصادف", value):
            body_clean = True
            continue
        if re.search(r"تمام ?رنگ", value):
            car_level.append("full_paint")
        if re.search(r"اوراق", value):
            car_level.append("scrap")
        if re.search(r"تصادفی", value):
            car_level.append("accident_car")
        dtype = _form_dtype(value) or "paint"
        m = re.search(r"(\d+) ?ناحیه", value)
        declared += int(m.group(1)) if m else 0
        for tag in info.get("tags", []):
            t = normalize(tag)
            parts = _find_parts(t)
            tag_type = _form_dtype(t) or dtype
            for _, _, key, fa in parts[:1]:
                items.append(DamageItem(
                    part=key, part_fa=fa, dtype=tag_type, dtype_fa=L.DAMAGE_FA[tag_type],
                    side=_detect(_SIDE_RES, t),
                    position="rear" if key == "rear_fender" else _detect(_POS_RES, t),
                    source="form", evidence=tag,
                ))
    declared = max(declared, len(items))
    return items, declared, body_clean, car_level


def _compatible(a: DamageItem, b: DamageItem) -> bool:
    fam_a, fam_b = L.PART_FAMILY.get(a.part, a.part), L.PART_FAMILY.get(b.part, b.part)
    if fam_a != fam_b:
        return False
    if a.side and b.side and a.side != b.side:
        return False
    if a.position and b.position and a.position != b.position:
        return False
    return True


def merge_damages(form: list[DamageItem], text: list[DamageItem]) -> list[DamageItem]:
    merged = [DamageItem(**vars(f)) for f in form]
    for t in text:
        match = next((m for m in merged if _compatible(m, t)), None)
        if match is None:
            merged.append(DamageItem(**vars(t)))
        elif L.DAMAGE_RANK[t.dtype] > L.DAMAGE_RANK[match.dtype]:
            match.dtype, match.dtype_fa = t.dtype, t.dtype_fa
    return merged


def body_severity(items: list[DamageItem], unspecified: int, car_level: list[str]) -> float:
    sev = sum(_PART_WEIGHT.get(i.part, 1.0) * L.DAMAGE_MULT.get(i.dtype, 1.0) for i in items)
    sev += 0.8 * unspecified
    car_weights = {k: w for k, _, w, _ in L.CAR_LEVEL_DAMAGE}
    car_weights["accident_car"] = 4.0
    sev += sum(car_weights.get(k, 0) for k in set(car_level))
    return round(sev, 2)


# ---------------------------------------------------------------- اجرای مرحله
def _component_status(ad: Ad, keyword: str) -> tuple[str, str] | None:
    """اگر چند ردیف باشد (شاسی جلو / شاسی عقب)، بدترین وضعیت برگردانده می‌شود."""
    rows = [(t, i.get("value", "")) for t, i in ad.seller_eval.items() if keyword in t]
    rows = [r for r in rows if "تعیین" not in r[1]]   # «تعیین‌نشده» یعنی اطلاعات نداریم
    if not rows:
        return None
    bad = [r for r in rows if not re.search(r"^سالم", normalize(r[1]))]
    return bad[0] if bad else rows[0]


def _km_from_text(text: str) -> int | None:
    t = normalize(text)
    m = re.search(r"(?:کارکرد|کیلومتر|کارکردگی|کار کرد)\s*:?\s*(\d[\d,]*(?:\.\d+)?)\s*(هزار|k|کیلو)?", t)
    if not m:
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(هزار)\s*(?:کیلومتر|کارکرد|تا کارکرد|کار)", t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    if m.group(2) or num < 1000:
        num *= 1000
    return int(num)


def _year_from_text(title: str, desc: str) -> int | None:
    """سال از عنوان؛ وگرنه از توضیحات (به‌جز جمله‌های معاوضه که سال خودروی دیگری است)."""
    pattern = r"مدل\s*(13\d\d|14\d\d|20\d\d|19\d\d|\d\d)\b"
    m = re.search(pattern, normalize(title))
    if not m:
        for clause in split_clauses(desc):
            c = normalize(clause)
            if re.search(r"معاوضه|معاوظه|عوض با|تعویض با|به جای", c):
                continue
            m = re.search(pattern, c)
            if m:
                break
    if not m:
        return None
    y = int(m.group(1))
    if y >= 1900:
        return y - 621
    if y < 100:
        return 1300 + y if y >= 50 else 1400 + y
    return y


def run(ad: Ad) -> StageResult:
    res = StageResult("text", "۲. استخراج اطلاعات و دروغ‌سنجی متن آگهی")
    desc = ad.description or ""
    ndesc = normalize(desc)
    full = normalize(f"{ad.title}\n{desc}")

    text_items, unspecified, hedges = extract_text_damages(f"{ad.title}\n{desc}")
    form_items, form_declared, form_body_clean, form_car_level = extract_form_damages(ad)
    merged = merge_damages(form_items, text_items)

    car_level = list(form_car_level)
    for key, fa, _, pattern in L.CAR_LEVEL_DAMAGE:
        m = re.search(pattern, full)
        if m:
            car_level.append(key)
            pts = L.CRITICAL_EVENT_POINTS.get(key, 0)
            res.add(Finding("text", f"car_{key}", f"اعلام «{fa}» در متن آگهی", points=pts,
                            level="high" if pts >= 12 else "info", evidence=m.group(0),
                            detail="رخداد شدید؛ ارزش، ایمنی و نقدشوندگی خودرو را به‌طور جدی کاهش می‌دهد." if pts else ""))
    if "accident_car" in form_car_level:
        res.add(Finding("text", "form_accident_car", "بدنه در فرم فروشنده «تصادفی» اعلام شده",
                        points=L.CRITICAL_EVENT_POINTS["accident_car"], level="high",
                        question="تصادف از کدام سمت بوده و کدام قطعات تعویض یا صافکاری شده‌اند؟"))
    severity = body_severity(merged, unspecified, car_level)

    # ---------- دروغ‌سنجی: متن در برابر فرم ----------
    if ad.has_seller_eval:
        for t in text_items:
            match = next((f for f in form_items if _compatible(f, t)), None)
            if match is None:
                if form_body_clean:
                    res.add(Finding(
                        "text", "contradiction_body_clean",
                        f"تناقض: «{t.label} {t.dtype_fa}» در متن آمده، اما بدنه در فرم «سالم» اعلام شده",
                        detail="فروشنده در فرم ارزیابی دیوار بدنه را سالم زده ولی در توضیحات به خسارت اشاره کرده است.",
                        points=15, level="high", evidence=t.evidence,
                        question=f"در توضیحات نوشته‌اید «{t.evidence}»، ولی در فرم بدنه را سالم زده‌اید. کدام درست است؟",
                    ))
                elif form_declared and t.dtype != "scratch":
                    res.add(Finding(
                        "text", "contradiction_undeclared_part",
                        f"تناقض: «{t.label}» ({t.dtype_fa}) در متن آمده ولی در فرم ارزیابی اعلام نشده",
                        detail=f"فرم ارزیابی فقط {form_declared} ناحیه را اعلام کرده و این قطعه جزو آن‌ها نیست.",
                        points=10, level="medium", evidence=t.evidence,
                        question=f"در متن نوشته‌اید «{t.evidence}» ولی در فرم ارزیابی ثبت نکرده‌اید. "
                                 f"دقیقاً کدام قطعات رنگ یا تعویض دارند؟",
                    ))
            elif L.DAMAGE_RANK[t.dtype] > L.DAMAGE_RANK[match.dtype]:
                res.add(Finding(
                    "text", "contradiction_damage_type",
                    f"تناقض در نوع خسارت «{t.label}»: متن می‌گوید «{t.dtype_fa}»، فرم می‌گوید «{match.dtype_fa}»",
                    detail="تعویض قطعه معمولاً نشانه ضربه شدیدتر از یک رنگ ساده است و روی قیمت اثر بیشتری دارد.",
                    points=10, level="medium", evidence=t.evidence,
                    question=f"«{t.label}» رنگ شده یا تعویض شده؟ اگر تعویض است، به‌خاطر چه ضربه‌ای؟",
                ))
        if unspecified and form_declared and unspecified > form_declared:
            res.add(Finding(
                "text", "contradiction_count",
                f"تعداد لکه‌ها در متن ({unspecified}) بیشتر از فرم ({form_declared}) است",
                points=8, level="medium",
                question="دقیقاً چند قسمت از بدنه رنگ دارد و کدام قسمت‌ها؟",
            ))
        if unspecified and form_body_clean:
            res.add(Finding(
                "text", "contradiction_body_clean",
                "تناقض: متن به رنگ‌شدگی اشاره دارد اما فرم بدنه «سالم» است",
                points=12, level="high",
                question="بدنه رنگ دارد یا نه؟ کدام قسمت‌ها؟",
            ))

    # ادعای کلی «بی‌رنگ/سالم» در متن در حالی که خسارت داریم.
    # ادعای قطعه‌ای («سقف و کاپوت بی‌رنگ») تناقض نیست؛ فقط جمله‌های بدون نام قطعه.
    claim = None
    for clause in split_clauses(desc):
        c = normalize(clause)
        m = re.search(L.CLEAN_BODY_CLAIMS, c)
        if m and not _find_parts(c):
            claim = m
            break
    if claim and (form_items or text_items or unspecified or (form_declared and not form_body_clean)):
        res.add(Finding(
            "text", "contradiction_clean_claim",
            f"ادعای «{claim.group(0)}» با خسارت‌های اعلام‌شده نمی‌خواند",
            points=8, level="medium", evidence=claim.group(0),
            question=f"نوشته‌اید «{claim.group(0)}» ولی رنگ/تعویض هم اعلام شده؛ توضیح می‌دهید؟",
        ))

    if hedges:
        res.add(Finding(
            "text", "hedging_language",
            "استفاده از کلمات کوچک‌نمایی خسارت («جزئی»، «یه ذره»، «لکه‌گیری»...)",
            detail="این کلمات اغلب برای کم‌اهمیت جلوه‌دادن خسارت واقعی استفاده می‌شوند؛ حتماً با رنگ‌سنج بررسی شود.",
            points=3, level="low", evidence=" | ".join(hedges[:3]),
        ))

    # ---------- وضعیت مکانیکی: فرم در برابر متن ----------
    chassis = _component_status(ad, "شاسی")
    engine = _component_status(ad, "موتور")
    gearbox = _component_status(ad, "گیربکس")
    flags = {"chassis_issue": False, "engine_repair": False, "gearbox_repair": False}

    if chassis:
        v = normalize(chassis[1])
        if re.search(r"ضربه|آسیب|جوش|رنگ|تعمیر|تعویض", v):
            flags["chassis_issue"] = True
            res.add(Finding("text", "form_chassis_issue", f"شاسی در فرم فروشنده: «{chassis[1]}»",
                            detail="آسیب شاسی جدی‌ترین نوع خسارت است و ارزش و ایمنی خودرو را به‌شدت کم می‌کند.",
                            points=30, level="high",
                            question="شاسی جلو یا عقب؟ به چه علت و کجا تعمیر شده؟"))
    if engine:
        v = normalize(engine[1])
        if re.search(r"تعویض", v):
            flags["engine_repair"] = True
            res.add(Finding("text", "form_engine_replaced", f"موتور در فرم: «{engine[1]}»", points=12,
                            question="موتور تعویضی در سند ثبت شده؟"))
        elif re.search(r"تعمیر|نیاز", v):
            flags["engine_repair"] = True
            res.add(Finding("text", "form_engine_repair", f"موتور در فرم: «{engine[1]}»", points=8,
                            question="موتور چه تعمیری شده یا لازم دارد؟"))
    if gearbox:
        v = normalize(gearbox[1])
        if re.search(r"تعمیر|نیاز|تعویض", v):
            flags["gearbox_repair"] = True
            res.add(Finding("text", "form_gearbox_repair", f"گیربکس در فرم: «{gearbox[1]}»", points=6,
                            question="گیربکس چه ایرادی دارد/داشته؟"))

    chassis_clauses = [c for c in map(normalize, split_clauses(desc))
                       if "شاسی" in c and re.search(r"ضربه|آسیب|جوش|تعمیر|رنگ|کشیده|ترک", c)
                       and not re.search(r"سالم|پلمپ|بدون|نداره|ندارد|نخورده", c)]
    if chassis_clauses:
        flags["chassis_issue"] = True
        healthy = chassis and re.search(r"سالم", normalize(chassis[1]))
        res.add(Finding(
            "text", "contradiction_chassis" if healthy else "text_chassis_issue",
            "تناقض: متن به آسیب شاسی اشاره دارد ولی فرم شاسی را سالم اعلام کرده" if healthy
            else "اشاره به آسیب یا تعمیر شاسی در متن",
            points=25 if healthy else 20, level="high", evidence=chassis_clauses[0],
            question="وضعیت دقیق شاسی جلو و عقب چیست؟",
        ))

    presale = re.search(L.PRESALE, full)
    if presale:
        res.add(Finding(
            "text", "presale", "آگهی پیش‌فروش / حواله (خودرو هنوز تحویل نشده)",
            detail="قیمت با خودروهای موجود قابل مقایسه نیست. ریسک اصلی: تأخیر یا عدم تحویل و پرداخت پیش از دریافت خودرو.",
            points=8, level="medium", evidence=presale.group(0),
            question="قرارداد رسمی دارید؟ تاریخ تحویل قطعی و جریمه تأخیر چیست؟ پول به حساب چه کسی واریز می‌شود؟",
        ))

    for code, fa, pts, pattern, question in L.MECHANICAL:
        m = re.search(pattern, ndesc)
        if not m:
            continue
        if code.startswith("engine") or code in ("head_gasket", "oil_burning", "overheat"):
            flags["engine_repair"] = True
        if code == "gearbox_repair":
            flags["gearbox_repair"] = True
        extra = 0
        title = fa
        if code.startswith("engine") and engine and re.search(r"سالم", normalize(engine[1])):
            extra, title = 4, f"{fa} — در حالی که فرم موتور را «سالم» اعلام کرده"
        if code == "gearbox_repair" and gearbox and re.search(r"سالم", normalize(gearbox[1])):
            extra, title = 4, f"{fa} — در حالی که فرم گیربکس را «سالم» اعلام کرده"
        res.add(Finding("text", code, title, points=pts + extra, evidence=m.group(0), question=question))

    for code, fa, pts, pattern, question in L.DOCUMENT:
        m = re.search(pattern, full)
        if m:
            res.add(Finding("text", code, fa, points=pts, evidence=m.group(0), question=question))

    for code, fa, pts, pattern in L.PRESSURE:
        m = re.search(pattern, full)
        if m:
            res.add(Finding("text", code, fa, points=pts, evidence=m.group(0),
                            detail="الگوی رایج در آگهی‌های پرریسک و کلاهبرداری؛ قبل از بازدید پولی واریز نکنید."
                            if code == "pressure_deposit" else ""))

    for code, fa, pts, pattern in L.POSITIVE:
        m = re.search(pattern, full)
        if m:
            res.add(Finding("text", code, fa, points=pts, level="positive", evidence=m.group(0)))

    # ---------- دروغ‌سنجی: کارکرد و سال ----------
    km_text = _km_from_text(desc)
    if km_text and ad.km and abs(km_text - ad.km) / max(ad.km, 1) > 0.10 and abs(km_text - ad.km) > 5000:
        res.add(Finding(
            "text", "contradiction_km",
            f"کارکرد در متن ({km_text:,}) با فیلد کارکرد ({ad.km:,}) متفاوت است",
            points=10, level="medium",
            question="کارکرد واقعی خودرو چقدر است؟",
        ))
    year_text = _year_from_text(ad.title, desc)
    if year_text and ad.year_jalali and abs(year_text - ad.year_jalali) >= 1:
        res.add(Finding(
            "text", "contradiction_year",
            f"سال ساخت در متن/عنوان ({year_text}) با فیلد سال ({ad.year_jalali}) متفاوت است",
            points=8, level="medium",
            question="سال دقیق تولید طبق سند چیست؟",
        ))

    age = None
    km_per_year = None
    if ad.year_jalali:
        age = max(current_jalali_year() - ad.year_jalali, 0) + 0.5
        if ad.km is not None:
            km_per_year = ad.km / age
            if age >= 3 and km_per_year < 4000:
                res.add(Finding(
                    "text", "km_suspicious_low",
                    f"کارکرد خیلی کم برای سن خودرو (حدود {km_per_year:,.0f} کیلومتر در سال)",
                    detail="کارکرد غیرعادی پایین می‌تواند نشانه دستکاری کیلومترشمار باشد.",
                    points=10, level="medium",
                    question="سوابق سرویس یا برگه معاینه فنی با کارکرد ثبت‌شده دارید؟",
                ))
            elif km_per_year > 45000:
                res.add(Finding(
                    "text", "km_heavy_use",
                    f"کارکرد سنگین (حدود {km_per_year:,.0f} کیلومتر در سال)",
                    detail="احتمال استفاده در تاکسی اینترنتی یا سفرهای جاده‌ای زیاد؛ فرسودگی موتور، کلاچ و جلوبندی را بررسی کنید.",
                    points=5, level="low",
                ))

    ex = re.search(L.EXAGGERATION, ndesc)
    if ex and ((ad.km or 0) > 50000 or (age or 0) > 5):
        res.add(Finding("text", "exaggeration", f"ادعای اغراق‌آمیز «{ex.group(0)}» برای خودروی کارکرده",
                        points=3, level="low", evidence=ex.group(0)))

    if not ad.has_seller_eval:
        res.add(Finding("text", "no_seller_eval", "فروشنده فرم ارزیابی وضعیت (موتور/شاسی/بدنه) را پر نکرده",
                        points=6, level="low",
                        question="وضعیت موتور، شاسی و بدنه را دقیق اعلام می‌کنید؟"))

    res.data = {
        "form_damages": [vars(i) | {"label": i.label} for i in form_items],
        "text_damages": [vars(i) | {"label": i.label} for i in text_items],
        "merged_damages": [vars(i) | {"label": i.label} for i in merged],
        "form_declared_areas": form_declared,
        "form_body_clean": form_body_clean,
        "unspecified_paint_spots": unspecified,
        "car_level_damage": sorted(set(car_level)),
        "body_severity": severity,
        "flags": flags,
        "presale": bool(presale),
        "km_text": km_text,
        "year_text": year_text,
        "age_years": age,
        "km_per_year": round(km_per_year) if km_per_year else None,
        "seller_eval": ad.seller_eval,
        "desc_len": len(desc),
    }
    return res


def body_stage(text_result: StageResult) -> StageResult:
    """امتیاز ریسک شدت خسارت بدنه (جدا از تناقض‌ها)."""
    res = StageResult("body", "شدت خسارت بدنه (فرم + متن)")
    d = text_result.data
    sev = d["body_severity"]
    if d["merged_damages"] or d["unspecified_paint_spots"] or d["car_level_damage"]:
        parts = "، ".join(f"{m['label']} ({m['dtype_fa']})" for m in d["merged_damages"])
        if d["unspecified_paint_spots"]:
            parts += ("، " if parts else "") + f"{d['unspecified_paint_spots']} لکه بدون ذکر محل"
        res.add(Finding("body", "body_damage", f"خسارت بدنه با شدت {sev:g}",
                        detail=parts, points=round(sev * 5.5, 1)))
    elif d["form_body_clean"]:
        res.add(Finding("body", "body_clean", "بدنه طبق اعلام فروشنده سالم است (نیاز به تأیید با رنگ‌سنج)",
                        points=0, level="info"))
    return res
