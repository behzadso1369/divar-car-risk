"""ابزارهای متن فارسی: یکسان‌سازی حروف و ارقام، تبدیل تاریخ شمسی."""
import re
from datetime import date

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHARS = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "آ": "آ",
    "‌": " ",  # نیم‌فاصله
    "‏": "", "‎": "", "‪": "", "‫": "", "‬": "",
    "ـ": "",
})
_DIACRITICS = re.compile(r"[ً-ٰٟ]")
_SPACES = re.compile(r"[ \t ]+")

PERSIAN_NUMBER_WORDS = {
    "یک": 1, "یه": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "شیش": 6,
    "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
}

JALALI_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6,
    "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}


def normalize(text: str | None) -> str:
    """یکسان‌سازی متن برای جستجو با الگو (نیم‌فاصله → فاصله، ارقام → لاتین)."""
    if not text:
        return ""
    text = text.translate(_DIGITS).translate(_CHARS)
    text = _DIACRITICS.sub("", text)
    text = _SPACES.sub(" ", text)
    return text.strip().lower()


def parse_int(text: str | None) -> int | None:
    """«‏۱,۴۹۰,۰۰۰,۰۰۰ تومان» → 1490000000"""
    if not text:
        return None
    digits = re.sub(r"\D", "", normalize(text))
    return int(digits) if digits else None


def word_or_digit_to_int(token: str) -> int | None:
    token = normalize(token)
    if token.isdigit():
        return int(token)
    return PERSIAN_NUMBER_WORDS.get(token)


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> date:
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd
    days += (jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 12 and gd > month_days[gm]:
        gd -= month_days[gm]
        gm += 1
    return date(gy, gm + 1, gd)


def current_jalali_year(today: date | None = None) -> int:
    today = today or date.today()
    # نوروز تقریباً ۲۰ یا ۲۱ مارس است؛ دقت روزانه برای محاسبه سن خودرو لازم نیست
    return today.year - 621 if (today.month, today.day) >= (3, 21) else today.year - 622


def parse_jalali_datetime_text(text: str) -> date | None:
    """«۱۰ مهر ۱۴۰۵، ۱۲:۵۴» → date(2026, 10, 2)"""
    text = normalize(text)
    m = re.search(r"(\d{1,2}) (" + "|".join(JALALI_MONTHS) + r") (\d{4})", text)
    if not m:
        return None
    try:
        return jalali_to_gregorian(int(m.group(3)), JALALI_MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None


def split_clauses(text: str) -> list[str]:
    """تقسیم توضیحات به جمله‌های کوتاه (هر خط / ویرگول / نقطه / خط تیره)."""
    parts = re.split(r"[\n\r]+|[،,.؛;!؟?]+| - |\*|•", text)
    return [p.strip() for p in parts if p and p.strip()]


def fmt_toman(value: int | float | None) -> str:
    if value is None:
        return "—"
    value = int(round(value))
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} میلیارد تومان".replace(".00 ", " ")
    if value >= 1_000_000:
        return f"{value / 1_000_000:,.0f} میلیون تومان"
    return f"{value:,} تومان"
