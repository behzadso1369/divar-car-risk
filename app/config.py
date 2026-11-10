"""تنظیمات مرکزی پروژه. همه ضرایب قابل تنظیم اینجا هستند."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.sqlite3"
MODELS_DIR = DATA_DIR / "models"
KNOWLEDGE_FILE = DATA_DIR / "knowledge" / "car_models.json"
REPORTS_DIR = BASE_DIR / "reports"

# --- دیوار ---
DIVAR_POST_API = "https://api.divar.ir/v8/posts-v2/web/{token}"
DIVAR_SEARCH_API = "https://api.divar.ir/v8/postlist/w/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20
REQUEST_DELAY_SEC = 1.0          # فاصله بین درخواست‌ها (احترام به سرور دیوار)
AD_CACHE_HOURS = 3               # کش آگهی
COMPARABLE_CACHE_HOURS = 12      # کش نتایج جستجوی آگهی‌های مشابه

# --- قیمت‌گذاری ---
COMPARABLE_MAX_PAGES = 4         # هر صفحه حدود ۲۵ آگهی
MIN_COMPARABLES = 8              # کمتر از این، بازه سال ±۱ می‌شود
COMPARABLE_MAX_AGE_DAYS = 30     # برای مدل سراسری قیمت (تورم!)
# فرض: آگهی‌های بازار به‌طور میانگین این‌قدر افت قیمت به‌خاطر خسارت دارند
MARKET_AVG_DAMAGE_DISCOUNT = 0.03
DAMAGE_DISCOUNT_PER_SEVERITY = 0.025   # هر واحد شدت خسارت بدنه
CHASSIS_DISCOUNT = 0.20
ENGINE_REPAIR_DISCOUNT = 0.06
GEARBOX_REPAIR_DISCOUNT = 0.04
MAX_DAMAGE_DISCOUNT = 0.45
SUSPICIOUS_LOW_PRICE = -0.18     # ۱۸٪ زیر قیمت منصفانه = مشکوک
VERY_SUSPICIOUS_LOW_PRICE = -0.28

# --- امتیازدهی ---
TIER_LOW_MAX = 25                # کمتر از این: ریسک پایین
TIER_MEDIUM_MAX = 50             # کمتر از این: ریسک متوسط، بیشتر: ریسک بالا
STAGE_CAPS = {                   # سقف امتیاز ریسک هر مرحله
    "body": 30,
    "text": 40,
    "pricing": 25,
    "behavior": 25,
    "images": 10,
    "knowledge": 10,
    "honesty": 20,
}
POSITIVE_FLOOR = -10             # حداکثر اثر نشانه‌های مثبت
ML_BLEND_WEIGHT = 0.4            # وزن مدل یادگرفته‌شده وقتی آموزش داده شده باشد
ML_MIN_LABELS = 30               # حداقل نتیجه کارشناسی برای آموزش مدل ریسک
PRICE_MODEL_MIN_ROWS = 200       # حداقل داده برای مدل سراسری قیمت

# --- عکس‌ها ---
IMAGE_MAX_DOWNLOAD = 12
