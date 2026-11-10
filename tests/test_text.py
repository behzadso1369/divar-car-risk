"""تست‌های مرحله استخراج متن و دروغ‌سنجی. اجرا: python -m unittest discover tests"""
import json
import unittest
from pathlib import Path

from app.models import Ad
from app.parser import parse_post
from app import stage2_text
from app.text_utils import jalali_to_gregorian, normalize, parse_int

FIXTURES = Path(__file__).parent / "fixtures"


def make_ad(desc: str, seller_eval: dict | None = None, **kw) -> Ad:
    ad = Ad(token="test123", url="", description=desc, title=kw.pop("title", ""), **kw)
    ad.seller_eval = seller_eval or {}
    return ad


def codes(result):
    return [f.code for f in result.findings]


class TextUtilsTest(unittest.TestCase):
    def test_jalali(self):
        self.assertEqual(str(jalali_to_gregorian(1405, 7, 10)), "2026-10-02")
        self.assertEqual(str(jalali_to_gregorian(1403, 1, 1)), "2024-03-20")

    def test_parse_int(self):
        self.assertEqual(parse_int("‏۱,۴۹۰,۰۰۰,۰۰۰ تومان"), 1490000000)

    def test_normalize(self):
        self.assertEqual(normalize("رنگ‌شدگی در ۱ ناحیه"), "رنگ شدگی در 1 ناحیه")


class SampleAdTest(unittest.TestCase):
    """آگهی واقعی: متن می‌گوید درب و گلگیر راننده تعویض، فرم فقط رنگ درب را اعلام کرده."""

    def setUp(self):
        raw = json.loads((FIXTURES / "ga_i3H7H.json").read_text(encoding="utf-8"))
        self.ad = parse_post("ga_i3H7H", raw)

    def test_parse(self):
        self.assertEqual(self.ad.km, 96000)
        self.assertEqual(self.ad.year_jalali, 1397)
        self.assertEqual(self.ad.price, 1490000000)
        self.assertEqual(self.ad.brand_model_slug, "Peugeot 206 5")
        self.assertEqual(self.ad.seller_eval["بدنه"]["tags"], ["درب سمت راننده جلو"])

    def test_contradictions(self):
        r = stage2_text.run(self.ad)
        c = codes(r)
        self.assertIn("contradiction_damage_type", c)
        self.assertIn("contradiction_undeclared_part", c)
        parts = {d["part"] for d in r.data["text_damages"]}
        self.assertEqual(parts, {"door", "fender"})   # «قفل کاپوت» خسارت نیست
        self.assertNotIn("contradiction_km", c)


class FalsePositiveTest(unittest.TestCase):
    def test_chassis_negation(self):
        r = stage2_text.run(make_ad("شاسی‌ها پلمپ سالم بدون ترک و جوش\nیک گلگیر رنگ داره",
                                    {"وضعیت شاسی‌ها": {"value": "سالم و پلمپ", "tags": []}}))
        self.assertFalse(r.data["flags"]["chassis_issue"])

    def test_melk_is_not_paint(self):
        r = stage2_text.run(make_ad("معاوضه با ملک یا زمین ندارم",
                                    {"بدنه": {"value": "سالم و بی‌خط و خش", "tags": []}}))
        self.assertEqual(r.data["unspecified_paint_spots"], 0)
        self.assertNotIn("contradiction_body_clean", codes(r))

    def test_color_is_not_paint(self):
        r = stage2_text.run(make_ad("رنگ سفید\nرنگ مشکی صدفی"))
        self.assertEqual(r.data["unspecified_paint_spots"], 0)

    def test_fabric_seat_is_not_body_claim(self):
        r = stage2_text.run(make_ad("روکش صندلی فابریکی چرم\nکاپوت رنگ",
                                    {"بدنه": {"value": "رنگ‌شدگی در ۱ ناحیه", "tags": ["کاپوت"]}}))
        self.assertNotIn("contradiction_clean_claim", codes(r))

    def test_delivery_is_not_urgent(self):
        r = stage2_text.run(make_ad("صفر کیلومتر تحویل فوری"))
        self.assertNotIn("pressure_urgent", codes(r))

    def test_oil_change_is_not_body_damage(self):
        r = stage2_text.run(make_ad("روغن و لنت تازه تعویض شده\nقفل کاپوت"))
        self.assertEqual(r.data["text_damages"], [])


class RealWorldCasesTest(unittest.TestCase):
    """موارد واقعی که در تست روی آگهی‌های دیوار خطا داشتند."""

    def test_palm_sized_paint_and_part_claims(self):
        desc = ("سمت شاگرد درب جلو یه کف دست رنگ گلگیر جلو یه کف دست رنگ سقف کاپوت صندوق بی رنگ"
                "معاوضه با ۲۰۷ مدل ۱۴۰۲ به بالا")
        r = stage2_text.run(make_ad(desc, year_jalali=1396, km=100000))
        got = {(d["part"], d["dtype"]) for d in r.data["text_damages"]}
        self.assertEqual(got, {("door", "paint"), ("fender", "paint")})   # نه کف، نه سقف/کاپوت/صندوق
        self.assertNotIn("contradiction_year", codes(r))
        self.assertNotIn("contradiction_clean_claim", codes(r))
        self.assertIn("hedging_language", codes(r))

    def test_rear_chassis_row(self):
        ad = make_ad("", {"شاسی جلو": {"value": "سالم و پلمپ", "tags": []},
                          "شاسی عقب": {"value": "ضربه‌خورده", "tags": []}})
        r = stage2_text.run(ad)
        self.assertTrue(r.data["flags"]["chassis_issue"])

    def test_sun_fade_is_not_fire(self):
        r = stage2_text.run(make_ad("سقف آفتاب‌سوختگی داره"))
        self.assertNotIn("fire", r.data["car_level_damage"])

    def test_hood_in_form_tag(self):
        ad = make_ad("کاپوت تعویض", {"بدنه": {"value": "رنگ‌شدگی در ۱ ناحیه", "tags": ["درب موتور جلو (کاپوت)"]}})
        r = stage2_text.run(ad)
        self.assertEqual(r.data["form_damages"][0]["part"], "hood")
        self.assertIn("contradiction_damage_type", codes(r))

    def test_airbag(self):
        r = stage2_text.run(make_ad("ایربک زده"))
        self.assertIn("airbag", r.data["car_level_damage"])

    def test_negated_accident(self):
        r = stage2_text.run(make_ad("کاپوت ضربه نخورده\nشاسی بدون ضربه"))
        self.assertEqual(r.data["text_damages"], [])

    def test_damage_before_parts(self):
        r = stage2_text.run(make_ad("رنگ شدگی کاپوت و گلگیر"))
        self.assertEqual({d["part"] for d in r.data["text_damages"]}, {"hood", "fender"})

    def test_benzini_is_not_benz(self):
        from app.stage7_knowledge import find_entry
        ad = make_ad("", brand_model_fa="ایران خودرو ری را 1.7 لیتر توربو بنزینی")
        self.assertNotEqual(find_entry(ad)["id"], "german_luxury")


class DetectionTest(unittest.TestCase):
    def test_clean_form_but_text_damage(self):
        r = stage2_text.run(make_ad("کاپوت رنگ دارد",
                                    {"بدنه": {"value": "سالم و بی‌خط و خش", "tags": []}}))
        self.assertIn("contradiction_body_clean", codes(r))

    def test_pdr_is_low_severity(self):
        r = stage2_text.run(make_ad("درب عقب شاگرد صافکاری بی رنگ"))
        self.assertEqual(r.data["text_damages"][0]["dtype"], "pdr")

    def test_shared_damage(self):
        r = stage2_text.run(make_ad("درب و گلگیر راننده تعویض"))
        self.assertEqual({(d["part"], d["dtype"], d["side"]) for d in r.data["text_damages"]},
                         {("door", "replaced", "driver"), ("fender", "replaced", "driver")})

    def test_unspecified_spots(self):
        r = stage2_text.run(make_ad("دو لکه رنگ جزئی"))
        self.assertEqual(r.data["unspecified_paint_spots"], 2)
        self.assertIn("hedging_language", codes(r))

    def test_documents_and_pressure(self):
        r = stage2_text.run(make_ad("سند وکالتی\nفوری به علت نیاز مالی\nبیعانه واریز کنید"))
        c = codes(r)
        for code in ("doc_power_of_attorney", "pressure_urgent", "pressure_money", "pressure_deposit"):
            self.assertIn(code, c)

    def test_km_mismatch(self):
        r = stage2_text.run(make_ad("کارکرد 150 هزار", km=90000, year_jalali=1398))
        self.assertIn("contradiction_km", codes(r))

    def test_odometer_suspicion(self):
        r = stage2_text.run(make_ad("خانمی تمیز", km=12000, year_jalali=1392))
        self.assertIn("km_suspicious_low", codes(r))

    def test_presale(self):
        r = stage2_text.run(make_ad("تحویل 30 روزه\nپرداخت دو مرحله ای"))
        self.assertIn("presale", codes(r))

    def test_inspection_sheet_positive(self):
        r = stage2_text.run(make_ad("برگ کارشناسی هم گذاشتم"))
        self.assertIn("pos_inspection_ok", codes(r))


if __name__ == "__main__":
    unittest.main()
