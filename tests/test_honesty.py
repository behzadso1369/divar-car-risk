"""تست‌های صداقت فروشنده."""
import json
import unittest
from pathlib import Path

from app import honesty, stage2_text
from app.models import Ad, StageResult
from app.parser import parse_post

FIXTURES = Path(__file__).parent / "fixtures"


def run(desc, seller_eval=None, diff=None, **kw):
    ad = Ad(token="t", url="", description=desc, title=kw.pop("title", ""), **kw)
    ad.seller_eval = seller_eval or {}
    text = stage2_text.run(ad)
    pricing = StageResult("pricing", "")
    pricing.data = {"available": diff is not None, "diff_pct": diff}
    return honesty.run(ad, text, pricing).data


def status(data, code):
    return next(c["status"] for c in data["claims"] if c["code"] == code)


CLEAN_FORM = {"بدنه": {"value": "سالم و بی‌خط و خش", "tags": []},
              "موتور": {"value": "سالم", "tags": []},
              "وضعیت شاسی‌ها": {"value": "سالم و پلمپ", "tags": []}}


class HonestyTest(unittest.TestCase):
    def test_over_seller(self):
        """فقط تعریف کلی، هیچ جزئیات، خودروی قدیمی پرکارکرد."""
        d = run("در حد صفر\nبی نظیر\nتمیز تمیز\nخانمی\nبدون هیچ ایرادی\nفنی عالی",
                year_jalali=1390, km=280000, image_count=2)
        self.assertEqual(status(d, "like_new"), "implausible")
        self.assertEqual(status(d, "no_issues"), "implausible")
        self.assertEqual(status(d, "female_owned"), "unverifiable")
        self.assertLess(d["scores"]["specificity"], 40)
        self.assertLess(d["score"], 60)

    def test_honest_detailed_seller(self):
        """ایراد را خودش گفته، جزئیات مشخص، اجازه کارشناسی."""
        d = run("گلگیر جلو راننده رنگ دارد\nتسمه تایم ۱۴۰۴ عوض شده فاکتور دارم\nکارکرد ۱۲۰ هزار\n"
                "لاستیک ها ۸۰ درصد\nکارشناسی آزاد",
                {"بدنه": {"value": "رنگ‌شدگی در ۱ ناحیه", "tags": ["گلگیر سمت راننده جلو"]}},
                year_jalali=1396, km=120000, image_count=10)
        self.assertGreaterEqual(d["score"], 75)
        self.assertGreater(d["scores"]["transparency"], 80)

    def test_claim_contradicts_form(self):
        d = run("بدون رنگ", {"بدنه": {"value": "رنگ‌شدگی در ۲ ناحیه", "tags": ["کاپوت", "سقف"]}})
        self.assertEqual(status(d, "body_no_paint"), "contradicted")
        self.assertLessEqual(d["score"], 74)

    def test_perfect_but_cheap(self):
        d = run("بدون رنگ\nموتور سالم", CLEAN_FORM, diff=-0.25, year_jalali=1400, km=50000)
        self.assertTrue(d["cheap_perfect"])
        self.assertEqual(status(d, "body_no_paint"), "implausible")

    def test_bait_price(self):
        d = run("صفر کیلومتر", diff=-0.45)
        self.assertEqual(status(d, "price_itself"), "implausible")

    def test_refuse_inspection(self):
        d = run("کارشناسی نمیدم\nسالم", CLEAN_FORM)
        self.assertLess(d["scores"]["transparency"], 30)

    def test_part_specific_claim(self):
        d = run("سقف و کاپوت بی رنگ\nگلگیر عقب رنگ")
        self.assertIn("parts_no_paint", [c["code"] for c in d["claims"]])
        self.assertEqual(status(d, "parts_no_paint"), "consistent")

    def test_insurance_mismatch(self):
        d = run("بیمه ۱۰ ماه", insurance_months=2)
        self.assertEqual(status(d, "insurance"), "contradicted")

    def test_dealer_first_owner(self):
        d = run("دست اول", business_type="premium-panel")
        self.assertEqual(status(d, "first_owner"), "implausible")

    def test_sample_ad(self):
        raw = json.loads((FIXTURES / "ga_i3H7H.json").read_text(encoding="utf-8"))
        ad = parse_post("ga_i3H7H", raw)
        text = stage2_text.run(ad)
        pricing = StageResult("pricing", "")
        pricing.data = {"available": False}
        d = honesty.run(ad, text, pricing).data
        self.assertEqual(status(d, "insurance"), "consistent")
        self.assertIn("service_done", [c["code"] for c in d["claims"]])
        self.assertIn("tech_inspection", [c["code"] for c in d["claims"]])
        self.assertLessEqual(d["score"], 74)          # تناقض دارد → «قابل اعتماد» نیست
        self.assertGreater(d["scores"]["transparency"], 70)   # خودش تعویض‌ها را گفته


if __name__ == "__main__":
    unittest.main()
