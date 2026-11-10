"""تست ساخت صفحه گزارش (بدون اینترنت): خطای قالب نباید به کاربر خطای ۵۰۰ بدهد."""
import json
import unittest
from pathlib import Path

from app.pipeline import analyze
from app.report import render_report
from app.stage1_needs import CustomerNeeds

FIXTURES = Path(__file__).parent / "fixtures"


class ReportRenderTest(unittest.TestCase):
    def setUp(self):
        raw = json.loads((FIXTURES / "ga_i3H7H.json").read_text(encoding="utf-8"))
        # offline_pricing: بدون جستجوی دیوار؛ save=False: دیتابیس دست نمی‌خورد
        self.report = analyze("ga_i3H7H", CustomerNeeds(budget_max=1_400_000_000, usage="city"),
                              raw=raw, offline_pricing=True, save=False)

    def test_render_web_and_standalone(self):
        for standalone in (True, False):
            html = render_report(self.report, standalone=standalone)
            self.assertIn("صداقت فروشنده", html)
            self.assertIn("تناقض‌های متن با فرم", html)   # آگهی نمونه دو تناقض دارد
            self.assertIn("گلگیر", html)

    def test_label_form_has_claims(self):
        html = render_report(self.report, standalone=False)
        self.assertIn('name="claim__insurance:', html)


if __name__ == "__main__":
    unittest.main()
