from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

class AppTests(unittest.TestCase):
    def test_forecast_flow(self):
        self.assertTrue((ROOT / "app.py").exists(), "app must exist")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(m.label == "Giá dự đoán" for m in app.metric))
        if len(app.selectbox):
            app.selectbox[1].set_value("organic").run()
            self.assertFalse(any(m.label == "Giá dự đoán" for m in app.metric))
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(m.label == "Giá dự đoán" for m in app.metric))

