from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def test_landing_page_and_one_click_sample_walkthrough(self) -> None:
        app = AppTest.from_file(
            PROJECT_ROOT / "app.py",
            default_timeout=45,
        ).run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertEqual(
            app.button(key="try-sample-dataset").label,
            "Try sample dataset",
        )

        app.button(key="try-sample-dataset").click().run()
        self.assertEqual([item.value for item in app.exception], [])
        self.assertIn(
            "Loaded sample_customers.csv successfully.",
            [item.value for item in app.success],
        )
        self.assertEqual(
            [item.label for item in app.metric[:6]],
            [
                "Rows",
                "Columns",
                "Memory",
                "Missing cells",
                "Duplicates",
                "Quality score",
            ],
        )
        self.assertTrue(
            {"Data", "Clean", "Dashboard", "Visual SQL", "SQL Coach"}.issubset(
                {item.label for item in app.tabs}
            )
        )


if __name__ == "__main__":
    unittest.main()
