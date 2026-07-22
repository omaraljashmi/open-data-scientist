from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from ods import load_dataset, profile_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "examples" / "sample_customers.csv"


class AppSmokeTests(unittest.TestCase):
    def test_landing_page_and_one_click_sample_load(self) -> None:
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
        success_messages = [item.value for item in app.success]
        self.assertTrue(
            any("sample_customers.csv" in message for message in success_messages),
            success_messages,
        )

        # The landing page must publish the shared session-state contract
        # that every sidebar page reads.
        for key in ("original_df", "current_df", "filename", "dataset_key"):
            self.assertIn(key, app.session_state)
        dataset_key = app.session_state["dataset_key"]
        self.assertIn(f"profile-{dataset_key}", app.session_state)

    def test_profile_page_renders_from_session_state(self) -> None:
        sample_bytes = SAMPLE_PATH.read_bytes()
        dataframe = load_dataset(SAMPLE_PATH.name, sample_bytes)
        profile = profile_dataset(dataframe)

        page = AppTest.from_file(
            PROJECT_ROOT / "pages" / "1_Profile.py",
            default_timeout=45,
        )
        page.session_state["current_df"] = dataframe
        page.session_state["filename"] = SAMPLE_PATH.name
        page.session_state["dataset_key"] = "test-key"
        page.session_state["profile-test-key"] = profile
        page.run()

        self.assertEqual([item.value for item in page.exception], [])
        self.assertEqual(
            [item.label for item in page.metric[:6]],
            [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
                "Memory",
                "Quality score",
            ],
        )


if __name__ == "__main__":
    unittest.main()
