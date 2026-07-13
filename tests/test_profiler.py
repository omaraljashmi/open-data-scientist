from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

from ods.dashboard import build_chart_data, infer_column_roles, suggest_dashboard
from ods.loader import DatasetLoadError, load_dataset
from ods.profiler import profile_dataset
from ods.reporting import build_markdown_report


class DatasetLoaderTests(unittest.TestCase):
    def test_loads_csv_with_inferred_delimiter(self) -> None:
        frame = load_dataset("people.csv", b"name;age\nOmar;21\nAseel;22\n")
        self.assertEqual(frame.shape, (2, 2))
        self.assertEqual(frame.columns.tolist(), ["name", "age"])

    def test_loads_excel(self) -> None:
        source = pd.DataFrame({"value": [1, 2, 3]})
        stream = BytesIO()
        source.to_excel(stream, index=False)
        loaded = load_dataset("sample.xlsx", stream.getvalue())
        pd.testing.assert_frame_equal(source, loaded)

    def test_rejects_unsupported_type(self) -> None:
        with self.assertRaises(DatasetLoadError):
            load_dataset("notes.txt", b"hello")


class DatasetProfilerTests(unittest.TestCase):
    def test_profiles_missing_values_duplicates_and_constants(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2, 2, 4],
                "segment": ["A", "A", "A", "A"],
                "score": [10.0, None, None, 40.0],
            }
        )
        profile = profile_dataset(frame)
        self.assertEqual(profile.rows, 4)
        self.assertEqual(profile.columns, 3)
        self.assertEqual(profile.missing_cells, 2)
        self.assertLess(profile.health_score, 100)
        titles = {issue.title for issue in profile.issues}
        self.assertIn("High missingness", titles)
        self.assertIn("Constant or empty column", titles)

    def test_report_contains_key_results(self) -> None:
        profile = profile_dataset(pd.DataFrame({"value": [1, 2, None]}))
        report = build_markdown_report("sample.csv", profile)
        self.assertIn("Data Quality Report: sample.csv", report)
        self.assertIn("Quality score", report)
        self.assertIn("value", report)


class DashboardRecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "customer_id": [1001, 1002, 1003, 1004, 1005, 1005],
                "segment": ["Enterprise", "Small Business", "Enterprise", "Consumer", "Small Business", "Small Business"],
                "monthly_spend": [4200.0, 850.0, 5100.0, None, 920.0, 920.0],
                "satisfaction_score": [9.0, 7.0, 8.0, 6.0, None, None],
                "last_contact": ["2026-06-02", "2026-06-11", "2026-06-09", "2026-05-28", "2026-06-13", "2026-06-13"],
            }
        )

    def test_infers_semantic_roles_and_excludes_identifier(self) -> None:
        roles = infer_column_roles(self.frame)
        self.assertEqual(roles.identifiers, ("customer_id",))
        self.assertEqual(roles.categorical, ("segment",))
        self.assertEqual(roles.numeric, ("monthly_spend", "satisfaction_score"))
        self.assertEqual(roles.datetime, ("last_contact",))

    def test_recommends_a_small_standard_dashboard(self) -> None:
        suggestions = suggest_dashboard(self.frame)
        self.assertEqual(len(suggestions), 4)
        self.assertEqual(
            [suggestion.kind for suggestion in suggestions],
            ["missingness", "category_count", "histogram", "time_series"],
        )

    def test_builds_category_counts(self) -> None:
        suggestion = suggest_dashboard(self.frame)[1]
        chart = build_chart_data(self.frame, suggestion)
        counts = dict(zip(chart["segment"], chart["count"], strict=True))
        self.assertEqual(counts["Small Business"], 3)
        self.assertEqual(counts["Enterprise"], 2)

    def test_empty_dataset_has_no_suggestions(self) -> None:
        self.assertEqual(suggest_dashboard(pd.DataFrame()), ())


if __name__ == "__main__":
    unittest.main()
