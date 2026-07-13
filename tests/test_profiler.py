from __future__ import annotations

import unittest
from io import BytesIO

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()

