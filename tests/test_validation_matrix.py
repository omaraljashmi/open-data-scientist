from __future__ import annotations

import unittest

from scripts.validation_matrix import run_validation_matrix


class StableValidationMatrixTests(unittest.TestCase):
    def test_supported_and_rejected_files_complete_the_release_matrix(self) -> None:
        evidence = run_validation_matrix()

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(
            evidence["summary"],
            {
                "valid_files": 7,
                "rejected_files": 9,
                "formats": ["csv", "xlsx", "xls"],
            },
        )

        valid = {item["name"]: item for item in evidence["valid_cases"]}
        self.assertEqual(
            set(valid),
            {
                "standard-csv",
                "utf8-bom-semicolon-csv",
                "quoted-multiline-csv",
                "single-column-csv",
                "sparse-pipe-csv",
                "mixed-type-xlsx",
                "legacy-xls",
            },
        )
        self.assertEqual(valid["single-column-csv"]["columns"], 1)
        self.assertEqual(valid["legacy-xls"]["input_rows"], 3)
        for item in valid.values():
            with self.subTest(case=item["name"]):
                self.assertGreaterEqual(item["dashboard_cards"], 1)
                self.assertGreaterEqual(item["query_rows"], 1)
                self.assertGreater(item["sql_score"], 0)

        rejected = {
            item["name"]: item["message"]
            for item in evidence["rejection_cases"]
        }
        self.assertIn("non-empty header", rejected["blank-csv-header"])
        self.assertIn("must be unique", rejected["duplicate-csv-header"])


if __name__ == "__main__":
    unittest.main()
