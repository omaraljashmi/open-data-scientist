from __future__ import annotations

from base64 import b64decode
from pathlib import Path
import unittest

from ods import __release__, __version__


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_version_and_lockfile_match_the_release_candidate(self) -> None:
        self.assertEqual(__version__, "1.0.0rc1")
        self.assertEqual(__release__, "v1.0.0-rc.1")

        lock = (PROJECT_ROOT / "requirements.lock").read_text(encoding="utf-8")
        for dependency in (
            "pandas==",
            "streamlit==",
            "plotly==",
            "openpyxl==",
            "xlrd==",
            "duckdb==",
            "sqlglot==",
        ):
            self.assertIn(dependency, lock)

    def test_stable_promotion_assets_are_present_and_privacy_safe(self) -> None:
        required = (
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/ISSUE_TEMPLATE/config.yml",
            ".github/PULL_REQUEST_TEMPLATE.md",
            "docs/VALIDATION.md",
            "docs/STABLE_RELEASE_CHECKLIST.md",
            "docs/releases/v1.0.0.md",
            "scripts/validation_matrix.py",
            "scripts/fixtures/legacy_customers.xls.b64",
        )
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).is_file())

        bug_form = (
            PROJECT_ROOT / ".github/ISSUE_TEMPLATE/bug_report.yml"
        ).read_text(encoding="utf-8")
        feature_form = (
            PROJECT_ROOT / ".github/ISSUE_TEMPLATE/feature_request.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("Do not attach confidential", bug_form)
        self.assertIn("contains no confidential data", feature_form)

        legacy_bytes = b64decode(
            (PROJECT_ROOT / "scripts/fixtures/legacy_customers.xls.b64").read_text(
                encoding="ascii"
            )
        )
        self.assertTrue(legacy_bytes.startswith(bytes.fromhex("D0CF11E0A1B11AE1")))

    def test_validation_matrix_is_required_by_ci_and_contributors(self) -> None:
        command = "python -m scripts.validation_matrix"
        for relative_path in (
            ".github/workflows/ci.yml",
            "README.md",
            "CONTRIBUTING.md",
        ):
            with self.subTest(path=relative_path):
                content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(command, content)


if __name__ == "__main__":
    unittest.main()
