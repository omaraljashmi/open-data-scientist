from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
