from __future__ import annotations

import unittest

import pandas as pd

from ods.sql_coach import SqlCoachError, analyze_query


class SqlCoachTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4],
                "segment": ["Enterprise", "Consumer", "Enterprise", None],
                "revenue": [1200.0, 400.0, 800.0, 100.0],
            }
        )

    def test_explains_and_plans_a_read_only_summary(self) -> None:
        analysis = analyze_query(
            self.frame,
            """
            SELECT segment, SUM(revenue) AS total_revenue
            FROM uploaded_data
            GROUP BY segment
            ORDER BY total_revenue DESC
            LIMIT 10
            """,
        )
        self.assertEqual(analysis.referenced_tables, ("uploaded_data",))
        self.assertEqual(analysis.score, 100)
        clauses = {clause.clause for clause in analysis.clauses}
        self.assertTrue({"SELECT", "FROM", "GROUP BY", "ORDER BY", "LIMIT"} <= clauses)
        operators = {step.operator for step in analysis.plan_steps}
        self.assertIn("PANDAS_SCAN", operators)
        self.assertIn("HASH_GROUP_BY", operators)
        self.assertTrue(analysis.physical_plan)

    def test_blocks_data_changing_and_management_statements(self) -> None:
        for sql in (
            "DELETE FROM uploaded_data",
            "UPDATE uploaded_data SET revenue = 0",
            "DROP TABLE uploaded_data",
            "CREATE TABLE copy AS SELECT * FROM uploaded_data",
            "WITH changed AS (DELETE FROM uploaded_data RETURNING *) SELECT * FROM changed",
            (
                "WITH changed AS ("
                "INSERT INTO uploaded_data SELECT * FROM uploaded_data RETURNING *"
                ") SELECT * FROM changed"
            ),
        ):
            with self.subTest(sql=sql), self.assertRaises(SqlCoachError):
                analyze_query(self.frame, sql)

    def test_blocks_multiple_statements(self) -> None:
        with self.assertRaisesRegex(SqlCoachError, "exactly one"):
            analyze_query(
                self.frame,
                "SELECT * FROM uploaded_data; SELECT * FROM uploaded_data;",
            )

    def test_blocks_unknown_tables_and_external_file_readers(self) -> None:
        with self.assertRaisesRegex(SqlCoachError, "Unknown table"):
            analyze_query(self.frame, "SELECT * FROM private_table")
        with self.assertRaisesRegex(SqlCoachError, "External table functions"):
            analyze_query(self.frame, "SELECT * FROM read_csv_auto('/etc/passwd')")

    def test_expands_a_safe_star_and_flags_unbounded_detail(self) -> None:
        analysis = analyze_query(self.frame, "SELECT * FROM uploaded_data")
        rule_ids = {finding.rule_id for finding in analysis.findings}
        self.assertIn("explicit-columns", rule_ids)
        self.assertIn("unbounded-detail", rule_ids)
        self.assertIn('"customer_id"', analysis.suggested_sql)
        self.assertNotIn("SELECT\n  *", analysis.suggested_sql)

    def test_flags_leading_wildcard_and_function_filter(self) -> None:
        analysis = analyze_query(
            self.frame,
            "SELECT segment FROM uploaded_data WHERE LOWER(segment) LIKE '%enter%' LIMIT 20",
        )
        rule_ids = {finding.rule_id for finding in analysis.findings}
        self.assertIn("leading-wildcard", rule_ids)
        self.assertIn("function-on-filter-column", rule_ids)

    def test_flags_incorrect_null_comparison(self) -> None:
        analysis = analyze_query(
            self.frame,
            "SELECT segment FROM uploaded_data WHERE segment = NULL",
        )
        finding = next(item for item in analysis.findings if item.rule_id == "null-comparison")
        self.assertEqual(finding.severity, "high")
        self.assertIn("EMPTY_RESULT", {step.operator for step in analysis.plan_steps})

    def test_flags_not_in_subquery_null_risk(self) -> None:
        analysis = analyze_query(
            self.frame,
            """
            SELECT customer_id
            FROM uploaded_data
            WHERE customer_id NOT IN (
                SELECT customer_id FROM uploaded_data WHERE segment IS NULL
            )
            """,
        )
        self.assertIn(
            "not-in-subquery",
            {finding.rule_id for finding in analysis.findings},
        )

    def test_flags_cross_join_and_repeated_scan(self) -> None:
        analysis = analyze_query(
            self.frame,
            """
            SELECT a.customer_id, b.customer_id
            FROM uploaded_data AS a
            CROSS JOIN uploaded_data AS b
            LIMIT 10
            """,
        )
        rule_ids = {finding.rule_id for finding in analysis.findings}
        self.assertIn("cross-join", rule_ids)
        self.assertIn("repeated-upload-scan", rule_ids)

    def test_uses_uploaded_data_to_flag_nearly_unique_groups(self) -> None:
        frame = pd.DataFrame({"unique_key": range(30), "value": range(30)})
        analysis = analyze_query(
            frame,
            "SELECT unique_key, SUM(value) FROM uploaded_data GROUP BY unique_key",
        )
        self.assertIn(
            "high-cardinality-group-unique_key",
            {finding.rule_id for finding in analysis.findings},
        )

    def test_flags_unbounded_ordinal_sort(self) -> None:
        analysis = analyze_query(
            self.frame,
            "SELECT segment, revenue FROM uploaded_data ORDER BY 1",
        )
        rule_ids = {finding.rule_id for finding in analysis.findings}
        self.assertIn("unbounded-detail", rule_ids)
        self.assertIn("unbounded-sort", rule_ids)
        self.assertIn("ordinal-order", rule_ids)

    def test_reports_a_useful_syntax_error(self) -> None:
        with self.assertRaisesRegex(SqlCoachError, "syntax error"):
            analyze_query(self.frame, "SELECT ( FROM uploaded_data")


if __name__ == "__main__":
    unittest.main()
