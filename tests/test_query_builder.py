from __future__ import annotations

import unittest

import pandas as pd

from ods.query_builder import (
    AggregateRule,
    FilterRule,
    QueryBuilderError,
    QuerySpec,
    build_query,
    execute_query,
)


class QueryBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "customer segment": ["Enterprise", "Consumer", "Enterprise", None],
                "revenue": [1200.0, 400.0, 800.0, 100.0],
                "active": [True, False, True, False],
            }
        )

    def test_quotes_identifiers_and_parameterizes_values(self) -> None:
        query = build_query(
            self.frame,
            QuerySpec(
                selected_columns=("customer segment",),
                filters=(FilterRule("customer segment", "eq", "Enterprise"),),
            ),
        )
        self.assertIn('"customer segment" = ?', query.sql)
        self.assertNotIn("Enterprise", query.sql)
        self.assertEqual(query.parameters, ("Enterprise",))
        self.assertIn("'Enterprise'", query.display_sql)

    def test_filters_sorts_and_limits_detail_rows(self) -> None:
        result = execute_query(
            self.frame,
            QuerySpec(
                selected_columns=("customer segment", "revenue"),
                filters=(FilterRule("revenue", "gte", 500),),
                sort_by="revenue",
                sort_descending=True,
                limit=2,
            ),
        ).dataframe
        self.assertEqual(result["revenue"].tolist(), [1200.0, 800.0])

    def test_groups_and_averages_with_a_readable_alias(self) -> None:
        result = execute_query(
            self.frame,
            QuerySpec(
                group_by=("customer segment",),
                aggregates=(AggregateRule("mean", "revenue", "average_revenue"),),
                sort_by="average_revenue",
                sort_descending=True,
            ),
        ).dataframe
        enterprise = result.loc[
            result["customer segment"] == "Enterprise", "average_revenue"
        ].iloc[0]
        self.assertEqual(enterprise, 1000.0)

    def test_counts_rows_for_the_whole_dataset(self) -> None:
        result = execute_query(
            self.frame,
            QuerySpec(aggregates=(AggregateRule("count_rows", alias="records"),)),
        ).dataframe
        self.assertEqual(result.loc[0, "records"], 4)

    def test_supports_missing_value_filters(self) -> None:
        result = execute_query(
            self.frame,
            QuerySpec(filters=(FilterRule("customer segment", "is_null"),)),
        ).dataframe
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["revenue"], 100.0)

    def test_coerces_formatted_numeric_filter_values(self) -> None:
        result = execute_query(
            self.frame,
            QuerySpec(filters=(FilterRule("revenue", "gte", "1,000"),)),
        ).dataframe
        self.assertEqual(result["revenue"].tolist(), [1200.0])

    def test_sql_injection_payload_is_data_not_code(self) -> None:
        payload = "Enterprise'; DROP TABLE uploaded_data; --"
        result = execute_query(
            self.frame,
            QuerySpec(filters=(FilterRule("customer segment", "eq", payload),)),
        ).dataframe
        self.assertTrue(result.empty)
        query = build_query(
            self.frame,
            QuerySpec(filters=(FilterRule("customer segment", "eq", payload),)),
        )
        self.assertNotIn("DROP TABLE", query.sql)
        self.assertIn("DROP TABLE", query.parameters[0])

    def test_rejects_unknown_columns_and_unsafe_limits(self) -> None:
        with self.assertRaises(QueryBuilderError):
            build_query(self.frame, QuerySpec(selected_columns=("missing",)))
        with self.assertRaises(QueryBuilderError):
            build_query(self.frame, QuerySpec(limit=5001))


if __name__ == "__main__":
    unittest.main()
