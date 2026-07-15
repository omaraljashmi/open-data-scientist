from __future__ import annotations

from time import perf_counter
import unittest

import pandas as pd

from ods import (
    AggregateRule,
    FilterRule,
    QuerySpec,
    analyze_query,
    build_card_result,
    build_query,
    default_dashboard_config,
    execute_query,
    profile_dataset,
    suggest_cleaning_actions,
)


def synthetic_dataset(rows: int = 25_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(1, rows + 1),
            "segment": [
                ("Enterprise", "Consumer", "Small Business")[index % 3]
                for index in range(rows)
            ],
            "region": [
                ("North", "South", "East", "West")[index % 4]
                for index in range(rows)
            ],
            "monthly_revenue": [
                None if index % 101 == 0 else float(500 + (index % 2500))
                for index in range(rows)
            ],
            "satisfaction": [float(1 + (index % 10)) for index in range(rows)],
            "signup_date": pd.date_range("2022-01-01", periods=rows, freq="h"),
            "status": [" Active " if index % 5 else "Paused" for index in range(rows)],
        }
    )


class ReleasePerformanceTests(unittest.TestCase):
    def test_representative_workflow_stays_within_release_budget(self) -> None:
        dataframe = synthetic_dataset()
        started = perf_counter()

        profile = profile_dataset(dataframe)
        actions = suggest_cleaning_actions(dataframe)
        config = default_dashboard_config(dataframe)
        card_results = [build_card_result(dataframe, card) for card in config.cards]

        query_spec = QuerySpec(
            group_by=("segment",),
            aggregates=(AggregateRule("mean", "monthly_revenue", "average_revenue"),),
            filters=(FilterRule("status", "contains", "Active"),),
            sort_by="average_revenue",
            sort_descending=True,
            limit=100,
        )
        built = build_query(dataframe, query_spec)
        query_result = execute_query(dataframe, query_spec)
        analysis = analyze_query(
            dataframe,
            "SELECT segment, AVG(monthly_revenue) AS average_revenue "
            "FROM uploaded_data GROUP BY segment ORDER BY average_revenue DESC LIMIT 100",
        )
        elapsed = perf_counter() - started

        self.assertEqual(profile.rows, 25_000)
        self.assertTrue(actions)
        self.assertEqual(len(card_results), 4)
        self.assertIn("GROUP BY", built.display_sql)
        self.assertEqual(len(query_result.dataframe), 3)
        self.assertTrue(analysis.plan_steps)
        self.assertLess(
            elapsed,
            20.0,
            f"Representative 25,000-row workflow took {elapsed:.2f}s.",
        )


if __name__ == "__main__":
    unittest.main()
