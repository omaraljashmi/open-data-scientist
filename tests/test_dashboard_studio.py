from __future__ import annotations

from dataclasses import replace
import json
import unittest

import pandas as pd

from ods.dashboard_studio import (
    DashboardCard,
    DashboardConfig,
    DashboardFilter,
    DashboardStudioError,
    apply_dashboard_filters,
    build_card_result,
    build_dashboard_html,
    categorical_filter_options,
    dashboard_config_from_json,
    dashboard_config_to_json,
    default_dashboard_config,
    default_filter_for_column,
    move_dashboard_card,
    remove_dashboard_card,
    replace_dashboard_card,
    validate_dashboard_config,
)


class DashboardStudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
                "date": [
                    "2026-01-02",
                    "2026-01-08",
                    "2026-01-19",
                    "2026-02-01",
                    "2026-02-07",
                    "2026-02-14",
                    "2026-02-20",
                    "invalid",
                ],
                "segment": ["A", "B", "A", "B", "A", None, "B", "A"],
                "revenue": [100.0, 200.0, 300.0, None, 500.0, 600.0, 700.0, 800.0],
                "units": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

    def test_builds_a_useful_four_card_default(self) -> None:
        config = default_dashboard_config(self.frame)
        self.assertEqual([card.kind for card in config.cards], ["kpi", "kpi", "pie", "line"])
        self.assertEqual(config.cards[0].metric, "row_count")
        validate_dashboard_config(self.frame, config)

    def test_pie_card_shares_are_exact(self) -> None:
        card = DashboardCard("pie", "pie", "Share by segment", x="segment")
        result = build_card_result(self.frame, card)
        audit = result.audit_table
        self.assertEqual(list(audit.columns), ["category", "value", "share_percent"])
        shares = dict(zip(audit["category"], audit["share_percent"]))
        self.assertEqual(shares, {"A": 50.0, "B": 37.5, "(missing)": 12.5})
        self.assertAlmostEqual(float(audit["share_percent"].sum()), 100.0)
        self.assertEqual(result.figure.data[0].type, "pie")

    def test_pie_card_lumps_small_groups_into_other(self) -> None:
        frame = pd.DataFrame(
            {"city": ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"] * 1}
        )
        card = DashboardCard("pie", "pie", "Cities", x="city", top_n=3)
        audit = build_card_result(frame, card).audit_table
        self.assertEqual(list(audit["category"]), ["a", "b", "c", "(other)"])
        self.assertEqual(int(audit.loc[audit["category"] == "(other)", "value"].iloc[0]), 3)
        self.assertAlmostEqual(float(audit["share_percent"].sum()), 100.0, places=1)

    def test_pie_card_rejects_non_additive_aggregations(self) -> None:
        card = DashboardCard(
            "pie", "pie", "Average by segment", x="segment", y="revenue", aggregation="mean"
        )
        with self.assertRaises(DashboardStudioError):
            build_card_result(self.frame, card)

    def test_calculates_all_kpi_metrics_exactly(self) -> None:
        cases = [
            (DashboardCard("rows", "kpi", "Rows"), 8),
            (DashboardCard("sum", "kpi", "Total", metric="sum", column="revenue"), 3200.0),
            (DashboardCard("mean", "kpi", "Average", metric="mean", column="revenue"), 3200 / 7),
            (DashboardCard("median", "kpi", "Median", metric="median", column="revenue"), 500.0),
            (
                DashboardCard(
                    "distinct",
                    "kpi",
                    "Segments",
                    metric="distinct_count",
                    column="segment",
                ),
                2,
            ),
        ]
        for card, expected in cases:
            with self.subTest(metric=card.metric):
                result = build_card_result(self.frame, card)
                self.assertAlmostEqual(float(result.value), float(expected))
                self.assertEqual(len(result.audit_table), 1)

    def test_applies_value_numeric_and_date_filters_with_and_logic(self) -> None:
        filters = (
            DashboardFilter("f1", "segment", "values", values=("A",)),
            DashboardFilter("f2", "revenue", "range", minimum=250, maximum=800),
            DashboardFilter(
                "f3",
                "date",
                "date_range",
                start="2026-01-01",
                end="2026-02-10",
            ),
        )
        filtered = apply_dashboard_filters(self.frame, filters)
        self.assertEqual(filtered["order_id"].tolist(), [103, 105])
        self.assertEqual(len(self.frame), 8)

    def test_value_filter_can_select_missing_and_lists_frequency_order(self) -> None:
        options = categorical_filter_options(self.frame, "segment")
        self.assertEqual(options, ("A", "B", "(missing)"))
        filtered = apply_dashboard_filters(
            self.frame,
            (DashboardFilter("f1", "segment", "values", values=("(missing)",)),),
        )
        self.assertEqual(filtered["order_id"].tolist(), [106])

    def test_default_range_filters_use_available_boundaries(self) -> None:
        filters = tuple(
            default_filter_for_column(self.frame, column, f"filter-{index}")
            for index, column in enumerate(("segment", "revenue", "date"), start=1)
        )
        filtered = apply_dashboard_filters(self.frame, filters)
        # Range controls intentionally exclude missing/invalid values; value filters with no
        # selection preserve all categories. This is explicit in the saved configuration.
        self.assertEqual(filtered["order_id"].tolist(), [101, 102, 103, 105, 106, 107])

    def test_bar_chart_uses_exact_grouped_aggregation(self) -> None:
        card = DashboardCard(
            "bar",
            "bar",
            "Revenue by segment",
            x="segment",
            y="revenue",
            aggregation="sum",
        )
        result = build_card_result(self.frame, card)
        values = dict(zip(result.audit_table["category"], result.audit_table["value"], strict=True))
        self.assertEqual(values, {"A": 1700.0, "B": 900.0, "(missing)": 600.0})
        self.assertEqual(len(result.figure.data), 1)

    def test_line_chart_groups_dates_and_excludes_invalid_dates(self) -> None:
        card = DashboardCard(
            "line",
            "line",
            "Monthly revenue",
            x="date",
            y="revenue",
            aggregation="sum",
            date_grain="month",
        )
        result = build_card_result(self.frame, card)
        self.assertEqual(result.audit_table["period"].dt.month.tolist(), [1, 2])
        self.assertEqual(result.audit_table["value"].tolist(), [600.0, 1800.0])

    def test_numeric_year_line_chart_uses_calendar_years(self) -> None:
        frame = pd.DataFrame({"report_year": [2024, 2025, 2025], "sales": [10, 20, 30]})
        card = DashboardCard(
            "line",
            "line",
            "Sales by year",
            x="report_year",
            y="sales",
            aggregation="sum",
            date_grain="year",
        )
        result = build_card_result(frame, card)
        self.assertEqual(result.audit_table["period"].dt.year.tolist(), [2024, 2025])
        self.assertEqual(result.audit_table["value"].tolist(), [10, 50])

    def test_scatter_sampling_is_bounded_and_deterministic(self) -> None:
        frame = pd.DataFrame({"x": range(6000), "y": range(6000, 12000)})
        card = DashboardCard("scatter", "scatter", "Relationship", x="x", y="y")
        first = build_card_result(frame, card).audit_table
        second = build_card_result(frame, card).audit_table
        self.assertEqual(len(first), 5000)
        pd.testing.assert_frame_equal(first, second)

    def test_distribution_bins_reconcile_to_valid_numeric_rows(self) -> None:
        card = DashboardCard(
            "distribution",
            "distribution",
            "Revenue distribution",
            column="revenue",
            bins=6,
        )
        result = build_card_result(self.frame, card)
        self.assertEqual(int(result.audit_table["count"].sum()), 7)
        self.assertLessEqual(len(result.audit_table), 6)

    def test_configuration_round_trip_preserves_every_control(self) -> None:
        config = replace(
            default_dashboard_config(self.frame),
            name="Executive view",
            filters=(
                DashboardFilter("f1", "segment", "values", values=("A", "B")),
            ),
        )
        loaded = dashboard_config_from_json(dashboard_config_to_json(config), self.frame)
        self.assertEqual(loaded, config)

    def test_rejects_oversized_unknown_or_badly_typed_configuration(self) -> None:
        with self.assertRaises(DashboardStudioError):
            dashboard_config_from_json("x" * 100_001, self.frame)
        payload = json.loads(dashboard_config_to_json(default_dashboard_config(self.frame)))
        payload["unexpected"] = True
        with self.assertRaises(DashboardStudioError):
            dashboard_config_from_json(json.dumps(payload), self.frame)
        payload.pop("unexpected")
        payload["cards"][0]["top_n"] = "twelve"
        with self.assertRaises(DashboardStudioError):
            dashboard_config_from_json(json.dumps(payload), self.frame)

    def test_rejects_missing_columns_and_duplicate_ids(self) -> None:
        config = DashboardConfig(
            "Invalid",
            (
                DashboardCard("same", "kpi", "Rows"),
                DashboardCard("same", "distribution", "Bad", column="unknown"),
            ),
        )
        with self.assertRaises(DashboardStudioError):
            validate_dashboard_config(self.frame, config)

    def test_card_edit_move_and_remove_helpers_preserve_order(self) -> None:
        config = default_dashboard_config(self.frame)
        renamed = replace(config.cards[0], title="All records")
        config = replace_dashboard_card(config, renamed)
        self.assertEqual(config.cards[0].title, "All records")
        config = move_dashboard_card(config, renamed.card_id, 1)
        self.assertEqual(config.cards[1].card_id, renamed.card_id)
        config = remove_dashboard_card(config, renamed.card_id)
        self.assertNotIn(renamed.card_id, {card.card_id for card in config.cards})

    def test_empty_filtered_dataset_produces_truthful_results(self) -> None:
        filtered = apply_dashboard_filters(
            self.frame,
            (DashboardFilter("f1", "segment", "values", values=("unavailable",)),),
        )
        kpi = build_card_result(filtered, DashboardCard("rows", "kpi", "Rows"))
        chart = build_card_result(
            filtered,
            DashboardCard("bar", "bar", "Rows by segment", x="segment"),
        )
        self.assertEqual(kpi.value, 0)
        self.assertTrue(chart.audit_table.empty)

    def test_standalone_html_is_inline_responsive_and_escapes_titles(self) -> None:
        config = replace(default_dashboard_config(self.frame), name="<script>alert(1)</script>")
        html = build_dashboard_html(self.frame, config)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("plotly.js", html.lower())
        self.assertNotIn("<script src=", html.lower())
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("Calculation details", html)
        self.assertIn("@media", html)

    def test_calculations_do_not_mutate_source_dataframe(self) -> None:
        original = self.frame.copy(deep=True)
        config = default_dashboard_config(self.frame)
        filtered = apply_dashboard_filters(self.frame, config.filters)
        for card in config.cards:
            build_card_result(filtered, card)
        pd.testing.assert_frame_equal(self.frame, original)


if __name__ == "__main__":
    unittest.main()
