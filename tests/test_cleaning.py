from __future__ import annotations

import json
import unittest

import pandas as pd

from ods.cleaning import (
    CleaningError,
    apply_cleaning_actions,
    build_cleaning_recipe,
    replay_cleaning_batches,
    suggest_cleaning_actions,
)
from ods.profiler import profile_dataset


class CleaningRecommendationTests(unittest.TestCase):
    def test_detects_duplicates_whitespace_blanks_and_category_variants(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": [" Enterprise ", "enterprise", "Consumer", "CONSUMER", "", "Enterprise"],
                "value": [1, 2, 3, 4, 5, 1],
            }
        )
        actions = {action.action_id: action for action in suggest_cleaning_actions(frame)}
        self.assertIn("trim_whitespace:segment", actions)
        self.assertIn("blank_to_missing:segment", actions)
        self.assertIn("normalize_categories:segment", actions)
        self.assertEqual(actions["normalize_categories:segment"].confidence, "medium")

    def test_applies_text_fixes_without_mutating_the_source(self) -> None:
        frame = pd.DataFrame({"segment": [" A ", "a", "A", "", None]})
        actions = {
            action.action_id: action
            for action in suggest_cleaning_actions(frame)
        }
        selected = [
            actions["trim_whitespace:segment"],
            actions["blank_to_missing:segment"],
            actions["normalize_categories:segment"],
        ]
        cleaned = apply_cleaning_actions(frame, selected)
        self.assertEqual(cleaned["segment"].iloc[:3].tolist(), ["A", "A", "A"])
        self.assertTrue(pd.isna(cleaned["segment"].iloc[3]))
        self.assertEqual(frame["segment"].iloc[0], " A ")
        self.assertEqual(frame["segment"].iloc[3], "")

    def test_removes_only_later_exact_duplicate_rows(self) -> None:
        frame = pd.DataFrame({"name": ["A", "A", "B"], "value": [1, 1, 2]})
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "drop_duplicates"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        self.assertEqual(cleaned.to_dict(orient="records"), [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ])

    def test_converts_plain_numeric_text_but_preserves_identifier_codes(self) -> None:
        frame = pd.DataFrame(
            {
                "amount": ["10", "20", "30", "40"],
                "account_code": ["001", "002", "003", "004"],
            }
        )
        actions = {action.action_id: action for action in suggest_cleaning_actions(frame)}
        self.assertIn("convert_numeric:amount", actions)
        self.assertNotIn("convert_numeric:account_code", actions)
        cleaned = apply_cleaning_actions(frame, [actions["convert_numeric:amount"]])
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["amount"]))
        self.assertEqual(cleaned["amount"].tolist(), [10, 20, 30, 40])
        self.assertEqual(cleaned["account_code"].tolist(), ["001", "002", "003", "004"])

    def test_converts_unambiguous_date_text(self) -> None:
        frame = pd.DataFrame(
            {"event_date": ["2026-01-01", "2026-01-02", "2026-01-03"]}
        )
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "convert_datetime"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["event_date"]))
        self.assertEqual(cleaned["event_date"].dt.day.tolist(), [1, 2, 3])

    def test_offers_reviewable_missing_value_fills(self) -> None:
        frame = pd.DataFrame(
            {
                "score": [10.0, 20.0, 30.0, None, 40.0, 50.0],
                "segment": ["A", "A", "B", "B", "A", None],
            }
        )
        actions = {action.action_id: action for action in suggest_cleaning_actions(frame)}
        self.assertEqual(actions["fill_numeric_median:score"].confidence, "medium")
        self.assertEqual(actions["fill_category_missing:segment"].confidence, "medium")
        cleaned = apply_cleaning_actions(
            frame,
            [
                actions["fill_numeric_median:score"],
                actions["fill_category_missing:segment"],
            ],
        )
        self.assertEqual(cleaned["score"].iloc[3], 30.0)
        self.assertEqual(cleaned["segment"].iloc[5], "Missing")

    def test_missing_category_label_does_not_collide_by_case(self) -> None:
        frame = pd.DataFrame(
            {"segment": ["missing", "A", "A", "B", "B", None]}
        )
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "fill_category_missing"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        self.assertEqual(cleaned["segment"].iloc[5], "Missing (2)")

    def test_flags_robust_outliers_without_changing_values(self) -> None:
        frame = pd.DataFrame({"revenue": [10, 11, 12, 13, 14, 15, 16, 1000]})
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "flag_outliers"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        self.assertEqual(cleaned["revenue"].tolist(), frame["revenue"].tolist())
        self.assertEqual(cleaned["revenue_is_outlier"].tolist(), [False] * 7 + [True])
        self.assertNotIn(
            "flag_outliers",
            {item.kind for item in suggest_cleaning_actions(cleaned)},
        )

    def test_outlier_rule_handles_infinity_without_distorting_iqr_bounds(self) -> None:
        frame = pd.DataFrame(
            {"revenue": [10, 11, 12, 13, 14, 15, 16, 17, float("inf")]}
        )
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "flag_outliers"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        self.assertTrue(bool(cleaned["revenue_is_outlier"].iloc[-1]))

    def test_each_irregular_dataset_suggestion_applies_independently(self) -> None:
        frame = pd.DataFrame(
            {
                "segment": [" A ", "a", "B", "B", "", None, "A", "B"],
                "amount": ["10", "20", "30", "40", "50", "60", "70", "1000"],
                "event_date": [f"2026-01-{day:02d}" for day in range(1, 9)],
                "empty": [None] * 8,
            }
        )
        actions = suggest_cleaning_actions(frame)
        self.assertGreaterEqual(len(actions), 6)
        for action in actions:
            with self.subTest(action=action.action_id):
                cleaned = apply_cleaning_actions(frame, [action])
                self.assertIsInstance(cleaned, pd.DataFrame)

    def test_replays_batches_for_undo_without_storing_dataframe_snapshots(self) -> None:
        frame = pd.DataFrame({"label": [" A ", "B", "B"]})
        initial = {action.action_id: action for action in suggest_cleaning_actions(frame)}
        first_batch = (initial["trim_whitespace:label"],)
        after_first = apply_cleaning_actions(frame, first_batch)
        second = next(
            action
            for action in suggest_cleaning_actions(after_first)
            if action.kind == "drop_duplicates"
        )
        final = replay_cleaning_batches(frame, [first_batch, (second,)])
        undone = replay_cleaning_batches(frame, [first_batch])
        self.assertEqual(len(final), 2)
        self.assertEqual(len(undone), 3)
        self.assertEqual(undone["label"].tolist(), ["A", "B", "B"])

    def test_recipe_records_source_result_and_ordered_operations(self) -> None:
        frame = pd.DataFrame({"label": [" A ", "A"]})
        action = next(
            action
            for action in suggest_cleaning_actions(frame)
            if action.kind == "trim_whitespace"
        )
        cleaned = apply_cleaning_actions(frame, [action])
        recipe = json.loads(
            build_cleaning_recipe(
                "sample.csv",
                "abc123",
                frame,
                cleaned,
                [(action,)],
            )
        )
        self.assertEqual(recipe["version"], 1)
        self.assertEqual(recipe["source"]["sha256"], "abc123")
        self.assertEqual(recipe["operations"][0]["operation"], "trim_whitespace")
        self.assertEqual(recipe["result"]["rows"], 2)

    def test_recipe_uses_the_actual_execution_order(self) -> None:
        frame = pd.DataFrame({"label": [" A ", "", "A"]})
        actions = {action.kind: action for action in suggest_cleaning_actions(frame)}
        selected = [actions["blank_to_missing"], actions["trim_whitespace"]]
        cleaned = apply_cleaning_actions(frame, selected)
        recipe = json.loads(
            build_cleaning_recipe("sample.csv", "abc123", frame, cleaned, [selected])
        )
        self.assertEqual(
            [step["operation"] for step in recipe["operations"]],
            ["trim_whitespace", "blank_to_missing"],
        )

    def test_rejects_duplicate_column_names(self) -> None:
        frame = pd.DataFrame([[1, 2]], columns=["value", "value"])
        with self.assertRaisesRegex(CleaningError, "unique column names"):
            suggest_cleaning_actions(frame)

    def test_rejects_dropping_and_cleaning_the_same_column_in_one_batch(self) -> None:
        frame = pd.DataFrame({"label": [" A ", "A", None, None, None]})
        actions = list(suggest_cleaning_actions(frame))
        trim = next(action for action in actions if action.kind == "trim_whitespace")
        drop = next(action for action in actions if action.kind == "drop_column")
        with self.assertRaisesRegex(CleaningError, "either dropping or cleaning"):
            apply_cleaning_actions(frame, [trim, drop])


class ProfilingMissingValueTests(unittest.TestCase):
    def test_whitespace_only_text_counts_as_missing(self) -> None:
        frame = pd.DataFrame({"value": ["A", "", "   ", None]})
        profile = profile_dataset(frame)
        self.assertEqual(profile.missing_cells, 3)
        row = profile.column_profile.iloc[0]
        self.assertEqual(int(row["missing_count"]), 3)
        self.assertEqual(float(row["missing_percent"]), 75.0)


if __name__ == "__main__":
    unittest.main()
