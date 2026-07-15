"""Run the deterministic release performance check from the command line."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import pandas as pd

from ods import (
    __release__,
    build_card_result,
    default_dashboard_config,
    profile_dataset,
    suggest_cleaning_actions,
)


def build_dataset(rows: int) -> pd.DataFrame:
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


def run(rows: int) -> dict[str, object]:
    dataframe = build_dataset(rows)
    started = perf_counter()
    profile = profile_dataset(dataframe)
    actions = suggest_cleaning_actions(dataframe)
    config = default_dashboard_config(dataframe)
    results = [build_card_result(dataframe, card) for card in config.cards]
    elapsed = perf_counter() - started
    return {
        "release": __release__,
        "rows": rows,
        "columns": len(dataframe.columns),
        "seconds": round(elapsed, 3),
        "quality_score": profile.health_score,
        "cleaning_recommendations": len(actions),
        "dashboard_cards": len(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.rows <= 0 or arguments.max_seconds <= 0:
        parser.error("--rows and --max-seconds must be greater than zero")

    result = run(arguments.rows)
    print(json.dumps(result, indent=2))
    return 0 if float(result["seconds"]) <= arguments.max_seconds else 1


if __name__ == "__main__":
    raise SystemExit(main())
