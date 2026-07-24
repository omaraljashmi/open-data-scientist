"""Run a saved ODS export pipeline headlessly (cron, CI, or the command line).

Credentials are read from environment variables only, so tokens never appear
in pipeline files or shell history:

    AIRTABLE_TOKEN      personal access token for Airtable pipelines
    ODS_WEBHOOK_SECRET  value for the webhook's secret header, when configured

Example:

    python -m scripts.run_pipeline \
        --input examples/sample_customers.csv \
        --pipeline my-export-pipeline.json \
        --cleaning-recipe my-cleaning-recipe.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from ods import (
    CleaningError,
    DatasetLoadError,
    PipelineError,
    cleaning_actions_from_recipe,
    dataframe_to_records,
    load_dataset,
    pipeline_config_from_json,
    preview_payload,
    push_to_airtable,
    push_to_webhook,
    replay_cleaning_batches,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_pipeline",
        description="Load a dataset, optionally replay a cleaning recipe, and push to the pipeline destination.",
    )
    parser.add_argument("--input", required=True, help="CSV or Excel file to load.")
    parser.add_argument("--pipeline", required=True, help="Pipeline JSON saved from the Pipeline page.")
    parser.add_argument(
        "--cleaning-recipe",
        help="Optional cleaning recipe JSON to replay before sending.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        help="Override the pipeline's stored row limit for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first request payload and exit without sending anything.",
    )
    return parser.parse_args(argv)


def _print_progress(done: int, total: int) -> None:
    print(f"  sent {min(done, total):,} / {total:,} records", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)

    try:
        config = pipeline_config_from_json(Path(args.pipeline).read_text(encoding="utf-8"))
    except (OSError, PipelineError) as exc:
        print(f"error: could not load the pipeline file: {exc}", file=sys.stderr)
        return 1

    try:
        dataframe = load_dataset(input_path.name, input_path.read_bytes())
    except (OSError, DatasetLoadError) as exc:
        print(f"error: could not load the dataset: {exc}", file=sys.stderr)
        return 1
    print(f"loaded {input_path.name}: {len(dataframe):,} rows, {len(dataframe.columns):,} columns")

    if args.cleaning_recipe:
        try:
            batches = cleaning_actions_from_recipe(
                Path(args.cleaning_recipe).read_text(encoding="utf-8")
            )
            dataframe = replay_cleaning_batches(dataframe, batches)
        except (OSError, CleaningError) as exc:
            print(f"error: could not replay the cleaning recipe: {exc}", file=sys.stderr)
            return 1
        applied = sum(len(batch) for batch in batches)
        print(f"replayed cleaning recipe: {applied:,} fixes → {len(dataframe):,} rows")

    row_limit = args.row_limit if args.row_limit is not None else config.row_limit
    try:
        records = dataframe_to_records(dataframe, row_limit=row_limit)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry run — first request payload for {config.kind} destination:")
        print(preview_payload(records, config))
        return 0

    try:
        if config.kind == "airtable":
            assert config.airtable is not None
            token = os.environ.get("AIRTABLE_TOKEN", "")
            report = push_to_airtable(
                records, config.airtable, token, progress=_print_progress
            )
        else:
            assert config.webhook is not None
            secret = os.environ.get("ODS_WEBHOOK_SECRET") or None
            report = push_to_webhook(
                records,
                config.webhook,
                secret_value=secret,
                progress=_print_progress,
                pipeline_name=config.name,
            )
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"done: {report.sent_records:,}/{report.total_records:,} records "
        f"in {report.batches_sent:,} requests → {report.destination}"
    )
    for failure in report.failures:
        print(f"failure: {failure}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
