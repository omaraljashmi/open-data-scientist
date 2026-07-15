"""Run the deterministic multi-format validation matrix for stable promotion."""

from __future__ import annotations

from base64 import b64decode
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

from ods import (
    DEFAULT_LIMITS,
    DatasetLimits,
    DatasetLoadError,
    QuerySpec,
    analyze_query,
    apply_cleaning_actions,
    build_card_result,
    build_cleaning_recipe,
    build_dashboard_html,
    build_markdown_report,
    dashboard_config_from_json,
    dashboard_config_to_json,
    default_dashboard_config,
    execute_query,
    load_dataset,
    profile_dataset,
    suggest_cleaning_actions,
    validate_dashboard_config,
)


@dataclass(frozen=True)
class ValidationCase:
    """One safe synthetic file expected to complete the full ODS workflow."""

    name: str
    file_name: str
    content: bytes
    expected_rows: int
    expected_columns: tuple[str, ...]


@dataclass(frozen=True)
class RejectionCase:
    """One unsafe or malformed file expected to fail with a clear message."""

    name: str
    file_name: str
    content: bytes
    expected_message: str
    limits: DatasetLimits = DEFAULT_LIMITS


@dataclass(frozen=True)
class ValidationResult:
    """Compact evidence from one successful end-to-end validation case."""

    name: str
    file_name: str
    input_rows: int
    result_rows: int
    columns: int
    quality_score: int
    cleaning_recommendations: int
    applied_actions: int
    dashboard_cards: int
    query_rows: int
    sql_score: int


def build_validation_cases() -> tuple[ValidationCase, ...]:
    """Return representative CSV, XLSX, and legacy XLS inputs."""
    standard_csv = (
        "customer_id,segment,region,revenue,order_date,status\n"
        "C-001,Consumer,East,120.50,2026-01-03,Active\n"
        "C-002,Enterprise,West,980.00,2026-01-04,Active\n"
        "C-003,Consumer,East,,2026-01-05,Paused\n"
        "C-004,Small Business,South,250.25,2026-01-06, Active \n"
    ).encode("utf-8")
    international_csv = (
        "record_id;category;amount;event_date\n"
        "R-001;Café;15.50;2026-02-01\n"
        "R-002;Éducation;30.00;2026-02-02\n"
        "R-003;Café;12.75;2026-02-03\n"
    ).encode("utf-8-sig")
    quoted_csv = (
        "case_id,category,amount,event_date,notes\n"
        "A-1,Support,45.00,2026-03-01,\"Line one\nLine two, with comma\"\n"
        "A-2,Sales,82.25,2026-03-02,\"Quoted, safely\"\n"
    ).encode("utf-8")
    single_column_csv = b"status\n\"active, urgent\"\npaused\nreview\n"
    sparse_pipe_csv = (
        "row_id|segment|score|event_date|comment\n"
        "1|A|10|2026-04-01|ok\n"
        "2|B||2026-04-02|\n"
        "3|A|30|2026-04-03|follow up\n"
        "4|B|40|2026-04-04|ok\n"
    ).encode("utf-8")

    workbook = BytesIO()
    mixed_frame = pd.DataFrame(
        {
            "order_id": ["O-001", "O-002", "O-003"],
            "region": ["North", "South", "North"],
            "quantity": [2, 5, 1],
            "revenue": [42.50, 125.00, 20.25],
            "order_date": pd.to_datetime(
                ["2026-05-01", "2026-05-02", "2026-05-03"]
            ),
            "fulfilled": [True, False, True],
        }
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        mixed_frame.to_excel(writer, sheet_name="orders", index=False)
        pd.DataFrame({"note": ["The loader intentionally reads the first sheet."]}).to_excel(
            writer,
            sheet_name="metadata",
            index=False,
        )

    legacy_fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "legacy_customers.xls.b64"
    )
    legacy_xls = b64decode(legacy_fixture.read_text(encoding="ascii"))

    return (
        ValidationCase(
            "standard-csv",
            "standard_customers.csv",
            standard_csv,
            4,
            ("customer_id", "segment", "region", "revenue", "order_date", "status"),
        ),
        ValidationCase(
            "utf8-bom-semicolon-csv",
            "international.csv",
            international_csv,
            3,
            ("record_id", "category", "amount", "event_date"),
        ),
        ValidationCase(
            "quoted-multiline-csv",
            "quoted_notes.csv",
            quoted_csv,
            2,
            ("case_id", "category", "amount", "event_date", "notes"),
        ),
        ValidationCase(
            "single-column-csv",
            "statuses.csv",
            single_column_csv,
            3,
            ("status",),
        ),
        ValidationCase(
            "sparse-pipe-csv",
            "sparse_records.csv",
            sparse_pipe_csv,
            4,
            ("row_id", "segment", "score", "event_date", "comment"),
        ),
        ValidationCase(
            "mixed-type-xlsx",
            "orders.xlsx",
            workbook.getvalue(),
            3,
            tuple(str(column) for column in mixed_frame.columns),
        ),
        ValidationCase(
            "legacy-xls",
            "legacy_customers.xls",
            legacy_xls,
            3,
            ("customer_id", "segment", "revenue"),
        ),
    )


def build_rejection_cases() -> tuple[RejectionCase, ...]:
    """Return files that must be rejected before analysis begins."""
    return (
        RejectionCase("unsupported-type", "notes.txt", b"hello", "Unsupported file type"),
        RejectionCase("empty-file", "empty.csv", b"", "empty"),
        RejectionCase("header-only", "header.csv", b"name,age\n", "no data rows"),
        RejectionCase(
            "binary-disguised-as-csv",
            "binary.csv",
            b"name\nvalue\x00\xff",
            "binary or UTF-16",
        ),
        RejectionCase(
            "malformed-csv",
            "malformed.csv",
            b"name,age\n\"Omar,21\n",
            "Could not parse",
        ),
        RejectionCase(
            "blank-csv-header",
            "blank_header.csv",
            b"name,,age\nOmar,x,21\n",
            "non-empty header",
        ),
        RejectionCase(
            "duplicate-csv-header",
            "duplicate_header.csv",
            b"name,name\nOmar,Ali\n",
            "must be unique",
        ),
        RejectionCase(
            "damaged-xlsx",
            "damaged.xlsx",
            b"not a workbook",
            "damaged",
        ),
        RejectionCase(
            "oversized-upload",
            "oversized.csv",
            b"name\nOmar\n",
            "per file",
            DatasetLimits(max_upload_bytes=4),
        ),
    )


def validate_case(case: ValidationCase) -> ValidationResult:
    """Exercise loading, profiling, cleaning, dashboards, querying, and exports."""
    frame = load_dataset(case.file_name, case.content)
    if frame.shape != (case.expected_rows, len(case.expected_columns)):
        raise AssertionError(
            f"{case.name} loaded as {frame.shape}, expected "
            f"{(case.expected_rows, len(case.expected_columns))}."
        )
    if tuple(frame.columns) != case.expected_columns:
        raise AssertionError(
            f"{case.name} columns were {tuple(frame.columns)!r}, "
            f"expected {case.expected_columns!r}."
        )

    profile = profile_dataset(frame)
    recommendations = suggest_cleaning_actions(frame)
    selected_actions = tuple(
        action
        for action in recommendations
        if action.confidence == "high" and action.kind != "drop_column"
    )
    cleaned = apply_cleaning_actions(frame, selected_actions)
    batches = (selected_actions,) if selected_actions else ()
    recipe = json.loads(
        build_cleaning_recipe(
            case.file_name,
            sha256(case.content).hexdigest(),
            frame,
            cleaned,
            batches,
        )
    )
    if recipe["result"]["rows"] != len(cleaned):
        raise AssertionError(f"{case.name} cleaning recipe row count is inconsistent.")

    dashboard = default_dashboard_config(cleaned)
    validate_dashboard_config(cleaned, dashboard)
    card_results = tuple(build_card_result(cleaned, card) for card in dashboard.cards)
    dashboard_json = dashboard_config_to_json(dashboard)
    if dashboard_config_from_json(dashboard_json, cleaned) != dashboard:
        raise AssertionError(f"{case.name} dashboard JSON did not round-trip.")
    dashboard_html = build_dashboard_html(cleaned, dashboard)
    if "<!doctype html>" not in dashboard_html.casefold():
        raise AssertionError(f"{case.name} dashboard export is not standalone HTML.")

    selected_columns = tuple(str(column) for column in cleaned.columns[:2])
    query_result = execute_query(
        cleaned,
        QuerySpec(selected_columns=selected_columns, limit=10),
    )
    sql_analysis = analyze_query(cleaned, "SELECT * FROM uploaded_data LIMIT 10")
    if not sql_analysis.plan_steps:
        raise AssertionError(f"{case.name} SQL Coach returned no plan evidence.")

    report = build_markdown_report(case.file_name, profile)
    if case.file_name not in report:
        raise AssertionError(f"{case.name} quality report omitted the source name.")

    return ValidationResult(
        name=case.name,
        file_name=case.file_name,
        input_rows=len(frame),
        result_rows=len(cleaned),
        columns=len(cleaned.columns),
        quality_score=profile.health_score,
        cleaning_recommendations=len(recommendations),
        applied_actions=len(selected_actions),
        dashboard_cards=len(card_results),
        query_rows=len(query_result.dataframe),
        sql_score=sql_analysis.score,
    )


def validate_rejection(case: RejectionCase) -> dict[str, str]:
    """Confirm one unsafe input fails with the documented error family."""
    try:
        load_dataset(case.file_name, case.content, limits=case.limits)
    except DatasetLoadError as exc:
        message = str(exc)
        if case.expected_message.casefold() not in message.casefold():
            raise AssertionError(
                f"{case.name} returned {message!r}; expected "
                f"{case.expected_message!r}."
            ) from exc
        return {"name": case.name, "message": message}
    raise AssertionError(f"{case.name} was accepted but should have been rejected.")


def run_validation_matrix() -> dict[str, object]:
    """Run every valid and invalid case and return JSON-serializable evidence."""
    valid = [asdict(validate_case(case)) for case in build_validation_cases()]
    rejected = [validate_rejection(case) for case in build_rejection_cases()]
    return {
        "status": "passed",
        "valid_cases": valid,
        "rejection_cases": rejected,
        "summary": {
            "valid_files": len(valid),
            "rejected_files": len(rejected),
            "formats": ["csv", "xlsx", "xls"],
        },
    }


def main() -> int:
    print(json.dumps(run_validation_matrix(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
