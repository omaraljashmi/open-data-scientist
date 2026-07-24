"""Local, auditable dashboard composition for Data Insight Studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from html import escape
import json
from math import isfinite
import re
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)
import plotly.graph_objects as go

from . import chart_theme
from .dashboard import infer_column_semantics


CardKind = Literal["kpi", "bar", "pie", "line", "scatter", "distribution"]
Metric = Literal["row_count", "sum", "mean", "median", "distinct_count"]
FilterKind = Literal["values", "range", "date_range"]
DateGrain = Literal["day", "week", "month", "quarter", "year"]

MAX_CARDS = 12
MAX_FILTERS = 5
MAX_CONFIG_BYTES = 100_000
MAX_SCATTER_POINTS = 5_000
MISSING_LABEL = "(missing)"

CARD_KINDS: tuple[CardKind, ...] = (
    "kpi",
    "bar",
    "pie",
    "line",
    "scatter",
    "distribution",
)
METRICS: tuple[Metric, ...] = (
    "row_count",
    "sum",
    "mean",
    "median",
    "distinct_count",
)
DATE_GRAINS: tuple[DateGrain, ...] = (
    "day",
    "week",
    "month",
    "quarter",
    "year",
)
METRIC_LABELS: dict[Metric, str] = {
    "row_count": "Row count",
    "sum": "Total",
    "mean": "Average",
    "median": "Median",
    "distinct_count": "Distinct count",
}


class DashboardStudioError(ValueError):
    """Raised when a dashboard configuration or calculation is unsafe."""


@dataclass(frozen=True)
class DashboardFilter:
    """One global filter applied to every dashboard card."""

    filter_id: str
    column: str
    kind: FilterKind
    values: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class DashboardCard:
    """A KPI or chart definition that can be serialized without source data."""

    card_id: str
    kind: CardKind
    title: str
    metric: Metric = "row_count"
    column: str | None = None
    x: str | None = None
    y: str | None = None
    aggregation: Metric = "row_count"
    date_grain: DateGrain = "month"
    top_n: int = 12
    bins: int = 12


@dataclass(frozen=True)
class DashboardConfig:
    """Portable dashboard layout and its active filters."""

    name: str
    cards: tuple[DashboardCard, ...]
    filters: tuple[DashboardFilter, ...] = ()


@dataclass(frozen=True)
class CardResult:
    """The exact locally calculated result consumed by one rendered card."""

    card: DashboardCard
    value: int | float | None
    figure: go.Figure | None
    audit_table: pd.DataFrame
    calculation: str


def default_dashboard_config(df: pd.DataFrame) -> DashboardConfig:
    """Create a useful, deterministic starter dashboard for the dataset."""
    _validate_dataframe(df)
    semantics = infer_column_semantics(df)
    numeric = [item.column for item in semantics if item.role == "numeric"]
    categorical = [item.column for item in semantics if item.role == "categorical"]
    datetimes = [item.column for item in semantics if item.role == "datetime"]

    cards: list[DashboardCard] = [
        DashboardCard("card-1", "kpi", "Rows", metric="row_count")
    ]
    next_index = 2
    if numeric:
        cards.append(
            DashboardCard(
                f"card-{next_index}",
                "kpi",
                f"Average {numeric[0]}",
                metric="mean",
                column=numeric[0],
            )
        )
        next_index += 1
    if categorical:
        cards.append(
            DashboardCard(
                f"card-{next_index}",
                "pie",
                f"Share of rows by {categorical[0]}",
                x=categorical[0],
                aggregation="row_count",
            )
        )
        next_index += 1
    if datetimes:
        cards.append(
            DashboardCard(
                f"card-{next_index}",
                "line",
                (
                    f"Average {numeric[0]} over time"
                    if numeric
                    else "Rows over time"
                ),
                x=datetimes[0],
                y=numeric[0] if numeric else None,
                aggregation="mean" if numeric else "row_count",
                date_grain="month",
            )
        )
        next_index += 1
    elif numeric:
        cards.append(
            DashboardCard(
                f"card-{next_index}",
                "distribution",
                f"Distribution of {numeric[0]}",
                column=numeric[0],
            )
        )
        next_index += 1

    if len(cards) < 4 and len(numeric) >= 2:
        cards.append(
            DashboardCard(
                f"card-{next_index}",
                "scatter",
                f"{numeric[1]} versus {numeric[0]}",
                x=numeric[0],
                y=numeric[1],
            )
        )
    return DashboardConfig("My data dashboard", tuple(cards[:4]))


def filter_kind_for_column(df: pd.DataFrame, column: str) -> FilterKind:
    """Choose the safest visual filter control for a source column."""
    _require_column(df, column)
    semantic = next(
        item for item in infer_column_semantics(df) if item.column == column
    )
    if semantic.role == "datetime":
        return "date_range"
    if is_numeric_dtype(df[column]) and not is_bool_dtype(df[column]):
        return "range"
    return "values"


def default_filter_for_column(
    df: pd.DataFrame,
    column: str,
    filter_id: str,
) -> DashboardFilter:
    """Create a filter spanning all available non-missing values."""
    kind = filter_kind_for_column(df, column)
    if kind == "range":
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        minimum = float(values.min()) if not values.empty else 0.0
        maximum = float(values.max()) if not values.empty else 0.0
        return DashboardFilter(
            filter_id,
            column,
            kind,
            minimum=minimum,
            maximum=maximum,
        )
    if kind == "date_range":
        values = _parse_dates(df[column]).dropna()
        today = pd.Timestamp.today().normalize()
        start = values.min().normalize() if not values.empty else today
        end = values.max().normalize() if not values.empty else today
        return DashboardFilter(
            filter_id,
            column,
            kind,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
        )
    return DashboardFilter(filter_id, column, kind)


def categorical_filter_options(
    df: pd.DataFrame,
    column: str,
    *,
    limit: int = 200,
) -> tuple[str, ...]:
    """Return deterministic, frequency-ranked labels for a value filter."""
    _require_column(df, column)
    labels = _category_labels(df[column])
    counts = labels.value_counts(dropna=False)
    ordered = sorted(counts.index.astype(str), key=lambda value: (-int(counts[value]), value))
    return tuple(ordered[:limit])


def apply_dashboard_filters(
    df: pd.DataFrame,
    filters: tuple[DashboardFilter, ...],
) -> pd.DataFrame:
    """Apply validated global filters with inclusive boundaries and AND logic."""
    _validate_dataframe(df)
    result = df.copy(deep=True)
    for item in filters:
        _validate_filter(df, item)
        if item.kind == "values":
            if item.values:
                selected = set(item.values)
                result = result[_category_labels(result[item.column]).isin(selected)]
        elif item.kind == "range":
            numeric = pd.to_numeric(result[item.column], errors="coerce")
            result = result[
                numeric.ge(float(item.minimum)) & numeric.le(float(item.maximum))
            ]
        else:
            dates = _parse_dates(result[item.column])
            start = pd.Timestamp(item.start)
            end = pd.Timestamp(item.end) + pd.Timedelta(1, unit="D")
            result = result[dates.ge(start) & dates.lt(end)]
    return result.copy(deep=True)


def build_card_result(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    """Calculate one KPI or Plotly figure plus its exact audit table."""
    _validate_dataframe(df)
    _validate_card(df, card)
    if card.kind == "kpi":
        return _build_kpi(df, card)
    if card.kind == "bar":
        return _build_bar(df, card)
    if card.kind == "pie":
        return _build_pie(df, card)
    if card.kind == "line":
        return _build_line(df, card)
    if card.kind == "scatter":
        return _build_scatter(df, card)
    return _build_distribution(df, card)


def validate_dashboard_config(df: pd.DataFrame, config: DashboardConfig) -> None:
    """Validate all limits, identifiers, source columns, and calculations."""
    _validate_dataframe(df)
    _validate_text(config.name, "Dashboard name", 80)
    if not 1 <= len(config.cards) <= MAX_CARDS:
        raise DashboardStudioError(
            f"A dashboard must contain between 1 and {MAX_CARDS} cards."
        )
    if len(config.filters) > MAX_FILTERS:
        raise DashboardStudioError(
            f"A dashboard can contain at most {MAX_FILTERS} global filters."
        )
    card_ids = [card.card_id for card in config.cards]
    filter_ids = [item.filter_id for item in config.filters]
    if len(card_ids) != len(set(card_ids)):
        raise DashboardStudioError("Every dashboard card must have a unique ID.")
    if len(filter_ids) != len(set(filter_ids)):
        raise DashboardStudioError("Every global filter must have a unique ID.")
    if len({item.column for item in config.filters}) != len(config.filters):
        raise DashboardStudioError("Add at most one global filter per column.")
    for card in config.cards:
        _validate_card(df, card)
    for item in config.filters:
        _validate_filter(df, item)


def dashboard_config_to_json(config: DashboardConfig) -> str:
    """Serialize a dashboard without embedding its source dataset."""
    payload = {
        "format": "ods-dashboard",
        "version": 1,
        "name": config.name,
        "cards": [asdict(card) for card in config.cards],
        "filters": [asdict(item) for item in config.filters],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def dashboard_config_from_json(text: str, df: pd.DataFrame) -> DashboardConfig:
    """Load and fully validate an untrusted dashboard configuration."""
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise DashboardStudioError("Dashboard configuration files must be 100 KB or smaller.")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DashboardStudioError("This is not valid dashboard JSON.") from exc
    if not isinstance(payload, dict):
        raise DashboardStudioError("Dashboard JSON must contain one object.")
    if payload.get("format") != "ods-dashboard" or payload.get("version") != 1:
        raise DashboardStudioError("Only Data Insight Studio dashboard format version 1 is supported.")
    if set(payload) - {"format", "version", "name", "cards", "filters"}:
        raise DashboardStudioError("The dashboard JSON contains unsupported fields.")
    raw_cards = payload.get("cards")
    raw_filters = payload.get("filters", [])
    if not isinstance(raw_cards, list) or not isinstance(raw_filters, list):
        raise DashboardStudioError("Dashboard cards and filters must be JSON arrays.")
    try:
        cards = tuple(_card_from_payload(item) for item in raw_cards)
        filters = tuple(_filter_from_payload(item) for item in raw_filters)
        config = DashboardConfig(payload.get("name", ""), cards, filters)
    except (TypeError, ValueError, KeyError) as exc:
        raise DashboardStudioError("The dashboard configuration has invalid fields.") from exc
    try:
        validate_dashboard_config(df, config)
    except DashboardStudioError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise DashboardStudioError(
            "The dashboard configuration has invalid field types."
        ) from exc
    return config


def next_dashboard_id(config: DashboardConfig, prefix: str) -> str:
    """Return a stable unused card or filter identifier."""
    used = {card.card_id for card in config.cards} | {
        item.filter_id for item in config.filters
    }
    index = 1
    while f"{prefix}-{index}" in used:
        index += 1
    return f"{prefix}-{index}"


def replace_dashboard_card(
    config: DashboardConfig,
    updated: DashboardCard,
) -> DashboardConfig:
    """Replace one card while preserving its position."""
    if updated.card_id not in {card.card_id for card in config.cards}:
        raise DashboardStudioError("The dashboard card no longer exists.")
    return replace(
        config,
        cards=tuple(
            updated if card.card_id == updated.card_id else card
            for card in config.cards
        ),
    )


def move_dashboard_card(
    config: DashboardConfig,
    card_id: str,
    direction: int,
) -> DashboardConfig:
    """Move a card one position earlier or later."""
    cards = list(config.cards)
    try:
        index = next(i for i, card in enumerate(cards) if card.card_id == card_id)
    except StopIteration as exc:
        raise DashboardStudioError("The dashboard card no longer exists.") from exc
    destination = index + (-1 if direction < 0 else 1)
    if not 0 <= destination < len(cards):
        return config
    cards[index], cards[destination] = cards[destination], cards[index]
    return replace(config, cards=tuple(cards))


def remove_dashboard_card(config: DashboardConfig, card_id: str) -> DashboardConfig:
    """Remove one card while keeping at least one visible result."""
    if len(config.cards) <= 1:
        raise DashboardStudioError("A dashboard must keep at least one card.")
    cards = tuple(card for card in config.cards if card.card_id != card_id)
    if len(cards) == len(config.cards):
        raise DashboardStudioError("The dashboard card no longer exists.")
    return replace(config, cards=cards)


def replace_dashboard_filter(
    config: DashboardConfig,
    updated: DashboardFilter,
) -> DashboardConfig:
    """Replace one global filter while preserving its position."""
    if updated.filter_id not in {item.filter_id for item in config.filters}:
        raise DashboardStudioError("The global filter no longer exists.")
    return replace(
        config,
        filters=tuple(
            updated if item.filter_id == updated.filter_id else item
            for item in config.filters
        ),
    )


def build_dashboard_html(df: pd.DataFrame, config: DashboardConfig) -> str:
    """Export a responsive, standalone dashboard with Plotly.js embedded inline."""
    validate_dashboard_config(df, config)
    filtered = apply_dashboard_filters(df, config.filters)
    card_html: list[str] = []
    plotly_included = False
    for card in config.cards:
        result = build_card_result(filtered, card)
        title = escape(card.title)
        if card.kind == "kpi":
            body = (
                '<div class="kpi-value">'
                f"{escape(format_metric_value(result.value))}</div>"
            )
        elif result.figure is not None:
            body = result.figure.to_html(
                full_html=False,
                include_plotlyjs=True if not plotly_included else False,
                config={"displaylogo": False, "responsive": True},
            )
            plotly_included = True
        else:
            body = '<p class="empty">No rows match this card.</p>'
        audit = result.audit_table
        visible_audit = audit.head(1_000)
        omitted = len(audit) - len(visible_audit)
        audit_note = (
            f"<p>{omitted:,} additional rows are not shown in this table.</p>"
            if omitted
            else ""
        )
        details = (
            "<details><summary>Calculation details</summary>"
            f"<p>{escape(result.calculation)}</p>"
            f"{visible_audit.to_html(index=False, border=0, escape=True)}"
            f"{audit_note}</details>"
        )
        card_html.append(
            f'<article class="card"><h2>{title}</h2>{body}{details}</article>'
        )

    filters = "No global filters"
    if config.filters:
        filters = "; ".join(_filter_description(item) for item in config.filters)
    safe_name = escape(config.name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{safe_name}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f7f3e7; color: #16281e; }}
    main {{ width: min(1480px, calc(100% - 40px)); margin: 0 auto; padding: 42px 0 64px; }}
    header {{ margin-bottom: 24px; }}
    .kicker {{ color: #15803d; letter-spacing: .12em; font-size: 12px; font-weight: 800; }}
    h1 {{ margin: 8px 0; font-size: clamp(32px, 5vw, 58px); letter-spacing: -.045em; }}
    .meta, .empty {{ color: #5f6f5f; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .card {{ min-width: 0; border: 1px solid rgba(20,83,45,.2); border-radius: 18px; background: #fffdf6; padding: 20px; box-shadow: 0 18px 45px rgba(22,40,30,.08); }}
    .card h2 {{ margin: 0 0 12px; font-size: 19px; }}
    .kpi-value {{ color: #15803d; font-size: clamp(42px, 7vw, 76px); font-weight: 800; letter-spacing: -.04em; padding: 20px 0 28px; }}
    details {{ border-top: 1px solid rgba(20,83,45,.16); margin-top: 14px; padding-top: 12px; color: #47584c; }}
    summary {{ cursor: pointer; color: #15803d; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid rgba(20,83,45,.15); padding: 7px; text-align: left; }}
    @media (max-width: 820px) {{ main {{ width: min(100% - 24px, 1480px); }} .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header><div class="kicker">DATA INSIGHT STUDIO · LOCAL EXPORT</div><h1>{safe_name}</h1>
  <p class="meta">{len(filtered):,} of {len(df):,} rows · {escape(filters)}</p></header>
  <section class="grid">{''.join(card_html)}</section>
</main>
</body>
</html>"""


def format_metric_value(value: int | float | None) -> str:
    """Format KPI values consistently without hiding their precision."""
    if value is None or pd.isna(value):
        return "No data"
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    magnitude = abs(number)
    if magnitude >= 1_000_000:
        return f"{number:,.1f}"
    if magnitude >= 100:
        return f"{number:,.2f}"
    return f"{number:,.3f}".rstrip("0").rstrip(".")


def _build_kpi(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    if card.metric == "row_count":
        value: int | float | None = len(df)
        valid_rows = len(df)
        source = "all rows"
    elif card.metric == "distinct_count":
        assert card.column is not None
        value = int(df[card.column].nunique(dropna=True))
        valid_rows = int(df[card.column].notna().sum())
        source = f"non-missing {card.column} values"
    else:
        assert card.column is not None
        values = pd.to_numeric(df[card.column], errors="coerce").dropna()
        valid_rows = len(values)
        source = f"valid numeric {card.column} values"
        if values.empty:
            value = None
        elif card.metric == "sum":
            value = float(values.sum())
        elif card.metric == "mean":
            value = float(values.mean())
        else:
            value = float(values.median())
    audit = pd.DataFrame(
        [
            {
                "metric": METRIC_LABELS[card.metric],
                "source_column": card.column or "(rows)",
                "valid_rows": valid_rows,
                "value": value,
            }
        ]
    )
    calculation = (
        f"{METRIC_LABELS[card.metric]} calculated from {source} after global filters; "
        f"the filtered dataset contains {len(df):,} rows."
    )
    return CardResult(card, value, None, audit, calculation)


def _build_bar(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    assert card.x is not None
    categories = _category_labels(df[card.x])
    prepared = pd.DataFrame({"category": categories})
    if card.aggregation == "row_count":
        audit = prepared.groupby("category", as_index=False).size().rename(
            columns={"size": "value"}
        )
        source = "row count"
    else:
        assert card.y is not None
        prepared["raw_value"] = df[card.y]
        if card.aggregation == "distinct_count":
            audit = (
                prepared.groupby("category", as_index=False)["raw_value"]
                .nunique(dropna=True)
                .rename(columns={"raw_value": "value"})
            )
        else:
            prepared["value"] = pd.to_numeric(prepared["raw_value"], errors="coerce")
            prepared = prepared.dropna(subset=["value"])
            grouped = prepared.groupby("category", as_index=False)["value"]
            audit = _run_group_aggregation(grouped, card.aggregation)
        source = f"{METRIC_LABELS[card.aggregation].lower()} of {card.y}"
    audit = audit.sort_values(
        ["value", "category"], ascending=[False, True], kind="stable"
    ).head(card.top_n).reset_index(drop=True)
    figure = go.Figure(
        go.Bar(
            x=[escape(str(value)) for value in audit["category"]],
            y=audit["value"],
            marker_color="#15803d",
            hovertemplate="%{x}<br>%{y:,.4g}<extra></extra>",
        )
    )
    _style_figure(figure, card, x_title=card.x, y_title=source)
    calculation = (
        f"{source.title()} grouped by {card.x}; the {card.top_n} largest groups are shown "
        f"from {len(df):,} filtered rows. Missing categories use {MISSING_LABEL}."
    )
    return CardResult(card, None, figure, audit, calculation)


def _build_pie(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    assert card.x is not None
    categories = _category_labels(df[card.x])
    prepared = pd.DataFrame({"category": categories})
    if card.aggregation == "row_count":
        audit = prepared.groupby("category", as_index=False).size().rename(
            columns={"size": "value"}
        )
        source = "row count"
    elif card.aggregation == "distinct_count":
        assert card.y is not None
        prepared["raw_value"] = df[card.y]
        audit = (
            prepared.groupby("category", as_index=False)["raw_value"]
            .nunique(dropna=True)
            .rename(columns={"raw_value": "value"})
        )
        source = f"distinct count of {card.y}"
    else:  # validation only lets additive aggregations reach a pie
        assert card.y is not None
        prepared["value"] = pd.to_numeric(df[card.y], errors="coerce")
        prepared = prepared.dropna(subset=["value"])
        grouped = prepared.groupby("category", as_index=False)["value"]
        audit = _run_group_aggregation(grouped, "sum")
        source = f"total of {card.y}"
    audit = audit.sort_values(
        ["value", "category"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    top = audit.head(card.top_n).copy()
    hidden = len(audit) - len(top)
    # Distinct counts are not additive across groups, so smaller groups are
    # excluded rather than merged into a misleading "(other)" slice.
    lumped = False
    if hidden > 0 and card.aggregation in {"row_count", "sum"}:
        rest = float(audit.iloc[card.top_n :]["value"].sum())
        if rest > 0:
            top.loc[len(top)] = {"category": "(other)", "value": rest}
            lumped = True
    total = float(top["value"].sum())
    top["share_percent"] = (
        (top["value"] / total * 100).round(2) if total else 0.0
    )

    figure = go.Figure(
        go.Pie(
            labels=[escape(str(value)) for value in top["category"]],
            values=top["value"],
            hole=0.45,
            sort=False,
            marker={"colors": list(chart_theme.CATEGORICAL)},
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,.4g} · %{percent}<extra></extra>",
        )
    )
    _style_figure(figure, card, x_title="", y_title="")
    if lumped:
        tail_note = f"groups beyond the top {card.top_n} are combined into (other)"
    elif hidden > 0:
        tail_note = (
            f"{hidden} smaller group(s) are excluded because {source} is not additive"
        )
    else:
        tail_note = "all groups are shown"
    calculation = (
        f"Share of {source} grouped by {card.x} from {len(df):,} filtered rows; "
        f"{tail_note}. Missing categories use {MISSING_LABEL}."
    )
    return CardResult(card, None, figure, top, calculation)


def _build_line(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    assert card.x is not None
    periods = _group_dates(_parse_dates(df[card.x]), card.date_grain)
    prepared = pd.DataFrame({"period": periods})
    if card.aggregation == "row_count":
        audit = (
            prepared.dropna()
            .groupby("period", as_index=False)
            .size()
            .rename(columns={"size": "value"})
        )
        source = "row count"
    else:
        assert card.y is not None
        prepared["raw_value"] = df[card.y]
        prepared = prepared.dropna(subset=["period"])
        if card.aggregation == "distinct_count":
            audit = (
                prepared.groupby("period", as_index=False)["raw_value"]
                .nunique(dropna=True)
                .rename(columns={"raw_value": "value"})
            )
        else:
            prepared["value"] = pd.to_numeric(prepared["raw_value"], errors="coerce")
            prepared = prepared.dropna(subset=["value"])
            grouped = prepared.groupby("period", as_index=False)["value"]
            audit = _run_group_aggregation(grouped, card.aggregation)
        source = f"{METRIC_LABELS[card.aggregation].lower()} of {card.y}"
    audit = audit.sort_values("period", kind="stable").reset_index(drop=True)
    figure = go.Figure(
        go.Scatter(
            x=audit["period"],
            y=audit["value"],
            mode="lines+markers",
            line={"color": "#15803d", "width": 3},
            marker={"size": 7, "color": "#1f9a4c"},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4g}<extra></extra>",
        )
    )
    _style_figure(figure, card, x_title=f"{card.x} ({card.date_grain})", y_title=source)
    calculation = (
        f"{source.title()} grouped by valid {card.x} values at {card.date_grain} grain "
        f"after global filters. Invalid or missing dates are excluded."
    )
    return CardResult(card, None, figure, audit, calculation)


def _build_scatter(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    assert card.x is not None and card.y is not None
    audit = pd.DataFrame(
        {
            card.x: pd.to_numeric(df[card.x], errors="coerce"),
            card.y: pd.to_numeric(df[card.y], errors="coerce"),
        }
    ).dropna()
    original_points = len(audit)
    if original_points > MAX_SCATTER_POINTS:
        audit = audit.sample(n=MAX_SCATTER_POINTS, random_state=42).sort_index()
    audit = audit.reset_index(drop=True)
    figure = go.Figure(
        go.Scatter(
            x=audit[card.x],
            y=audit[card.y],
            mode="markers",
            marker={"color": "#15803d", "size": 7, "opacity": 0.68},
            hovertemplate="x=%{x:,.4g}<br>y=%{y:,.4g}<extra></extra>",
        )
    )
    _style_figure(figure, card, x_title=card.x, y_title=card.y)
    sample_note = (
        f"a deterministic sample of {len(audit):,} from {original_points:,} complete rows"
        if original_points > MAX_SCATTER_POINTS
        else f"all {len(audit):,} complete rows"
    )
    calculation = (
        f"Numeric {card.x} plotted against numeric {card.y} using {sample_note}; "
        "rows missing either value are excluded."
    )
    return CardResult(card, None, figure, audit, calculation)


def _build_distribution(df: pd.DataFrame, card: DashboardCard) -> CardResult:
    assert card.column is not None
    values = pd.to_numeric(df[card.column], errors="coerce").dropna()
    if values.empty:
        audit = pd.DataFrame(columns=["bin_start", "bin_end", "bin", "count"])
    else:
        buckets = pd.cut(
            values,
            bins=card.bins,
            include_lowest=True,
            duplicates="drop",
        )
        counts = buckets.value_counts(sort=False)
        audit = pd.DataFrame(
            {
                "bin_start": [float(interval.left) for interval in counts.index],
                "bin_end": [float(interval.right) for interval in counts.index],
                "bin": counts.index.astype(str),
                "count": counts.to_numpy(),
            }
        )
    figure = go.Figure(
        go.Bar(
            x=audit["bin"],
            y=audit["count"],
            marker_color="#15803d",
            hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
        )
    )
    _style_figure(figure, card, x_title=card.column, y_title="Row count")
    calculation = (
        f"{len(values):,} valid numeric {card.column} values grouped into "
        f"{len(audit):,} equal-width bins; non-numeric and missing values are excluded."
    )
    return CardResult(card, None, figure, audit, calculation)


def _run_group_aggregation(grouped: object, metric: Metric) -> pd.DataFrame:
    operations = {
        "sum": grouped.sum,
        "mean": grouped.mean,
        "median": grouped.median,
    }
    return operations[metric]()


def _style_figure(
    figure: go.Figure,
    card: DashboardCard,
    *,
    x_title: str,
    y_title: str,
) -> None:
    figure.update_layout(
        title={"text": ""},
        height=370,
        margin={"l": 48, "r": 18, "t": 18, "b": 60},
        paper_bgcolor="#fffdf6",
        plot_bgcolor="#fffdf6",
        font={"color": "#23392b"},
        xaxis={
            "title": escape(x_title),
            "gridcolor": "rgba(20,83,45,.12)",
            "automargin": True,
        },
        yaxis={
            "title": escape(y_title),
            "gridcolor": "rgba(20,83,45,.16)",
            "automargin": True,
        },
        hoverlabel={"bgcolor": "#16281e", "font_color": "#f7f3e7"},
        showlegend=False,
    )


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DashboardStudioError("Dashboard data must be a pandas DataFrame.")
    names = list(df.columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise DashboardStudioError("Dashboard columns must have non-empty text names.")
    if len(names) != len(set(names)):
        raise DashboardStudioError("Dashboard Studio requires unique column names.")


def _validate_card(df: pd.DataFrame, card: DashboardCard) -> None:
    _validate_id(card.card_id, "Card ID")
    _validate_text(card.title, "Card title", 100)
    if card.kind not in CARD_KINDS:
        raise DashboardStudioError(f"Unsupported card type: {card.kind}.")
    if card.metric not in METRICS or card.aggregation not in METRICS:
        raise DashboardStudioError("Unsupported dashboard calculation.")
    if card.date_grain not in DATE_GRAINS:
        raise DashboardStudioError("Unsupported date grain.")
    if not isinstance(card.top_n, int) or isinstance(card.top_n, bool):
        raise DashboardStudioError("Bar chart group limits must be whole numbers.")
    if not 3 <= card.top_n <= 50:
        raise DashboardStudioError("Bar charts must show between 3 and 50 groups.")
    if not isinstance(card.bins, int) or isinstance(card.bins, bool):
        raise DashboardStudioError("Distribution bin counts must be whole numbers.")
    if not 3 <= card.bins <= 50:
        raise DashboardStudioError("Distributions must use between 3 and 50 bins.")

    if card.kind == "kpi":
        if card.metric != "row_count":
            _require_column(df, card.column)
        if card.metric in {"sum", "mean", "median"}:
            _require_numeric(df, card.column)
        return
    if card.kind in {"bar", "pie"}:
        if card.kind == "pie" and card.aggregation in {"mean", "median"}:
            raise DashboardStudioError(
                "Pie charts need additive values: use row count, total, or distinct count."
            )
        _require_column(df, card.x)
        if card.aggregation != "row_count":
            _require_column(df, card.y)
        if card.aggregation in {"sum", "mean", "median"}:
            _require_numeric(df, card.y)
        return
    if card.kind == "line":
        _require_column(df, card.x)
        assert card.x is not None
        if _parse_dates(df[card.x]).notna().sum() == 0:
            raise DashboardStudioError(f"{card.x} has no valid dates for a line chart.")
        if card.aggregation != "row_count":
            _require_column(df, card.y)
        if card.aggregation in {"sum", "mean", "median"}:
            _require_numeric(df, card.y)
        return
    if card.kind == "scatter":
        _require_numeric(df, card.x)
        _require_numeric(df, card.y)
        if card.x == card.y:
            raise DashboardStudioError("Scatter charts need two different columns.")
        return
    _require_numeric(df, card.column)


def _validate_filter(df: pd.DataFrame, item: DashboardFilter) -> None:
    _validate_id(item.filter_id, "Filter ID")
    _require_column(df, item.column)
    if item.kind not in {"values", "range", "date_range"}:
        raise DashboardStudioError("Unsupported global filter type.")
    expected = filter_kind_for_column(df, item.column)
    if item.kind != expected:
        raise DashboardStudioError(
            f"{item.column} requires a {expected.replace('_', ' ')} filter."
        )
    if item.kind == "values":
        if not isinstance(item.values, tuple):
            raise DashboardStudioError("Filter values must be a list of text values.")
        if len(item.values) > 200:
            raise DashboardStudioError("A value filter can select at most 200 values.")
        for value in item.values:
            _validate_text(value, "Filter value", 200, allow_empty=True)
    elif item.kind == "range":
        if item.minimum is None or item.maximum is None:
            raise DashboardStudioError("Numeric filters need a minimum and maximum.")
        try:
            minimum = float(item.minimum)
            maximum = float(item.maximum)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DashboardStudioError("The numeric filter range is invalid.") from exc
        if not isfinite(minimum) or not isfinite(maximum) or minimum > maximum:
            raise DashboardStudioError("The numeric filter range is invalid.")
    else:
        if not isinstance(item.start, str) or not isinstance(item.end, str):
            raise DashboardStudioError("Date filters need valid start and end dates.")
        try:
            start = pd.Timestamp(item.start)
            end = pd.Timestamp(item.end)
        except (TypeError, ValueError) as exc:
            raise DashboardStudioError("Date filters need valid start and end dates.") from exc
        if pd.isna(start) or pd.isna(end) or start > end:
            raise DashboardStudioError("The date filter range is invalid.")


def _require_column(df: pd.DataFrame, column: str | None) -> None:
    if column is None or column not in df.columns:
        raise DashboardStudioError(f"Dashboard column {column!r} is unavailable.")


def _require_numeric(df: pd.DataFrame, column: str | None) -> None:
    _require_column(df, column)
    assert column is not None
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        raise DashboardStudioError(f"{column} has no valid numeric values.")


def _validate_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise DashboardStudioError(
            f"{label} must use 1–64 letters, numbers, hyphens, or underscores."
        )


def _validate_text(
    value: str,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str):
        raise DashboardStudioError(f"{label} must be text.")
    if (not allow_empty and not value.strip()) or len(value) > maximum:
        raise DashboardStudioError(f"{label} must contain 1–{maximum} characters.")
    if any(ord(character) < 32 and character not in "\t" for character in value):
        raise DashboardStudioError(f"{label} contains unsupported control characters.")


def _card_from_payload(payload: object) -> DashboardCard:
    if not isinstance(payload, dict):
        raise TypeError("Card must be an object")
    allowed = {
        "card_id",
        "kind",
        "title",
        "metric",
        "column",
        "x",
        "y",
        "aggregation",
        "date_grain",
        "top_n",
        "bins",
    }
    if set(payload) - allowed:
        raise ValueError("Unknown card field")
    return DashboardCard(**payload)


def _filter_from_payload(payload: object) -> DashboardFilter:
    if not isinstance(payload, dict):
        raise TypeError("Filter must be an object")
    allowed = {
        "filter_id",
        "column",
        "kind",
        "values",
        "minimum",
        "maximum",
        "start",
        "end",
    }
    if set(payload) - allowed:
        raise ValueError("Unknown filter field")
    normalized = dict(payload)
    values = normalized.get("values", [])
    if not isinstance(values, list):
        raise TypeError("Filter values must be an array")
    normalized["values"] = tuple(values)
    return DashboardFilter(**normalized)


def _category_labels(series: pd.Series) -> pd.Series:
    labels = series.astype("string").str.strip().replace("", pd.NA)
    return labels.fillna(MISSING_LABEL)


def _parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if (
        not non_null.empty
        and len(non_null) == int(series.notna().sum())
        and bool(non_null.between(1900, 2100).all())
        and bool((non_null % 1 == 0).all())
    ):
        return pd.to_datetime(
            numeric.astype("Int64").astype("string"), errors="coerce", format="%Y"
        )
    parsed = pd.to_datetime(series, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_convert(None)


def _group_dates(dates: pd.Series, grain: DateGrain) -> pd.Series:
    frequencies = {
        "day": "D",
        "week": "W-SUN",
        "month": "M",
        "quarter": "Q",
        "year": "Y",
    }
    return dates.dt.to_period(frequencies[grain]).dt.start_time


def _filter_description(item: DashboardFilter) -> str:
    if item.kind == "values":
        return (
            f"{item.column}: {', '.join(item.values)}"
            if item.values
            else f"{item.column}: all values"
        )
    if item.kind == "range":
        return f"{item.column}: {item.minimum:g} to {item.maximum:g}"
    return f"{item.column}: {item.start} to {item.end}"
