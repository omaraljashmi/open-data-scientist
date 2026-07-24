"""Optional AI chart advisor — zero-cost, bring-your-own free endpoint.

Design rules, in line with the rest of Open Data Scientist:

- Off by default and never required: the deterministic recommendations in
  ``suggest_dashboard`` remain the primary path.
- Zero cost: the user supplies their own endpoint — a free-tier key
  (e.g. Google Gemini or Groq) or a local server (Ollama). Anything that
  speaks the OpenAI-compatible ``/chat/completions`` wire format works.
  Anthropic/Claude is deliberately not a preset: it has no free tier.
- Privacy: the advisor receives **dataset metadata only** — column names,
  inferred roles, display formats, unique and missing counts, and the row
  count. Cell values, example values, and statistics of values never leave
  the machine.
- Trust boundary: the model only *picks chart parameters*. Every suggestion
  is validated against the real columns and roles, then rendered by the same
  local, auditable calculations as any other ODS chart. Invalid suggestions
  are dropped, never guessed at.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import requests

from .dashboard import (
    Aggregation,
    ChartSuggestion,
    ColumnSemantic,
    DateGrain,
    infer_column_semantics,
)


class AdvisorError(ValueError):
    """Raised when the advisor endpoint or its answer cannot be used safely."""


MAX_ADVISOR_CHARTS = 4
_ADVISOR_TIMEOUT_SECONDS = 60

AGGREGATIONS: tuple[Aggregation, ...] = ("mean", "median", "sum", "count", "min", "max")
DATE_GRAINS: tuple[DateGrain, ...] = ("day", "week", "month", "quarter", "year")

ADVISOR_PRESETS: dict[str, dict[str, Any]] = {
    "Google Gemini (free tier)": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "needs_key": True,
    },
    "Groq (free tier)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "needs_key": True,
    },
    "Local Ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "needs_key": False,
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "model": "",
        "needs_key": False,
    },
}


# ── Metadata brief (what the advisor is allowed to see) ───────────────────────

def dataset_brief(
    dataframe: pd.DataFrame,
    semantics: tuple[ColumnSemantic, ...] | None = None,
) -> dict[str, Any]:
    """Describe the dataset without exposing any cell values.

    The brief carries column names, inferred roles and formats, unique and
    missing counts, and the row count — nothing else.
    """
    semantics = semantics if semantics is not None else infer_column_semantics(dataframe)
    row_count = len(dataframe)
    columns = []
    for semantic in semantics:
        series = dataframe[semantic.column]
        missing = int(series.isna().sum())
        columns.append(
            {
                "name": semantic.column,
                "role": semantic.role,
                "format": semantic.display_format,
                "unique_values": int(series.nunique(dropna=True)),
                "missing_percent": round(missing / row_count * 100, 2) if row_count else 0.0,
            }
        )
    return {"rows": row_count, "columns": columns}


def build_advisor_messages(brief: dict[str, Any], intent: str) -> list[dict[str, str]]:
    """Build the chat messages sent to the user's endpoint (metadata only)."""
    instructions = {
        "task": (
            "Recommend up to "
            f"{MAX_ADVISOR_CHARTS} charts for this dataset that best serve the analysis goal. "
            "You only see metadata; pick chart parameters, nothing else."
        ),
        "analysis_goal": intent,
        "dataset": brief,
        "chart_kinds": {
            "category_count": "x = any column; counts rows per value of x.",
            "histogram": "x = a column with role 'numeric'; distribution of x.",
            "time_series": (
                "x = a column with role 'datetime'; optional y = a 'numeric' column with an "
                "aggregation and date_grain; y omitted means row counts over time."
            ),
            "category_aggregate": (
                "x = a 'categorical' column, y = a 'numeric' column, plus an aggregation."
            ),
            "scatter": "x and y = two different 'numeric' columns.",
        },
        "aggregations": list(AGGREGATIONS),
        "date_grains": list(DATE_GRAINS),
        "response_format": {
            "charts": [
                {
                    "kind": "one of the chart_kinds keys",
                    "title": "short chart title",
                    "explanation": "one sentence on why this chart serves the goal",
                    "x": "column name",
                    "y": "column name or null",
                    "aggregation": "one of aggregations (when applicable)",
                    "date_grain": "one of date_grains (time_series only)",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a data-visualization advisor inside Open Data Scientist. "
                "Respond with a single JSON object matching response_format. "
                "No prose, no markdown fences."
            ),
        },
        {"role": "user", "content": json.dumps(instructions, ensure_ascii=False)},
    ]


# ── Endpoint call ─────────────────────────────────────────────────────────────

def request_chart_advice(
    dataframe: pd.DataFrame,
    *,
    intent: str,
    base_url: str,
    model: str,
    api_key: str | None = None,
    session: requests.Session | None = None,
    timeout: float = _ADVISOR_TIMEOUT_SECONDS,
) -> tuple[ChartSuggestion, ...]:
    """Ask the user's endpoint for chart picks and validate them locally."""
    _validate_endpoint(base_url, model)
    semantics = infer_column_semantics(dataframe)
    messages = build_advisor_messages(dataset_brief(dataframe, semantics), intent)

    http = session or requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = http.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 700,
            },
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise AdvisorError(f"Could not reach the advisor endpoint: {exc}") from exc

    if response.status_code in {401, 403}:
        raise AdvisorError(
            "The advisor endpoint rejected the API key. Check the key and its free-tier access."
        )
    if response.status_code == 404:
        raise AdvisorError(
            "The advisor endpoint returned 404. Check the base URL and the model name."
        )
    if response.status_code == 429:
        raise AdvisorError(
            "The advisor endpoint is rate limited (free tiers throttle). Wait a moment and retry."
        )
    if response.status_code != 200:
        raise AdvisorError(f"The advisor endpoint answered HTTP {response.status_code}.")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AdvisorError(
            "The advisor endpoint did not return an OpenAI-compatible chat response."
        ) from exc

    suggestions = _suggestions_from_content(content, dataframe, semantics)
    if not suggestions:
        raise AdvisorError(
            "The advisor returned no usable chart suggestions for these columns. "
            "Try again, or use the deterministic recommendations above."
        )
    return suggestions


def _validate_endpoint(base_url: str, model: str) -> None:
    parts = urlsplit(base_url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise AdvisorError("The advisor base URL must start with http:// or https://.")
    if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1"}:
        raise AdvisorError(
            "Plain http:// advisor endpoints are allowed only for localhost (e.g. Ollama)."
        )
    if not model.strip():
        raise AdvisorError("Provide the model name for the advisor endpoint.")


# ── Response validation (the advisor never bypasses local rules) ──────────────

def _suggestions_from_content(
    content: str,
    dataframe: pd.DataFrame,
    semantics: tuple[ColumnSemantic, ...],
) -> tuple[ChartSuggestion, ...]:
    payload = _parse_json_object(content)
    charts = payload.get("charts")
    if not isinstance(charts, list):
        raise AdvisorError("The advisor answer did not contain a charts list.")

    roles = {semantic.column: semantic.role for semantic in semantics}
    available = {str(column) for column in dataframe.columns}
    suggestions: list[ChartSuggestion] = []
    for entry in charts:
        suggestion = _validated_suggestion(entry, roles, available)
        if suggestion is not None:
            suggestions.append(suggestion)
        if len(suggestions) >= MAX_ADVISOR_CHARTS:
            break
    return tuple(suggestions)


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    if "{" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise AdvisorError("The advisor did not answer with valid JSON.") from exc
    if not isinstance(payload, dict):
        raise AdvisorError("The advisor answer was not a JSON object.")
    return payload


def _validated_suggestion(
    entry: Any,
    roles: dict[str, str],
    available: set[str],
) -> ChartSuggestion | None:
    if not isinstance(entry, dict):
        return None
    kind = entry.get("kind")
    x = entry.get("x")
    y = entry.get("y")
    aggregation = entry.get("aggregation") if entry.get("aggregation") in AGGREGATIONS else "mean"
    date_grain = entry.get("date_grain") if entry.get("date_grain") in DATE_GRAINS else "month"
    title = _clean_text(entry.get("title"), fallback=f"{kind} chart", limit=100)
    explanation = _clean_text(
        entry.get("explanation"), fallback="Suggested by the AI chart advisor.", limit=240
    )

    if not isinstance(x, str) or x not in available:
        return None
    if kind == "category_count":
        return ChartSuggestion("category_count", title, explanation, x, "count", confidence=0.75)
    if kind == "histogram":
        if roles.get(x) != "numeric":
            return None
        return ChartSuggestion("histogram", title, explanation, x, "count", confidence=0.75)
    if kind == "time_series":
        if roles.get(x) != "datetime":
            return None
        if y is not None and (not isinstance(y, str) or roles.get(y) != "numeric"):
            y = None
        return ChartSuggestion(
            "time_series", title, explanation, x, y, aggregation, date_grain, 0.75
        )
    if kind == "category_aggregate":
        if roles.get(x) != "categorical":
            return None
        if not isinstance(y, str) or roles.get(y) != "numeric":
            return None
        return ChartSuggestion(
            "category_aggregate", title, explanation, x, y, aggregation, confidence=0.75
        )
    if kind == "scatter":
        if roles.get(x) != "numeric":
            return None
        if not isinstance(y, str) or roles.get(y) != "numeric" or y == x:
            return None
        return ChartSuggestion("scatter", title, explanation, x, y, confidence=0.75)
    return None


def _clean_text(value: Any, *, fallback: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return " ".join(value.split())[:limit]
