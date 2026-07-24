"""Zero-cost export pipeline: push the working dataset to user-owned destinations.

Design rules, in line with the rest of Open Data Scientist:

- No paid API and no AI model. Destinations are services the user already
  owns (an Airtable base on the free plan, or any webhook endpoint), reached
  with credentials the user supplies at run time.
- Credentials are never serialized. A pipeline configuration stores the
  *shape* of the destination (base, table, URL, header name); tokens and
  secret values are provided per run through the UI or environment variables.
- Everything is auditable: the exact first-batch payload can be previewed
  before anything is sent, and every push returns a full report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from math import isfinite
import re
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import pandas as pd
import requests


class PipelineError(ValueError):
    """Raised when a pipeline configuration or push cannot proceed safely."""


AIRTABLE_API_URL = "https://api.airtable.com/v0"
AIRTABLE_BATCH_SIZE = 10  # hard Airtable API limit per create request
AIRTABLE_RATE_DELAY_SECONDS = 0.25  # keeps well under the 5 requests/second cap
AIRTABLE_FREE_RECORD_LIMIT = 1_000  # free-plan records per base at time of writing
WEBHOOK_MAX_BATCH_SIZE = 1_000
MAX_EXPORT_RECORDS = 250_000  # mirrors DatasetLimits.max_rows

_BASE_ID_PATTERN = re.compile(r"^app[a-zA-Z0-9]{14}$")

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class AirtableDestination:
    """An existing Airtable table the user owns; token supplied at run time."""

    base_id: str
    table: str
    typecast: bool = True


@dataclass(frozen=True)
class WebhookDestination:
    """Any HTTP endpoint the user controls; secret value supplied at run time."""

    url: str
    batch_size: int = 500
    secret_header: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Portable, credential-free description of one export pipeline."""

    name: str
    kind: str  # "airtable" | "webhook"
    airtable: AirtableDestination | None = None
    webhook: WebhookDestination | None = None
    row_limit: int | None = None


@dataclass(frozen=True)
class PushReport:
    """The complete outcome of one pipeline run."""

    destination: str
    total_records: int
    sent_records: int
    batches_sent: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures and self.sent_records == self.total_records


# ── Record preparation ────────────────────────────────────────────────────────

def dataframe_to_records(
    dataframe: pd.DataFrame,
    *,
    row_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert the working dataset into JSON-safe records.

    Missing values are omitted from each record, timestamps become ISO-8601
    strings, and NumPy scalars become plain Python values.
    """
    columns = [str(column) for column in dataframe.columns]
    if len(columns) != len(set(columns)):
        raise PipelineError("Pipelines need unique column names. Rename duplicates first.")
    if row_limit is not None:
        if row_limit < 1:
            raise PipelineError("The pipeline row limit must be at least 1.")
        dataframe = dataframe.head(row_limit)
    if len(dataframe) > MAX_EXPORT_RECORDS:
        raise PipelineError(
            f"Pipelines are limited to {MAX_EXPORT_RECORDS:,} records per run."
        )

    records: list[dict[str, Any]] = []
    for row in dataframe.itertuples(index=False, name=None):
        record = {}
        for column, value in zip(columns, row):
            safe = _json_safe(value)
            if safe is not None:
                record[column] = safe
        records.append(record)
    return records


def _json_safe(value: Any) -> Any:
    """Convert one cell to a JSON-serializable value, or None when missing."""
    if value is None:
        return None
    if isinstance(value, float) and not isfinite(value):
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, datetime, date)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)


def _batches(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


# ── Payload preview (audit before send) ───────────────────────────────────────

def preview_payload(records: list[dict[str, Any]], config: PipelineConfig) -> str:
    """Return the exact JSON body of the first batch, for review before sending."""
    _validate_config_shape(config)
    if not records:
        return json.dumps({"records": []}, indent=2)
    if config.kind == "airtable":
        assert config.airtable is not None
        batch = records[:AIRTABLE_BATCH_SIZE]
        payload: dict[str, Any] = {
            "records": [{"fields": record} for record in batch],
            "typecast": config.airtable.typecast,
        }
    else:
        assert config.webhook is not None
        batch = records[: config.webhook.batch_size]
        payload = _webhook_body(config.name, batch, 0, 1)
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ── Airtable destination ──────────────────────────────────────────────────────

def check_airtable_connection(
    destination: AirtableDestination,
    token: str,
    *,
    api_url: str = AIRTABLE_API_URL,
    session: requests.Session | None = None,
) -> str:
    """Verify token, base, and table with a one-record read. Returns a message."""
    _validate_airtable(destination, token)
    http = session or requests.Session()
    try:
        response = http.get(
            f"{api_url}/{destination.base_id}/{quote(destination.table, safe='')}",
            headers={"Authorization": f"Bearer {token}"},
            params={"maxRecords": 1},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise PipelineError(f"Could not reach Airtable: {exc}") from exc
    if response.status_code == 200:
        return f"Connected. Table {destination.table!r} is reachable and writable tokens can push to it."
    raise PipelineError(_airtable_error_message(response))


def push_to_airtable(
    records: list[dict[str, Any]],
    destination: AirtableDestination,
    token: str,
    *,
    api_url: str = AIRTABLE_API_URL,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
) -> PushReport:
    """Append records to an existing Airtable table in API-sized batches."""
    _validate_airtable(destination, token)
    http = session or requests.Session()
    url = f"{api_url}/{destination.base_id}/{quote(destination.table, safe='')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    batches = _batches(records, AIRTABLE_BATCH_SIZE)
    sent_records = 0
    batches_sent = 0
    failures: list[str] = []
    for index, batch in enumerate(batches):
        body = {
            "records": [{"fields": record} for record in batch],
            "typecast": destination.typecast,
        }
        response = _post_with_one_retry(http, url, body, headers, sleep)
        if response is None:
            failures.append(f"Batch {index + 1}: Airtable could not be reached.")
        elif response.status_code == 200:
            sent_records += len(batch)
            batches_sent += 1
        elif response.status_code in {401, 403, 404}:
            # Nothing later in the run can succeed with bad credentials or target.
            raise PipelineError(_airtable_error_message(response))
        else:
            failures.append(f"Batch {index + 1}: {_airtable_error_message(response)}")
        if progress is not None:
            progress(min((index + 1) * AIRTABLE_BATCH_SIZE, len(records)), len(records))
        if index + 1 < len(batches):
            sleep(AIRTABLE_RATE_DELAY_SECONDS)

    return PushReport(
        destination=f"Airtable · {destination.base_id}/{destination.table}",
        total_records=len(records),
        sent_records=sent_records,
        batches_sent=batches_sent,
        failures=tuple(failures),
    )


def _post_with_one_retry(
    http: requests.Session,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    sleep: Callable[[float], None],
) -> requests.Response | None:
    for attempt in (1, 2):
        try:
            response = http.post(url, json=body, headers=headers, timeout=30)
        except requests.RequestException:
            return None
        if response.status_code == 429 and attempt == 1:
            sleep(2.0)
            continue
        return response
    return response


def _airtable_error_message(response: requests.Response) -> str:
    detail = ""
    try:
        payload = response.json()
        error = payload.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("type") or "")
        elif isinstance(error, str):
            detail = error
    except ValueError:
        detail = ""
    hints = {
        401: "The personal access token was rejected. Create one at airtable.com/create/tokens with data.records:write scope.",
        403: "The token does not have access to this base. Add the base to the token's scope.",
        404: "The base ID or table was not found. Copy the base ID (app…) from the base URL and check the table name.",
        422: "Airtable rejected the field values. Keep typecast enabled or align the table's fields with the dataset columns.",
        429: "Airtable rate limit reached. Try again in about 30 seconds.",
    }
    hint = hints.get(response.status_code, "Airtable returned an unexpected response.")
    return f"HTTP {response.status_code}: {hint}" + (f" ({detail})" if detail else "")


def _validate_airtable(destination: AirtableDestination, token: str) -> None:
    if not token or not token.strip():
        raise PipelineError("Provide an Airtable personal access token for this run.")
    if not _BASE_ID_PATTERN.fullmatch(destination.base_id.strip()):
        raise PipelineError(
            "The Airtable base ID must look like appXXXXXXXXXXXXXX (17 characters, from the base URL)."
        )
    if not destination.table.strip():
        raise PipelineError("Provide the Airtable table name or table ID to append to.")


# ── Webhook destination ───────────────────────────────────────────────────────

def push_to_webhook(
    records: list[dict[str, Any]],
    destination: WebhookDestination,
    *,
    secret_value: str | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
    pipeline_name: str = "ods-pipeline",
) -> PushReport:
    """POST records as JSON batches to any endpoint the user controls."""
    _validate_webhook(destination)
    http = session or requests.Session()
    headers = {"Content-Type": "application/json"}
    if destination.secret_header:
        if not secret_value:
            raise PipelineError(
                f"This pipeline expects a value for the {destination.secret_header!r} header at run time."
            )
        headers[destination.secret_header] = secret_value

    batches = _batches(records, destination.batch_size)
    sent_records = 0
    batches_sent = 0
    failures: list[str] = []
    for index, batch in enumerate(batches):
        body = _webhook_body(pipeline_name, batch, index, len(batches))
        try:
            response = http.post(destination.url, json=body, headers=headers, timeout=30)
        except requests.RequestException as exc:
            if index == 0:
                raise PipelineError(f"Could not reach the webhook endpoint: {exc}") from exc
            failures.append(f"Batch {index + 1}: endpoint unreachable ({exc}).")
            continue
        if 200 <= response.status_code < 300:
            sent_records += len(batch)
            batches_sent += 1
        else:
            failures.append(
                f"Batch {index + 1}: the endpoint answered HTTP {response.status_code}."
            )
        if progress is not None:
            progress(min((index + 1) * destination.batch_size, len(records)), len(records))
        if index + 1 < len(batches):
            sleep(0.05)

    return PushReport(
        destination=f"Webhook · {destination.url}",
        total_records=len(records),
        sent_records=sent_records,
        batches_sent=batches_sent,
        failures=tuple(failures),
    )


def _webhook_body(
    pipeline_name: str,
    batch: list[dict[str, Any]],
    batch_index: int,
    batch_count: int,
) -> dict[str, Any]:
    return {
        "source": "open-data-scientist",
        "pipeline": pipeline_name,
        "batch_index": batch_index,
        "batch_count": batch_count,
        "records": batch,
    }


def _validate_webhook(destination: WebhookDestination) -> None:
    parts = urlsplit(destination.url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise PipelineError("The webhook URL must start with http:// or https://.")
    if parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1"}:
        raise PipelineError(
            "Plain http:// webhooks are allowed only for localhost testing. Use https:// otherwise."
        )
    if not 1 <= destination.batch_size <= WEBHOOK_MAX_BATCH_SIZE:
        raise PipelineError(
            f"Webhook batch size must be between 1 and {WEBHOOK_MAX_BATCH_SIZE:,} records."
        )
    if destination.secret_header is not None and not destination.secret_header.strip():
        raise PipelineError("The secret header name cannot be blank.")


# ── Pipeline configuration (credential-free JSON) ─────────────────────────────

def pipeline_config_to_json(config: PipelineConfig) -> str:
    """Serialize a pipeline without any tokens or secret values."""
    _validate_config_shape(config)
    payload: dict[str, Any] = {
        "format": "ods-pipeline",
        "version": 1,
        "name": config.name,
        "kind": config.kind,
        "row_limit": config.row_limit,
    }
    if config.airtable is not None:
        payload["airtable"] = asdict(config.airtable)
    if config.webhook is not None:
        payload["webhook"] = asdict(config.webhook)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def pipeline_config_from_json(text: str) -> PipelineConfig:
    """Load and strictly validate an untrusted pipeline configuration."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PipelineError("This is not valid pipeline JSON.") from exc
    if not isinstance(payload, dict):
        raise PipelineError("Pipeline JSON must contain one object.")
    if payload.get("format") != "ods-pipeline" or payload.get("version") != 1:
        raise PipelineError("Only ODS pipeline format version 1 is supported.")
    if set(payload) - {"format", "version", "name", "kind", "row_limit", "airtable", "webhook"}:
        raise PipelineError("The pipeline JSON contains unsupported fields.")
    try:
        airtable = (
            AirtableDestination(**payload["airtable"]) if "airtable" in payload else None
        )
        webhook = (
            WebhookDestination(**payload["webhook"]) if "webhook" in payload else None
        )
        config = PipelineConfig(
            name=str(payload.get("name", "")),
            kind=str(payload.get("kind", "")),
            airtable=airtable,
            webhook=webhook,
            row_limit=payload.get("row_limit"),
        )
    except TypeError as exc:
        raise PipelineError("The pipeline configuration has invalid fields.") from exc
    _validate_config_shape(config)
    return config


def _validate_config_shape(config: PipelineConfig) -> None:
    if not config.name.strip() or len(config.name) > 80:
        raise PipelineError("Pipeline names must contain 1–80 characters.")
    if config.kind == "airtable":
        if config.airtable is None:
            raise PipelineError("Airtable pipelines need an airtable destination block.")
    elif config.kind == "webhook":
        if config.webhook is None:
            raise PipelineError("Webhook pipelines need a webhook destination block.")
        _validate_webhook(config.webhook)
    else:
        raise PipelineError("Pipeline kind must be 'airtable' or 'webhook'.")
    if config.row_limit is not None and (
        not isinstance(config.row_limit, int)
        or isinstance(config.row_limit, bool)
        or config.row_limit < 1
    ):
        raise PipelineError("The pipeline row limit must be a whole number of at least 1.")
