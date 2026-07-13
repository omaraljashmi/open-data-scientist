"""Safe, deterministic CSV and Excel loading."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


class DatasetLoadError(ValueError):
    """Raised when an uploaded dataset cannot be parsed."""


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(file_name: str, content: bytes) -> pd.DataFrame:
    """Load a CSV or Excel upload into a DataFrame.

    CSV parsing tries common encodings and lets pandas infer the delimiter.
    Excel workbooks currently load their first worksheet.
    """
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetLoadError(f"Unsupported file type. Use one of: {supported}.")
    if not content:
        raise DatasetLoadError("The uploaded file is empty.")

    try:
        if suffix == ".csv":
            return _load_csv(content)
        return pd.read_excel(BytesIO(content), sheet_name=0)
    except DatasetLoadError:
        raise
    except Exception as exc:  # pandas emits several parser-specific exceptions
        raise DatasetLoadError(f"Could not read {file_name}: {exc}") from exc


def _load_csv(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise DatasetLoadError(f"Could not parse the CSV file: {last_error}")

