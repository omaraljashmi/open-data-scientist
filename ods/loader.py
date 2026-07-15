"""Safe, deterministic CSV and Excel loading with explicit resource limits."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


class DatasetLoadError(ValueError):
    """Raised when an uploaded dataset cannot be parsed."""


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass(frozen=True)
class DatasetLimits:
    """Resource boundaries applied before a dataset reaches the analysis tools."""

    max_upload_bytes: int = 25 * 1024 * 1024
    max_rows: int = 250_000
    max_columns: int = 500
    max_excel_uncompressed_bytes: int = 250 * 1024 * 1024
    max_excel_archive_members: int = 10_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")


DEFAULT_LIMITS = DatasetLimits()


def load_dataset(
    file_name: str,
    content: bytes,
    *,
    limits: DatasetLimits = DEFAULT_LIMITS,
) -> pd.DataFrame:
    """Load a CSV or Excel upload into a DataFrame.

    CSV parsing tries common encodings and lets pandas infer the delimiter.
    Excel workbooks load their first worksheet. All inputs are bounded by a
    documented size, shape, and XLSX expansion policy before analysis.
    """
    safe_name = Path(str(file_name)).name or "dataset"
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DatasetLoadError(f"Unsupported file type. Use one of: {supported}.")
    if not content:
        raise DatasetLoadError("The uploaded file is empty.")
    if len(content) > limits.max_upload_bytes:
        raise DatasetLoadError(
            f"{safe_name} is {_format_bytes(len(content))}; the release-candidate "
            f"limit is {_format_bytes(limits.max_upload_bytes)} per file. "
            "Reduce the file or run ODS locally with a reviewed higher limit."
        )

    try:
        if suffix == ".csv":
            dataframe = _load_csv(content)
        else:
            if suffix == ".xlsx":
                _validate_xlsx_archive(content, limits)
            engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
            dataframe = pd.read_excel(
                BytesIO(content),
                sheet_name=0,
                engine=engine,
            )
    except DatasetLoadError:
        raise
    except MemoryError as exc:
        raise DatasetLoadError(
            "The dataset exceeded available memory while being read. "
            "Reduce its size or run ODS locally with more memory."
        ) from exc
    except ImportError as exc:
        raise DatasetLoadError(
            f"The required reader for {suffix} files is unavailable. "
            "Reinstall the release dependencies and try again."
        ) from exc
    except Exception as exc:  # pandas emits several parser-specific exceptions
        raise DatasetLoadError(
            f"Could not read {safe_name}. The file may be damaged, password-protected, "
            f"or not actually a {suffix} file."
        ) from exc

    return _validate_dataframe(dataframe, limits)


def _load_csv(content: bytes) -> pd.DataFrame:
    if b"\x00" in content[:8192]:
        raise DatasetLoadError(
            "The CSV appears to contain binary or UTF-16 data. Save it as UTF-8 CSV and try again."
        )
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


def _validate_xlsx_archive(content: bytes, limits: DatasetLimits) -> None:
    """Reject malformed, encrypted, or unexpectedly expanded XLSX archives."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if not members or not any(
                member.filename == "[Content_Types].xml" for member in members
            ):
                raise DatasetLoadError(
                    "The .xlsx file is not a valid Excel workbook archive."
                )
            if len(members) > limits.max_excel_archive_members:
                raise DatasetLoadError(
                    "The workbook contains too many internal files to process safely."
                )
            if any(member.flag_bits & 0x1 for member in members):
                raise DatasetLoadError(
                    "Password-protected Excel workbooks are not supported."
                )
            expanded_size = sum(member.file_size for member in members)
            if expanded_size > limits.max_excel_uncompressed_bytes:
                raise DatasetLoadError(
                    "The workbook expands beyond the safe processing limit "
                    f"({_format_bytes(expanded_size)} uncompressed)."
                )
    except BadZipFile as exc:
        raise DatasetLoadError(
            "The .xlsx file is damaged, encrypted, or not an Excel workbook."
        ) from exc


def _validate_dataframe(
    dataframe: pd.DataFrame,
    limits: DatasetLimits,
) -> pd.DataFrame:
    rows, columns = dataframe.shape
    if columns == 0:
        raise DatasetLoadError("The dataset does not contain any columns.")
    if rows == 0:
        raise DatasetLoadError(
            "The dataset contains a header but no data rows. Add at least one row and try again."
        )
    if rows > limits.max_rows:
        raise DatasetLoadError(
            f"The dataset has {rows:,} rows; the release-candidate limit is "
            f"{limits.max_rows:,}. Filter or aggregate the file before uploading it."
        )
    if columns > limits.max_columns:
        raise DatasetLoadError(
            f"The dataset has {columns:,} columns; the release-candidate limit is "
            f"{limits.max_columns:,}. Keep only the fields needed for analysis."
        )

    normalized_columns = [str(column).strip() for column in dataframe.columns]
    if any(not column for column in normalized_columns):
        raise DatasetLoadError(
            "Every column needs a non-empty header before ODS can analyze it."
        )
    if len(normalized_columns) != len(set(normalized_columns)):
        raise DatasetLoadError(
            "Column headers must be unique after converting them to text. Rename duplicate headers and try again."
        )

    result = dataframe.copy(deep=False)
    result.columns = normalized_columns
    return result


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"
