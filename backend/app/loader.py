"""Reading a client export, from a directory or from uploaded file objects.

Requirement 4 says a new week must load without a developer, so validation
lives here and every failure is a sentence a contract manager could read.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Mapping

import pandas as pd

from .config import DATA_DIR, EXPECTED_FILES, REQUIRED_FILES


class ExportError(Exception):
    """A client export we cannot load. The message is shown to the user."""


@dataclass
class Export:
    """One client export: the seven tables, validated but not yet interpreted."""

    tables: dict[str, pd.DataFrame]
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise ExportError(f"This export has no {name}.csv.")
        return self.tables[name]

    def get(self, name: str) -> pd.DataFrame | None:
        return self.tables.get(name)

    @property
    def present(self) -> list[str]:
        return sorted(self.tables)


def _match_name(filename: str) -> str | None:
    """Map an uploaded filename to one of our seven table names.

    Tolerant on purpose: the client may send `shifts (1).csv`, `SHIFTS.csv` or
    `2026-08-17_shifts.csv`. Longer names are matched first so that
    `weekly_summary` never gets claimed by a looser pattern.
    """
    stem = Path(filename).stem.lower().strip()
    stem = stem.replace("-", "_").replace(" ", "_")
    for table in sorted(EXPECTED_FILES, key=len, reverse=True):
        if table in stem:
            return table
    return None


def _validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    expected = EXPECTED_FILES[table]
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ExportError(
            f"{table}.csv is missing {'column' if len(missing) == 1 else 'columns'}: "
            f"{', '.join(missing)}. Expected: {', '.join(expected)}."
        )

    # Keep only the columns we know about, in a stable order. Extra columns in a
    # future export are ignored rather than fatal.
    return df[expected].copy()


def _read_csv(source: str | Path | IO[bytes] | bytes) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    # dtype=str so nothing is silently coerced; IDs stay strings, and a stray
    # value in a numeric column surfaces as a parse error rather than a NaN.
    return pd.read_csv(source, dtype=str, keep_default_na=True, skipinitialspace=True)


def load_export(
    files: Mapping[str, str | Path | IO[bytes] | bytes] | None = None,
    directory: str | Path | None = None,
) -> Export:
    """Load a client export.

    Pass `directory` for files on disk, or `files` as {filename: content} for an
    upload. Exactly one of the two.
    """
    if (files is None) == (directory is None):
        raise ValueError("Pass exactly one of `files` or `directory`.")

    raw: dict[str, str | Path | IO[bytes] | bytes] = {}
    warnings: list[str] = []

    if directory is not None:
        directory = Path(directory)
        if not directory.is_dir():
            raise ExportError(f"No such folder: {directory}")
        for path in sorted(directory.glob("*.csv")):
            table = _match_name(path.name)
            if table and table not in raw:
                raw[table] = path
            elif table is None:
                warnings.append(f"Ignored unrecognised file: {path.name}")
    else:
        for filename, content in files.items():
            table = _match_name(filename)
            if table is None:
                warnings.append(f"Ignored unrecognised file: {filename}")
            elif table in raw:
                warnings.append(f"Ignored duplicate {table} file: {filename}")
            else:
                raw[table] = content

    missing_required = REQUIRED_FILES - raw.keys()
    if missing_required:
        names = ", ".join(f"{t}.csv" for t in sorted(missing_required))
        raise ExportError(
            f"This export is missing {names}, which we cannot work without. "
            f"Found: {', '.join(sorted(raw)) or 'nothing'}."
        )

    tables: dict[str, pd.DataFrame] = {}
    for table, source in raw.items():
        try:
            df = _read_csv(source)
        except ExportError:
            raise
        except Exception as exc:  # pragma: no cover - pandas raises many types
            raise ExportError(f"Could not read {table}.csv: {exc}") from exc

        if df.empty:
            if table in REQUIRED_FILES:
                raise ExportError(f"{table}.csv is empty.")
            warnings.append(f"{table}.csv is empty and was skipped.")
            continue

        tables[table] = _validate(table, df)

    for table in sorted(EXPECTED_FILES.keys() - tables.keys()):
        warnings.append(f"No {table}.csv in this export.")

    return Export(tables=tables, warnings=warnings)


def load_default_export() -> Export:
    """The export that ships with the repo, used when nothing is uploaded."""
    return load_export(directory=DATA_DIR)
