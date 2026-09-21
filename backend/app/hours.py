"""Raw clock-in/clock-out rows -> hours per shift -> hours per employee-week.

This is the foundation. Every number the dashboard shows is derived from here,
so the rules are explicit, the assumptions are named, and the whole thing is
pinned by a regression test against the client's own roll-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .loader import Export, ExportError


# --------------------------------------------------------------------------
# Shift level
# --------------------------------------------------------------------------

def _parse_clock(series: pd.Series) -> pd.Series:
    """'HH:MM' -> Timedelta since midnight. Unparseable becomes NaT."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "nan", "None", "-"]))
    return pd.to_timedelta(cleaned + ":00", errors="coerce")


def build_shifts(export: Export, impute: bool | None = None) -> pd.DataFrame:
    """One row per shift, with hours worked and provenance flags.

    Columns added:
      raw_hours      hours from the clock data, NaN when there is no clock-out
      hours          the number we use downstream (raw_hours, or imputed)
      is_unclosed    no clock-out; `hours` is an estimate, not a measurement
      is_capped      sits exactly on the 13.5h ceiling, so it is a lower bound
      is_overnight   crossed midnight and was wrapped +24h
    """
    impute = cfg.IMPUTE_UNCLOSED_SHIFTS if impute is None else impute

    df = export["shifts"].copy()

    df["shift_date"] = pd.to_datetime(df["shift_date"], errors="coerce")
    if df["shift_date"].isna().any():
        bad = int(df["shift_date"].isna().sum())
        raise ExportError(
            f"{bad} shift rows have a date we cannot read. Dates must look like 2026-08-12."
        )

    t_in = _parse_clock(df["clock_in_time"])
    t_out = _parse_clock(df["clock_out_time"])

    if t_in.isna().any():
        bad = int(t_in.isna().sum())
        raise ExportError(
            f"{bad} shift rows have no readable clock-in time. Times must look like 07:30."
        )

    # A clock-out at or before the clock-in means the shift crossed midnight.
    # All 1,010 such shifts in the reference data are night-pattern guards, and
    # wrapping yields sane 3.5-13.5h durations.
    elapsed = (t_out - t_in).dt.total_seconds() / 3600.0
    df["is_overnight"] = (elapsed <= 0) & elapsed.notna()
    df["raw_hours"] = np.where(df["is_overnight"], elapsed + cfg.MIDNIGHT_WRAP_HOURS, elapsed)

    implausible = df["raw_hours"].notna() & (
        (df["raw_hours"] < cfg.MIN_PLAUSIBLE_SHIFT_HOURS)
        | (df["raw_hours"] > cfg.MAX_PLAUSIBLE_SHIFT_HOURS)
    )
    if implausible.any():
        df.loc[implausible, "raw_hours"] = np.nan

    df["is_unclosed"] = df["raw_hours"].isna()
    df["is_capped"] = np.isclose(df["raw_hours"].fillna(-1), cfg.SHIFT_LENGTH_CEILING)

    df["hours"] = df["raw_hours"]
    if impute:
        df["hours"] = df["hours"].fillna(_impute_hours(df))

    df["week_start"] = df["shift_date"] - pd.to_timedelta(
        df["shift_date"].dt.weekday, unit="D"
    )
    df["weekday"] = df["shift_date"].dt.weekday
    return df


def _impute_hours(df: pd.DataFrame) -> pd.Series:
    """Estimated length for an unclosed shift.

    The employee's own median closed shift, capped at the 13.5h ceiling so we
    stay inside the range the rest of the data occupies.

    Conservative by construction: unclosed shifts carry a supervisor note 40% of
    the time against a 23.6% baseline, and those notes describe covering for
    absent colleagues -- so these shifts ran LONG. A median therefore understates
    them, and any breach it reveals is one we can stand behind.
    """
    closed = df.loc[df["raw_hours"].notna()]

    personal = closed.groupby("employee_id")["raw_hours"].agg(["median", "size"])
    reliable = personal.loc[personal["size"] >= cfg.MIN_SHIFTS_FOR_PERSONAL_MEDIAN, "median"]
    estimate = df["employee_id"].map(reliable)

    # Fallbacks, only reachable on a future export with thin history.
    if estimate.isna().any():
        by_site = closed.groupby("site_id")["raw_hours"].median()
        estimate = estimate.fillna(df["site_id"].map(by_site))
    if estimate.isna().any():
        estimate = estimate.fillna(closed["raw_hours"].median())

    return estimate.clip(upper=cfg.SHIFT_LENGTH_CEILING)


# --------------------------------------------------------------------------
# Employee-week level
# --------------------------------------------------------------------------

def build_weekly(shifts: pd.DataFrame, employees: pd.DataFrame | None = None) -> pd.DataFrame:
    """Roll shifts up to one row per employee per Mon-Sun week.

    The overtime rule is the client's, reproduced exactly:
    `overtime = max(0, total_hours - 45)` and `breached = overtime > 10`.
    A flat weekly test -- no daily split, and Sunday/public-holiday multipliers
    play no part. Those affect cost, not compliance.
    """
    usable = shifts.loc[shifts["hours"].notna()].copy()

    # Split each shift's hours into what was measured and what we estimated, so
    # the roll-up can report both without a second pass over the shift table.
    usable["_imputed"] = usable["hours"].where(usable["is_unclosed"], 0.0)

    weekly = (
        usable.groupby(["employee_id", "week_start"], as_index=False)
        .agg(
            total_hours=("hours", "sum"),
            shift_count=("shift_id", "count"),
            longest_shift=("hours", "max"),
            imputed_hours=("_imputed", "sum"),
            unclosed_shifts=("is_unclosed", "sum"),
            capped_shifts=("is_capped", "sum"),
            last_worked=("shift_date", "max"),
        )
    )

    weekly["total_hours"] = weekly["total_hours"].round(2)
    weekly["imputed_hours"] = weekly["imputed_hours"].round(2)
    weekly["overtime_hours"] = (
        (weekly["total_hours"] - cfg.ORDINARY_HOURS_CAP).clip(lower=0).round(2)
    )
    weekly["breached"] = (weekly["overtime_hours"] > cfg.OVERTIME_CAP).astype(int)

    # `total_hours` excluding anything we estimated -- what the client's own
    # system would report for this week.
    weekly["measured_hours"] = (weekly["total_hours"] - weekly["imputed_hours"]).round(2)
    weekly["is_estimated"] = weekly["unclosed_shifts"] > 0

    if employees is not None:
        weekly = _complete_grid(weekly, employees)

    return weekly.sort_values(["week_start", "employee_id"]).reset_index(drop=True)


def _complete_grid(weekly: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """Give every employee a row in every week, including weeks they did not work.

    Without this, someone who worked no shifts simply vanishes -- which is how
    the client's system loses people. `predictions.csv` needs all 213 regardless.
    """
    ids = employees["employee_id"].drop_duplicates()
    weeks = weekly["week_start"].drop_duplicates()
    grid = pd.MultiIndex.from_product([ids, weeks], names=["employee_id", "week_start"])

    filled = (
        weekly.set_index(["employee_id", "week_start"])
        .reindex(grid)
        .reset_index()
    )
    zero_cols = [
        "total_hours", "overtime_hours", "shift_count", "imputed_hours",
        "measured_hours", "unclosed_shifts", "capped_shifts", "breached",
    ]
    filled[zero_cols] = filled[zero_cols].fillna(0)
    filled["is_estimated"] = filled["is_estimated"].fillna(False).astype(bool)
    filled["breached"] = filled["breached"].astype(int)
    filled["did_not_work"] = filled["shift_count"] == 0
    return filled


# --------------------------------------------------------------------------
# The week in progress
# --------------------------------------------------------------------------

@dataclass
class WeekContext:
    """Which week is live, and how far through it the data runs.

    Derived from the data, never hardcoded -- requirement 4 depends on this.
    """

    week_start: pd.Timestamp
    week_end: pd.Timestamp
    data_through: pd.Timestamp
    days_elapsed: int
    days_remaining: int
    is_complete: bool
    public_holidays: list[pd.Timestamp]

    @property
    def remaining_weekdays(self) -> list[int]:
        return list(range(self.days_elapsed, 7))


def week_context(shifts: pd.DataFrame, export: Export | None = None) -> WeekContext:
    """Work out the live week from the data itself."""
    data_through = shifts["shift_date"].max()
    week_start = data_through - pd.Timedelta(days=int(data_through.weekday()))
    week_end = week_start + pd.Timedelta(days=6)
    days_elapsed = int(data_through.weekday()) + 1

    holidays: list[pd.Timestamp] = []
    if export is not None and export.get("public_holidays") is not None:
        dates = pd.to_datetime(export["public_holidays"]["date"], errors="coerce").dropna()
        holidays = sorted(d for d in dates if week_start <= d <= week_end)

    return WeekContext(
        week_start=week_start,
        week_end=week_end,
        data_through=data_through,
        days_elapsed=days_elapsed,
        days_remaining=7 - days_elapsed,
        is_complete=days_elapsed == 7,
        public_holidays=holidays,
    )
