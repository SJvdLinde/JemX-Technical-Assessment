"""Turning hours into rands.

The contract manager does not act on "9.25 overtime hours". They act on
"R418 burned at Menlyn covering no-shows". This module is what makes that
sentence possible, and it is the only place `payroll_details.csv` is touched.

Note the separation: pay multipliers live HERE and nowhere else. They have no
bearing on whether someone breaches -- the client's compliance engine ignores
them entirely, and so does ours. Cost and compliance are different questions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .loader import Export


def hourly_rates(export: Export) -> pd.Series:
    """employee_id -> hourly rate. The only column we take from payroll.

    Everything else in that file is bank and tax detail which never leaves the
    backend; see cfg.PII_COLUMNS.
    """
    payroll = export.get("payroll_details")
    if payroll is None:
        return pd.Series(dtype="float64")

    rates = pd.to_numeric(payroll["hourly_rate"], errors="coerce")
    return pd.Series(rates.values, index=payroll["employee_id"].values).dropna()


def cost_shifts(
    shifts: pd.DataFrame,
    export: Export,
    weekly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a rand cost to each shift.

    The premium model, stated as an assumption rather than derived from the data
    (the client's export contains no pay run, so there is nothing to verify against):

      * Sunday and public-holiday hours   -> 2.0x
      * hours above the 45h weekly cap    -> 1.5x
      * everything else                   -> 1.0x

    A shift is charged at the highest premium that applies, not the sum of them.
    Overtime is a weekly concept, so we allocate it to the LAST shifts of the
    week -- the hours above 45 are by definition the ones worked last. That
    ordering is what makes "cancel Sunday's shift" a costable action.
    """
    rates = hourly_rates(export)
    df = shifts.loc[shifts["hours"].notna()].copy()
    df["hourly_rate"] = df["employee_id"].map(rates)

    holidays = set()
    if export.get("public_holidays") is not None:
        holidays = set(
            pd.to_datetime(export["public_holidays"]["date"], errors="coerce").dropna()
        )

    df["is_sunday"] = df["weekday"] == 6
    df["is_public_holiday"] = df["shift_date"].isin(holidays)
    df["is_premium_day"] = df["is_sunday"] | df["is_public_holiday"]

    df = df.sort_values(["employee_id", "week_start", "shift_date", "clock_in_time"])
    cumulative = df.groupby(["employee_id", "week_start"])["hours"].cumsum()

    # Hours on this shift that fall above the weekly ordinary cap.
    hours_before = cumulative - df["hours"]
    ot_hours = (cumulative - cfg.ORDINARY_HOURS_CAP).clip(lower=0) - (
        hours_before - cfg.ORDINARY_HOURS_CAP
    ).clip(lower=0)
    df["overtime_hours"] = ot_hours.clip(lower=0, upper=df["hours"])
    df["ordinary_hours"] = df["hours"] - df["overtime_hours"]

    multiplier = np.where(
        df["is_premium_day"], cfg.PUBLIC_HOLIDAY_MULTIPLIER, 1.0
    )
    ot_multiplier = np.where(
        df["is_premium_day"], cfg.PUBLIC_HOLIDAY_MULTIPLIER, cfg.OVERTIME_MULTIPLIER
    )

    df["cost"] = (
        df["ordinary_hours"] * df["hourly_rate"] * multiplier
        + df["overtime_hours"] * df["hourly_rate"] * ot_multiplier
    ).round(2)

    # What those same hours would have cost at flat rate -- the avoidable part.
    df["premium_cost"] = (
        df["cost"] - df["hours"] * df["hourly_rate"]
    ).round(2)

    return df


def cost_weekly(costed_shifts: pd.DataFrame) -> pd.DataFrame:
    """Roll shift costs up to employee-week."""
    df = costed_shifts.copy()
    df["_sunday_hours"] = df["hours"].where(df["is_sunday"], 0.0)
    df["_holiday_hours"] = df["hours"].where(df["is_public_holiday"], 0.0)

    rolled = df.groupby(["employee_id", "week_start"], as_index=False).agg(
        cost=("cost", "sum"),
        premium_cost=("premium_cost", "sum"),
        overtime_hours_costed=("overtime_hours", "sum"),
        sunday_hours=("_sunday_hours", "sum"),
        public_holiday_hours=("_holiday_hours", "sum"),
    )
    numeric = rolled.select_dtypes("number").columns
    rolled[numeric] = rolled[numeric].round(2)
    return rolled
