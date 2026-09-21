"""The modelling frame: one row per employee per week, split at the cut-off day.

The cut-off is a parameter, not a constant. The scenario stops on a Wednesday,
but next week's export might stop on a Friday, and requirement 4 says loading it
must not need a developer. Everything here re-derives from the data.

Two rules about leakage, because they are easy to get wrong:

  * Prior-history features use `.shift(1).expanding()`, so week N only ever sees
    weeks 1..N-1. Never the current week, never the future.
  * `h_rest` and `breach` are the target. For the live week they are NaN, because
    those days have not happened yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# The two features the model uses. Nested forward selection converged on this
# pair independently in the three most recent folds, and adding anything else
# measurably hurt (19 features -> ROC 0.803; 4 -> 0.840; these 2 -> 0.848).
#
#   p_h_full  the employee's historical mean weekly hours
#             -> "is this person even capable of a 55-hour week?"
#   h_sofar   hours worked before the cut-off
#             -> "are they running hot this week?"
#
# Neither works alone (ROC 0.779 and 0.697). The combination is what matters.
FEATURES = ["p_h_full", "h_sofar"]

TARGET = "h_rest"


def build_features(
    shifts: pd.DataFrame,
    employees: pd.DataFrame,
    cutoff_weekday: int,
) -> pd.DataFrame:
    """Employee-week frame split at `cutoff_weekday` (0=Monday ... 6=Sunday).

    Days <= cutoff are "so far" (known); days after are "rest" (to predict).
    Every employee gets a row in every week, including weeks they did not work --
    people who vanish are exactly the people the client's system loses.
    """
    usable = shifts.loc[shifts["hours"].notna()]

    before = usable.loc[usable["weekday"] <= cutoff_weekday]
    after = usable.loc[usable["weekday"] > cutoff_weekday]

    sofar = before.groupby(["employee_id", "week_start"], as_index=False).agg(
        h_sofar=("hours", "sum"),
        n_sofar=("shift_id", "count"),
        d_sofar=("shift_date", "nunique"),
    )
    rest = after.groupby(["employee_id", "week_start"], as_index=False).agg(
        h_rest=("hours", "sum"),
        d_rest=("shift_date", "nunique"),
    )

    ids = employees["employee_id"].drop_duplicates()
    weeks = pd.Index(sorted(usable["week_start"].unique()), name="week_start")
    grid = pd.MultiIndex.from_product(
        [ids, weeks], names=["employee_id", "week_start"]
    ).to_frame(index=False)

    df = grid.merge(sofar, how="left").merge(rest, how="left")
    df[["h_sofar", "n_sofar", "d_sofar"]] = df[["h_sofar", "n_sofar", "d_sofar"]].fillna(0)

    # A week is complete only if the data actually runs past the cut-off.
    last_day = usable["shift_date"].max()
    last_week = last_day - pd.Timedelta(days=int(last_day.weekday()))
    week_is_live = (df["week_start"] == last_week) & (
        int(last_day.weekday()) <= cutoff_weekday
    )

    df[["h_rest", "d_rest"]] = df[["h_rest", "d_rest"]].fillna(0)
    df.loc[week_is_live, ["h_rest", "d_rest"]] = np.nan

    df["h_full"] = df["h_sofar"] + df["h_rest"]
    df["breach"] = np.where(
        df["h_full"].isna(), np.nan, (df["h_full"] > cfg.BREACH_TOTAL_HOURS).astype(float)
    )
    df["is_live"] = week_is_live
    df["cutoff_weekday"] = cutoff_weekday

    df = df.sort_values(["employee_id", "week_start"]).reset_index(drop=True)
    return _add_history(df)


def days_remaining(frame: pd.DataFrame) -> int:
    """Days left to forecast after the cut-off. Zero means nothing to predict."""
    return 6 - int(frame["cutoff_weekday"].iloc[0])


def _add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Prior-weeks features. `.shift(1)` is what keeps this honest."""
    by_emp = df.groupby("employee_id")
    for col in ("h_full", "h_rest", "h_sofar"):
        df[f"p_{col}"] = by_emp[col].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    df["weeks_of_history"] = by_emp["h_full"].transform(
        lambda s: s.shift(1).expanding().count()
    )

    # An employee with no history yet falls back to the population mean of the
    # weeks we can see. Only reachable in week 1, or for a genuinely new starter.
    for col in ("p_h_full", "p_h_rest", "p_h_sofar"):
        df[col] = df[col].fillna(df[col].mean())

    return df


def training_rows(frame: pd.DataFrame, min_history: int = 1) -> pd.DataFrame:
    """Rows usable for fitting: completed weeks with enough history."""
    return frame.loc[
        (~frame["is_live"])
        & frame["h_rest"].notna()
        & (frame["weeks_of_history"] >= min_history)
    ].copy()


def live_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The week we are forecasting -- every employee, whether they worked or not."""
    return frame.loc[frame["is_live"]].copy()
