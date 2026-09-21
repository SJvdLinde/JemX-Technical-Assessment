"""The hours engine, pinned against the client's own roll-up.

The central test is `test_reproduces_client_weekly_summary`. With imputation off,
our engine must reproduce `weekly_summary.csv` on every row. That file is the
output of the system the client runs today, so matching it exactly proves our
arithmetic is theirs -- and any future change that silently alters parsing,
midnight-wrapping or the overtime rule will fail here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import config as cfg
from app.hours import build_shifts, build_weekly, week_context
from app.loader import load_default_export


@pytest.fixture(scope="module")
def export():
    return load_default_export()


@pytest.fixture(scope="module")
def shifts_measured(export):
    """Client convention: unclosed shifts contribute nothing."""
    return build_shifts(export, impute=False)


@pytest.fixture(scope="module")
def shifts_imputed(export):
    """Our convention: unclosed shifts are estimated."""
    return build_shifts(export, impute=True)


# --------------------------------------------------------------------------
# The regression test that matters
# --------------------------------------------------------------------------

def test_reproduces_client_weekly_summary(export, shifts_measured):
    client = export["weekly_summary"].copy()
    client["week_start"] = pd.to_datetime(client["week_starting"])
    for col in ("total_hours", "overtime_hours", "breached"):
        client[col] = pd.to_numeric(client[col])

    ours = build_weekly(shifts_measured)

    merged = client.merge(
        ours, on=["employee_id", "week_start"], how="outer",
        suffixes=("_client", "_ours"), indicator=True,
    )

    assert (merged["_merge"] == "both").all(), (
        "Row mismatch against the client's summary: "
        f"{(merged['_merge'] != 'both').sum()} rows do not line up."
    )

    for col in ("total_hours", "overtime_hours"):
        diff = (merged[f"{col}_client"] - merged[f"{col}_ours"]).abs()
        assert (diff < 0.01).all(), (
            f"{col} differs on {int((diff >= 0.01).sum())} of {len(merged)} rows "
            f"(max difference {diff.max():.2f}h)."
        )

    mismatched = merged["breached_client"] != merged["breached_ours"]
    assert not mismatched.any(), f"breached differs on {int(mismatched.sum())} rows."


def test_client_summary_row_count(export):
    """Guards the fixture itself: 2,122 rows is what we validated against."""
    assert len(export["weekly_summary"]) == 2122


# --------------------------------------------------------------------------
# The rules, individually
# --------------------------------------------------------------------------

def test_overnight_shifts_are_wrapped(shifts_measured):
    overnight = shifts_measured[shifts_measured["is_overnight"]]
    assert len(overnight) == 1010
    assert (overnight["raw_hours"] > 0).all()
    assert overnight["raw_hours"].max() <= cfg.SHIFT_LENGTH_CEILING


def test_overnight_shifts_are_all_night_pattern(export, shifts_measured):
    """Why wrapping is safe: it only ever touches night-pattern staff."""
    employees = export["employees"][["employee_id", "shift_pattern"]]
    merged = shifts_measured.merge(employees, on="employee_id", how="left")
    assert set(merged.loc[merged["is_overnight"], "shift_pattern"]) == {"night"}


def test_no_shift_exceeds_the_ceiling(shifts_imputed):
    """The 13.5h ceiling holds, including for values we estimated."""
    assert shifts_imputed["hours"].max() <= cfg.SHIFT_LENGTH_CEILING


def test_overtime_rule(shifts_measured):
    weekly = build_weekly(shifts_measured)
    expected = (weekly["total_hours"] - cfg.ORDINARY_HOURS_CAP).clip(lower=0)
    assert ((weekly["overtime_hours"] - expected).abs() < 0.01).all()
    assert (weekly["breached"] == (weekly["total_hours"] > cfg.BREACH_TOTAL_HOURS)).all()


# --------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------

def test_unclosed_shift_count(shifts_measured):
    assert int(shifts_measured["is_unclosed"].sum()) == 184


def test_imputation_fills_every_unclosed_shift(shifts_imputed):
    assert shifts_imputed["hours"].notna().all()
    assert int(shifts_imputed["is_unclosed"].sum()) == 184


def test_imputation_only_ever_reveals_breaches(export, shifts_measured, shifts_imputed):
    """The finding that justified imputing at all.

    Dropping unclosed shifts records them as zero hours, never as unknown, so the
    bias runs one way: it can only hide a breach, never invent one. Across the 9
    complete weeks, imputing flips 13 employee-weeks into breach and none out.
    """
    live = week_context(shifts_measured).week_start

    measured = build_weekly(shifts_measured)
    imputed = build_weekly(shifts_imputed)

    merged = measured.merge(
        imputed, on=["employee_id", "week_start"], suffixes=("_m", "_i")
    )
    complete = merged[merged["week_start"] < live]

    revealed = ((complete["breached_m"] == 0) & (complete["breached_i"] == 1)).sum()
    hidden = ((complete["breached_m"] == 1) & (complete["breached_i"] == 0)).sum()

    assert hidden == 0, "Imputation removed a breach, which should be impossible."
    assert revealed == 13, f"Expected 13 revealed breaches, found {revealed}."

    assert int(complete["breached_m"].sum()) == 66
    assert int(complete["breached_i"].sum()) == 79


def test_imputed_hours_are_tracked_separately(shifts_imputed):
    """Anything estimated must stay auditable in the roll-up."""
    weekly = build_weekly(shifts_imputed)
    estimated = weekly[weekly["is_estimated"]]
    assert len(estimated) > 0
    assert (estimated["imputed_hours"] > 0).all()
    assert ((weekly["measured_hours"] + weekly["imputed_hours"] - weekly["total_hours"]).abs() < 0.01).all()


# --------------------------------------------------------------------------
# The complete grid and the live week
# --------------------------------------------------------------------------

def test_every_employee_appears_in_every_week(export, shifts_imputed):
    employees = export["employees"]
    weekly = build_weekly(shifts_imputed, employees=employees)
    assert len(weekly) == len(employees) * weekly["week_start"].nunique()
    assert set(weekly["employee_id"]) == set(employees["employee_id"])


def test_employees_who_did_not_work_are_kept(export, shifts_imputed):
    """The 8 with no usable live-week hours still need a prediction."""
    weekly = build_weekly(shifts_imputed, employees=export["employees"])
    live = week_context(shifts_imputed).week_start
    this_week = weekly[weekly["week_start"] == live]

    assert len(this_week) == 213
    idle = this_week[this_week["did_not_work"]]
    assert set(idle["employee_id"]) >= {
        "E1009", "E1101", "E1145", "E1157", "E1170", "E1196"
    }
    assert (idle["total_hours"] == 0).all()


def test_live_week_is_derived_not_hardcoded(export, shifts_imputed):
    ctx = week_context(shifts_imputed, export)
    assert ctx.week_start == pd.Timestamp("2026-08-10")
    assert ctx.week_end == pd.Timestamp("2026-08-16")
    assert ctx.data_through == pd.Timestamp("2026-08-12")
    assert ctx.days_elapsed == 3
    assert ctx.days_remaining == 4
    assert not ctx.is_complete
    # The week opens on Women's Day (observed), the busiest Monday in the data.
    assert ctx.public_holidays == [pd.Timestamp("2026-08-10")]


def test_nobody_has_breached_yet(export, shifts_imputed):
    """It is a genuine forecast: no one is over 55h after three days."""
    weekly = build_weekly(shifts_imputed, employees=export["employees"])
    live = week_context(shifts_imputed).week_start
    this_week = weekly[weekly["week_start"] == live]
    assert this_week["breached"].sum() == 0
    assert this_week["total_hours"].max() < cfg.BREACH_TOTAL_HOURS
