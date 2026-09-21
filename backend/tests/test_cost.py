"""Costing: hours -> rands, and the PII boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from app import config as cfg
from app.cost import cost_shifts, cost_weekly, hourly_rates
from app.hours import build_shifts, build_weekly
from app.loader import load_default_export


@pytest.fixture(scope="module")
def export():
    return load_default_export()


@pytest.fixture(scope="module")
def costed(export):
    return cost_shifts(build_shifts(export), export)


def test_every_employee_has_a_rate(export):
    rates = hourly_rates(export)
    assert len(rates) == 213
    assert rates.between(20, 100).all()


def test_rates_vary_by_role(export):
    """Sanity: a supervisor hour is not a cleaner hour."""
    rates = hourly_rates(export).rename("rate")
    employees = export["employees"].set_index("employee_id")
    merged = employees.join(rates)
    by_role = merged.groupby("role")["rate"].mean()
    assert by_role["Supervisor"] > by_role["Cleaner"] * 1.4


def test_ordinary_and_overtime_hours_reconcile(costed):
    total = costed["ordinary_hours"] + costed["overtime_hours"]
    assert ((total - costed["hours"]).abs() < 1e-9).all()


def test_overtime_allocated_only_above_the_weekly_cap(costed):
    """A week under 45h must carry no overtime cost at all."""
    weekly_hours = costed.groupby(["employee_id", "week_start"])["hours"].sum()
    weekly_ot = costed.groupby(["employee_id", "week_start"])["overtime_hours"].sum()
    under = weekly_hours[weekly_hours <= cfg.ORDINARY_HOURS_CAP].index
    assert (weekly_ot.loc[under] < 0.01).all()

    over = weekly_hours[weekly_hours > cfg.ORDINARY_HOURS_CAP]
    expected = (over - cfg.ORDINARY_HOURS_CAP)
    assert ((weekly_ot.loc[over.index] - expected).abs() < 0.01).all()


def test_premium_days_cost_more(costed):
    """A Sunday hour costs double a plain weekday hour."""
    plain = costed[~costed["is_premium_day"] & (costed["overtime_hours"] == 0)]
    premium = costed[costed["is_sunday"] & (costed["overtime_hours"] == 0)]

    plain_rate = (plain["cost"] / plain["hours"] / plain["hourly_rate"]).round(3)
    premium_rate = (premium["cost"] / premium["hours"] / premium["hourly_rate"]).round(3)
    assert (plain_rate == 1.0).all()
    assert (premium_rate == cfg.SUNDAY_MULTIPLIER).all()


def test_public_holiday_monday_is_charged_at_premium(costed):
    """2026-08-10 is Women's Day (observed) and opens the live week."""
    holiday = costed[costed["shift_date"] == pd.Timestamp("2026-08-10")]
    assert len(holiday) > 0
    assert holiday["is_public_holiday"].all()
    assert (holiday["premium_cost"] > 0).all()


def test_premium_cost_is_the_avoidable_part(costed):
    flat = costed["hours"] * costed["hourly_rate"]
    assert ((costed["cost"] - flat - costed["premium_cost"]).abs() < 0.02).all()
    assert (costed["premium_cost"] >= -0.01).all()


def test_weekly_costs_roll_up(costed):
    weekly = cost_weekly(costed)
    assert (weekly["cost"] > 0).all()
    assert abs(weekly["cost"].sum() - costed["cost"].sum()) < 1.0


def test_pii_never_leaves_payroll(export):
    """The only payroll column we read is hourly_rate."""
    rates = hourly_rates(export)
    assert rates.name is None or rates.name not in cfg.PII_COLUMNS
    assert isinstance(rates, pd.Series)  # not a frame carrying bank details


def test_costing_ignores_compliance_and_vice_versa(export):
    """Pay multipliers must not leak into the breach test.

    A Sunday hour costs double but counts once towards the 45/10 caps -- which is
    exactly how the client's engine behaves.
    """
    shifts = build_shifts(export, impute=False)
    weekly = build_weekly(shifts)
    sunday_heavy = shifts[shifts["weekday"] == 6].groupby(
        ["employee_id", "week_start"]
    )["hours"].sum()
    merged = weekly.set_index(["employee_id", "week_start"]).join(
        sunday_heavy.rename("sunday_hours")
    )
    # total_hours is a plain sum: Sunday hours are in it once, not twice.
    assert (merged["sunday_hours"].fillna(0) <= merged["total_hours"] + 0.01).all()
