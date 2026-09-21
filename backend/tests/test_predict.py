"""The prediction model.

The tests that matter most are the leakage ones. A model that peeks at the
future scores brilliantly in backtest and fails on the held-out week, and the
failure is invisible from the inside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import config as cfg
from app.features import FEATURES, build_features, live_rows, training_rows
from app.hours import build_shifts, week_context
from app.loader import load_default_export
from app.predict import (
    backtest,
    baselines,
    evaluate,
    fit,
    forecast,
    to_predictions_csv,
)

CUTOFF = 2  # Wednesday


@pytest.fixture(scope="module")
def export():
    return load_default_export()


@pytest.fixture(scope="module")
def shifts(export):
    return build_shifts(export)


@pytest.fixture(scope="module")
def frame(shifts, export):
    return build_features(shifts, export["employees"], CUTOFF)


@pytest.fixture(scope="module")
def result(frame):
    return forecast(frame)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

def test_history_features_never_see_the_current_week(frame):
    """`p_h_full` in week N must equal the mean of weeks 1..N-1, not 1..N."""
    one = frame[frame["employee_id"] == "E1001"].sort_values("week_start")
    actual = one["h_full"].to_numpy()
    claimed = one["p_h_full"].to_numpy()
    for i in range(2, len(one) - 1):  # skip week 0 (fallback) and the live week
        assert claimed[i] == pytest.approx(actual[:i].mean(), abs=0.01), (
            f"p_h_full at week {i} leaks the present"
        )


def test_live_week_target_is_hidden(frame):
    live = live_rows(frame)
    assert live["h_rest"].isna().all()
    assert live["breach"].isna().all()
    assert live["h_sofar"].notna().all()


def test_backtest_only_trains_on_earlier_weeks(frame):
    """Scored weeks must start after the first, never include week 1."""
    scored = backtest(frame)
    train = training_rows(frame)
    assert scored["week_start"].min() > train["week_start"].min()
    assert scored["risk"].between(0, 1).all()


def test_history_feature_carries_real_person_specific_signal(frame):
    """Break the person-to-history link and performance must fall.

    Note what this does NOT do: shuffling `h_rest` would make the problem
    EASIER, not harder. Breach is `h_sofar + h_rest > 55`, so with `h_rest`
    randomised it stops being negatively correlated with `h_sofar` and the
    remaining signal gets cleaner -- a shuffle test there scores ~0.86, above
    the real model, which says nothing about leakage.

    Instead we permute each week's history column across employees, so people
    carry somebody else's past. `h_sofar` stays real, so the floor is the
    hours-so-far-only score (~0.70), not 0.5.
    """
    from sklearn.metrics import roc_auc_score

    scrambled = frame.copy()
    rng = np.random.default_rng(0)
    for week, block in scrambled.groupby("week_start"):
        scrambled.loc[block.index, "p_h_full"] = rng.permutation(
            block["p_h_full"].to_numpy()
        )

    real = backtest(frame)
    fake = backtest(scrambled)

    def roc(scored):
        usable = scored.loc[scored["breach"].notna()]
        return roc_auc_score(usable["breach"].astype(int), usable["risk"])

    assert roc(real) > roc(fake) + 0.05, (
        f"real {roc(real):.3f} vs scrambled history {roc(fake):.3f} -- "
        "the history feature is not contributing person-specific signal"
    )


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------

def test_beats_the_baselines(frame):
    scored = backtest(frame)
    scores = baselines(scored, frame)
    assert scores["model"]["roc_auc"] > scores["hours_so_far"]["roc_auc"] + 0.05
    assert scores["model"]["roc_auc"] > scores["breached_last_week"]["roc_auc"] + 0.2
    assert scores["model"]["pr_auc"] > scores["hours_so_far"]["pr_auc"] * 1.5


def test_performance_holds_up(frame):
    """Guards against a refactor quietly making the model worse."""
    metrics = evaluate(backtest(frame))
    matched = metrics["history_matched"]
    assert matched["roc_auc"] >= 0.80, matched
    assert matched["pr_auc"] >= 0.20, matched
    assert matched["lift"] >= 3.0, matched
    assert matched["top_n"][5]["precision"] >= 0.25, matched


def test_history_matched_beats_pooled(frame):
    """Performance rises with history depth, which is why we report both."""
    metrics = evaluate(backtest(frame))
    assert metrics["history_matched"]["pr_auc"] >= metrics["pooled"]["pr_auc"]


def test_model_uses_only_two_features():
    assert FEATURES == ["p_h_full", "h_sofar"]


def test_hours_so_far_lowers_predicted_remaining(frame):
    """The compensating effect, as a property of the fitted model.

    Site demand is near-constant (2% week to week), so hours are shared out of a
    fixed pool: working more early means working less later. The coefficient on
    `h_sofar` must be negative, or we have re-introduced naive projection.
    """
    model = fit(training_rows(frame))
    assert model.coef[model.features.index("h_sofar")] < 0
    assert model.coef[model.features.index("p_h_full")] > 0


# --------------------------------------------------------------------------
# The forecast itself
# --------------------------------------------------------------------------

def test_forecast_covers_the_live_week(result, export, shifts):
    ctx = week_context(shifts, export)
    live = result.predictions
    assert len(live) == 213
    assert (live["week_start"] == ctx.week_start).all()
    assert live["risk_score"].between(0, 1).all()
    assert live["risk_score"].is_monotonic_decreasing


def test_projected_hours_are_plausible(result):
    live = result.predictions
    assert (live["predicted_remaining"] >= 0).all()
    assert (live["projected_hours"] >= live["h_sofar"] - 0.01).all()
    assert live["projected_hours"].max() < 100


def test_risk_tracks_projection(result):
    """Someone projected far over 55 must not score below someone far under."""
    live = result.predictions
    over = live[live["projected_hours"] > 58]
    under = live[live["projected_hours"] < 40]
    if len(over) and len(under):
        assert over["risk_score"].min() > under["risk_score"].max()


def test_idle_employees_are_kept_at_zero_risk(result):
    """The 8 with no usable hours still need a prediction."""
    live = result.predictions
    idle = live[live["h_sofar"] == 0]
    assert len(idle) >= 6
    assert (idle["risk_score"] < 0.05).all()


# --------------------------------------------------------------------------
# predictions.csv
# --------------------------------------------------------------------------

def test_predictions_csv_shape(result, export):
    out = to_predictions_csv(result, export["employees"])
    assert list(out.columns) == ["employee_id", "will_breach", "risk_score"]
    assert len(out) == 213
    assert set(out["employee_id"]) == set(export["employees"]["employee_id"])
    assert out["employee_id"].is_unique
    assert out["will_breach"].isin([0, 1]).all()
    assert out["risk_score"].between(0, 1).all()
    assert out.notna().all().all()


def test_predictions_flag_a_sensible_number(result, export):
    out = to_predictions_csv(result, export["employees"])
    flagged = int(out["will_breach"].sum())
    assert 5 <= flagged <= 30, f"{flagged} flagged; a typical week has ~8 breaches"


def test_flagged_are_the_highest_risk(result, export):
    out = to_predictions_csv(result, export["employees"])
    flagged = out[out["will_breach"] == 1]["risk_score"]
    rest = out[out["will_breach"] == 0]["risk_score"]
    assert flagged.min() >= rest.max()


# --------------------------------------------------------------------------
# Requirement 4: any cut-off day
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [1, 2, 3, 4])
def test_works_at_any_cutoff_day(shifts, export, cutoff):
    """Next week's export may not stop on a Wednesday."""
    frame = build_features(shifts, export["employees"], cutoff)
    train = training_rows(frame)
    assert len(train) > 200
    model = fit(train)
    assert np.isfinite(model.coef).all()
    assert model.typical_error < 10


def test_complete_week_refuses_to_forecast(shifts, export):
    """Data ending on a Sunday leaves nothing to predict; say so, do not guess."""
    complete_only = shifts[shifts["week_start"] < shifts["week_start"].max()]
    assert complete_only["shift_date"].max().weekday() == 6  # ends on a Sunday
    frame = build_features(complete_only, export["employees"], CUTOFF)
    with pytest.raises(ValueError, match="no week in progress"):
        forecast(frame)


def test_sunday_cutoff_refuses_to_forecast(shifts, export):
    """A Sunday cut-off leaves no days to forecast, whatever the data says."""
    frame = build_features(shifts, export["employees"], cutoff_weekday=6)
    with pytest.raises(ValueError, match="no days remain"):
        forecast(frame)


def test_too_little_history_is_refused(shifts, export):
    """Two weeks of data is not enough to fit; fail loudly rather than invent."""
    early = shifts[shifts["week_start"] <= shifts["week_start"].min()]
    frame = build_features(early, export["employees"], CUTOFF)
    with pytest.raises(ValueError):
        forecast(frame)
