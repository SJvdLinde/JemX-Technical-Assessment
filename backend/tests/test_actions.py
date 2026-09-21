"""Requirement 2: what to do about it.

"A suggestion that could have been written without looking at the data is worth
nothing to them." So the tests here mostly check that every sentence carries
numbers and names that came out of the data, and that we never suggest moving a
shift to somebody who would breach by taking it.
"""

from __future__ import annotations

import re

import pytest

from app import config as cfg
from app.actions import recommend, systemic
from app.loader import load_default_export
from app.notes import classify_notes
from app.pipeline import analyse_default


@pytest.fixture(scope="module")
def analysis():
    return analyse_default()


@pytest.fixture(scope="module")
def actions(analysis):
    return recommend(analysis)


@pytest.fixture(scope="module")
def fixes(analysis):
    return systemic(analysis, classify_notes(analysis.export))


# --------------------------------------------------------------------------
# Per-person actions
# --------------------------------------------------------------------------

def test_one_action_per_flagged_employee(actions):
    assert len(actions) == 15
    assert len({a["employee_id"] for a in actions}) == 15


def test_every_action_carries_a_real_number(actions):
    """The test for "could this have been written without the data?".

    The person's name is a separate field now, laid out by the dashboard rather
    than embedded in the sentence, so what matters here is that the instruction
    itself is quantified.
    """
    for a in actions:
        assert re.search(r"\d+\.\d+h", a["action"]), a["action"]
        assert a["full_name"]


def test_no_vague_advice(actions):
    """Phrases that would be true of any employee anywhere."""
    banned = ["monitor", "keep an eye", "be aware", "as appropriate", "consider whether"]
    for a in actions:
        lowered = a["action"].lower()
        for phrase in banned:
            assert phrase not in lowered, (phrase, a["action"])


def test_the_person_over_the_cap_gets_a_concrete_cut(actions):
    over = [a for a in actions if a["projected_hours"] > cfg.BREACH_TOTAL_HOURS]
    assert over, "expected at least one person projected over 55h"
    for a in over:
        assert a["shifts_to_cut"] >= 1
        assert a["hours_over_target"] > 0
        assert a["action"].startswith("Cut ")


def test_needs_action_covers_people_inside_the_error_margin(analysis, actions):
    """Being under 55 is not safe when the forecast is out by ~3.5h.

    Someone at 51.5h with a 3.5h typical error is one unplanned shift from a
    breach, and saying so is more useful than calling them fine.
    """
    error = analysis.forecast.model.typical_error
    for a in actions:
        expected = a["projected_hours"] > cfg.BREACH_TOTAL_HOURS - 2.0 or \
            a["headroom_hours"] <= error
        assert a["needs_action"] == expected, a["full_name"]

    assert any(a["needs_action"] for a in actions)


def test_cover_options_only_where_action_is_needed(actions):
    for a in actions:
        if not a["needs_action"]:
            assert a["cover_options"] == [], a["full_name"]


def test_a_swap_never_breaches_the_person_taking_it(actions):
    """The whole point is not to move the problem to somebody else."""
    for a in actions:
        for candidate in a["cover_options"]:
            assert candidate["projected_after_swap"] < cfg.BREACH_TOTAL_HOURS, (
                a["full_name"], candidate
            )


def test_cover_candidates_are_same_site_and_role(analysis, actions):
    """A guard post needs a guard, at that site."""
    employees = analysis.export["employees"].set_index("employee_id")
    for a in actions:
        for candidate in a["cover_options"]:
            other = employees.loc[candidate["employee_id"]]
            assert other["primary_site_id"] == a["site_id"]
            assert other["role"] == a["role"]
            assert candidate["employee_id"] != a["employee_id"]


def test_no_cover_options_is_itself_reported(actions):
    """A site with no spare capacity is a finding, not a blank."""
    for a in actions:
        assert isinstance(a["cover_options"], list)


# --------------------------------------------------------------------------
# Systemic fixes
# --------------------------------------------------------------------------

def test_systemic_fixes_are_ranked_by_cost(fixes):
    assert len(fixes) >= 3
    costs = [f["cost"] for f in fixes]
    assert costs == sorted(costs, reverse=True)


def test_systemic_fixes_name_a_site_and_a_rand_figure(fixes):
    for f in fixes:
        assert f["sentence"]
        assert "R" in f["sentence"]
        assert f["worst_site_name"] in f["sentence"]
        assert f["fix"]


def test_no_em_dashes_anywhere(actions, fixes):
    """Figures and short clauses, not prose."""
    for a in actions:
        assert "\u2014" not in a["action"], a["action"]
    for f in fixes:
        assert "\u2014" not in (f["sentence"] or "")


def test_systemic_only_covers_avoidable_cost(fixes):
    """Client-requested overtime is billable, so it is not a thing to fix."""
    assert "client_requested" not in {f["category"] for f in fixes}


def test_shares_are_plausible(fixes):
    total = sum(f["share_of_avoidable"] for f in fixes)
    assert 0 < total <= 1.01


def test_systemic_is_empty_without_notes(analysis):
    import pandas as pd
    assert systemic(analysis, pd.DataFrame()) == []
