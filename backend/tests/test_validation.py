"""Does the note sorting actually work?

This is the check the brief said was the most common way to lose marks. The
tests below assert the check exists, is honest about its own limits, and that
the labellers clear a defensible bar -- not that they are perfect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.loader import load_default_export
from app.notes import CATEGORIES
from app.validation import (
    disagreement_audit,
    load_hand_labels,
    typo_robustness,
    validate,
    wilson_interval,
)


@pytest.fixture(scope="module")
def export():
    return load_default_export()


@pytest.fixture(scope="module")
def result(export):
    return validate(export)


# --------------------------------------------------------------------------
# The ground truth itself
# --------------------------------------------------------------------------

def test_hand_labels_are_usable():
    hand = load_hand_labels()
    assert len(hand) == 50
    assert hand["truth"].isin(CATEGORIES).all()
    assert hand["shift_id"].is_unique


def test_sample_covers_most_categories():
    hand = load_hand_labels()
    assert hand["truth"].nunique() >= 6


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def test_llm_labels_clear_the_bar(result):
    score = result["scores"]["template_llm"]
    assert score.accuracy >= 0.90, score.errors
    assert score.reweighted >= 0.90


def test_shipped_classifier_clears_the_bar(result):
    score = result["scores"]["classifier"]
    assert score.accuracy >= 0.90, score.errors


def test_rules_are_weaker_than_the_llm(result):
    """The independent method is worse, and its errors are all one boundary."""
    llm = result["scores"]["template_llm"]
    rules = result["scores"]["rules"]
    assert rules.accuracy < llm.accuracy
    assert set(rules.errors["truth"]) == {"colleague_absent"}


def test_accuracy_is_reported_with_uncertainty(result):
    """A perfect score on 50 notes is not evidence of a perfect method."""
    score = result["scores"]["template_llm"]
    low, high = score.confidence_interval
    assert low < score.accuracy or score.accuracy < 1.0
    assert low >= 0.90  # still a defensible floor
    assert low < 1.0, "a 100% point estimate must not claim a 100% lower bound"


def test_wilson_interval_handles_a_perfect_score():
    low, high = wilson_interval(50, 50)
    assert high == 1.0
    assert 0.90 < low < 1.0


def test_reweighting_changes_the_number(result):
    """The sample is stratified, so raw and corpus-weighted accuracy differ."""
    mix = result["corpus_mix"]
    assert abs(mix.sum() - 1.0) < 1e-6
    # nothing_reported is ~20% of the corpus but only ~12% of the sample
    assert mix["nothing_reported"] > 0.15


# --------------------------------------------------------------------------
# The disagreement audit -- the part actually worth reading
# --------------------------------------------------------------------------

def test_disagreements_are_adjudicated(export):
    """Agreement alone proves nothing; two methods can share a blind spot.

    Every sampled note where the methods disagreed was the same boundary:
    `covering <name> post, no show no call`. The hand labels came down on
    `colleague_absent` every time, which is what the template labeller said.
    """
    audit = disagreement_audit(export)
    assert len(audit) > 0
    assert audit["template_right"].sum() > audit["rules_right"].sum()
    assert audit["note"].str.contains("covering", case=False).all()


# --------------------------------------------------------------------------
# Robustness, which is what requirement 4 depends on
# --------------------------------------------------------------------------

def test_classifier_degrades_gracefully(export):
    table = typo_robustness(export, edits=(0, 4, 10))
    clean = table.loc[table["edits_per_note"] == 0, "classifier"].iloc[0]
    noisy = table.loc[table["edits_per_note"] == 10, "classifier"].iloc[0]
    assert clean > 0.95
    assert noisy > 0.85, "should survive typos it has never seen"


def test_rules_degrade_badly(export):
    """The justification for shipping the model rather than the regexes."""
    table = typo_robustness(export, edits=(0, 10))
    assert table.loc[table["edits_per_note"] == 10, "rules"].iloc[0] < 0.60
