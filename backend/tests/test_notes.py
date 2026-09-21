"""Note classification.

The interesting tests are the multilingual ones and the disputed-request one.
Getting `disputed_client_request` wrong is the most expensive single mistake
available here: it tells the contract manager to relax about the exact thing
costing him money.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.loader import load_default_export
from app.notes import (
    CATEGORIES,
    CLIENT_PAYS,
    COMPANY_PAYS,
    COST_OF_CATEGORY,
    NoteClassifier,
    build_template_labeller,
    classify_notes,
    label_by_rules,
    normalise,
    to_note_classifications_csv,
)


@pytest.fixture(scope="module")
def export():
    return load_default_export()


@pytest.fixture(scope="module")
def labeller(export):
    return build_template_labeller(export["employees"])


@pytest.fixture(scope="module")
def classified(export):
    return classify_notes(export)


def label(labeller, text: str) -> str:
    return labeller.label_one(text)[0]


# --------------------------------------------------------------------------
# The categories that carry money
# --------------------------------------------------------------------------

def test_disputed_request_is_not_billed_to_the_client(labeller):
    """The paperwork says client-approved; the note names a no-show."""
    for text in [
        "client signed for the extra hrs but real reason is relief no show again",
        "CLIENT SIGNED FOR THE EXTRA HOURS BUT REAL REASON IS RELIEF NO SHOW AGN",
        "client signed for the extra hours but real reason is relief no show agin",
    ]:
        category = label(labeller, text)
        assert category == "disputed_client_request", (text, category)
        assert COST_OF_CATEGORY[category] == COMPANY_PAYS


def test_genuine_client_requests_are_billable(labeller):
    for text in [
        "requested by centre management for load in, signed off",
        "CLIENT REQUESTED DEEP CLEAN BEFORE THE AUDIT - APPROVED",
        "client asked us to stay for the delivery, ok'd by centre mgmt",
        "klient het ekstra ure gevra vir stocktake",
        "additional cover requested by site manager, signed off",
    ]:
        category = label(labeller, text)
        assert category == "client_requested", (text, category)
        assert COST_OF_CATEGORY[category] == CLIENT_PAYS


def test_additional_cover_is_a_client_request_not_a_coverage_gap(labeller):
    """The word 'cover' must not override explicit sign-off.

    `additional cover requested by site manager, signed off` is billable work,
    not somebody standing in for an absent colleague. This was a real ordering
    bug in the rules, found by comparing the two methods on 58 notes.
    """
    text = "additional cover requested by site manager, signed off"
    assert label(labeller, text) == "client_requested"
    assert label_by_rules(pd.Series([text])).iloc[0] == "client_requested"


# --------------------------------------------------------------------------
# Three languages
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # isiZulu
    ("akukho lutho", "nothing_reported"),
    ("uMahlangu akezanga namhlanje, ngimele yena", "colleague_absent"),
    ("next shift akafikanga, ngihlale kuze kube 06h00", "relief_no_show"),
    # Afrikaans
    ("niks om te rapporteer nie", "nothing_reported"),
    ("gedek vir Tshabalala, siek gemeld", "colleague_absent"),
    ("aflos het nie opgedaag nie, moes aanbly", "relief_no_show"),
    ("oorhandiging was laat, gewag vir sleutels", "late_handover"),
    ("masjien is stukkend, alles met die hand gedoen", "equipment_failure"),
    ("klient het ekstra ure gevra vir stocktake", "client_requested"),
    # English
    ("buffer machine kaput, did the floor by hand", "equipment_failure"),
    ("control room says relief coming, nobody came", "relief_no_show"),
    ("double duty today, Radebe on family responsibility leave", "colleague_absent"),
    ("waited 25 min for handover, ob book not signed", "late_handover"),
    ("ntr", "nothing_reported"),
])
def test_multilingual_labelling(labeller, text, expected):
    assert label(labeller, text) == expected


def test_typos_land_on_the_right_template(labeller):
    """Heavy misspellings must still resolve. These are all real notes."""
    for text, expected in [
        ("bfufer machine kaput, did the floor by hand", "equipment_failure"),
        ("srubber broke down had to do the floor manualyl", "equipment_failure"),
        ("nexxt shift guuard did not pitch had to cover", "relief_no_show"),
        ("GEDK VIR MOKOENA, SEK GEMELD", "colleague_absent"),
        ("shift handovver delayed by 50 min", "late_handover"),
        ("relif no shwo again", "relief_no_show"),
        ("oorhandiging was laa gewag vir sleutels", "late_handover"),
    ]:
        assert label(labeller, text) == expected, text


def test_unclear_when_authorisation_is_unknown(labeller):
    """The supervisor himself does not know. Saying so beats guessing."""
    assert label(labeller, "client says stay till 06h00, dont know if office apprved") \
        == "unclear"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_time_formats_collapse():
    """6, six, 6am, 0600, 06h00 are one thing."""
    variants = ["stayed till 6", "stayed till six", "stayed till 6am",
                "stayed till 0600", "stayed till 06h00"]
    assert len({normalise(v) for v in variants}) == 1


def test_blank_notes_are_nothing_reported(labeller):
    for text in ["", "   ", "-", ".", None]:
        assert label(labeller, text) == "nothing_reported"


def test_names_do_not_create_new_templates(labeller):
    """`covering Wyk post` and `covering Ndlovu post` are the same sentence."""
    a = labeller.label_one("covering Wyk post, no show no call")
    b = labeller.label_one("covering Ndlovu post, no show no call")
    assert a[0] == b[0]
    assert a[2] == b[2]  # same matched template


# --------------------------------------------------------------------------
# The two methods, and their agreement
# --------------------------------------------------------------------------

def test_every_note_gets_a_valid_category(classified):
    assert len(classified) == 2117
    assert classified["category"].isin(CATEGORIES).all()
    assert classified["rules_category"].isin(CATEGORIES).all()
    assert classified["model_category"].isin(CATEGORIES).all()


def test_methods_agree_substantially(classified):
    """Two independently written methods. Disagreement marks the hard cases."""
    assert classified["methods_agree"].mean() > 0.90


def test_all_templates_matched_confidently(classified):
    """No note should be a distant guess -- the corpus really is ~35 sentences."""
    assert classified["match_score"].min() > 0.6


def test_classifier_reproduces_the_llm_labels(classified):
    """Distillation fidelity.

    This is NOT evidence the labels are correct -- only that the shipped model
    learned what the language model decided. Correctness is measured against the
    hand-labelled sample in test_validation.py.
    """
    assert (classified["category"] == classified["model_category"]).mean() > 0.98


def test_classifier_beats_rules_on_unseen_typos(export):
    """Why a character n-gram model ships instead of the regexes.

    Next week's export will contain misspellings nobody has seen. At ten random
    character edits the classifier holds ~94% while the rules fall below 40%.
    """
    import random

    from sklearn.metrics import accuracy_score

    notes = export["shift_notes"]["note"].fillna("")
    labels = classify_notes(export, train_classifier=False)["category"]

    rng = random.Random(0)

    def corrupt(text: str, edits: int = 10) -> str:
        chars = list(text)
        for _ in range(edits):
            if len(chars) < 4:
                break
            i = rng.randrange(len(chars))
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)

    split = int(len(notes) * 0.7)
    model = NoteClassifier().fit(notes.iloc[:split], labels.iloc[:split])

    noisy = notes.iloc[split:].map(corrupt)
    truth = labels.iloc[split:]
    assert accuracy_score(truth, model.predict(noisy)) > 0.85
    assert accuracy_score(truth, label_by_rules(noisy)) < 0.60


# --------------------------------------------------------------------------
# The cost split
# --------------------------------------------------------------------------

def test_cost_axis_covers_every_category():
    assert set(COST_OF_CATEGORY) == set(CATEGORIES)


def test_operational_failures_are_the_companys_cost(classified):
    failures = classified[classified["category"].isin(
        ["relief_no_show", "colleague_absent", "late_handover",
         "equipment_failure", "disputed_client_request"]
    )]
    assert (failures["who_pays"] == COMPANY_PAYS).all()
    assert len(failures) > 1000  # most overtime is avoidable, which is the point


# --------------------------------------------------------------------------
# note_classifications.csv
# --------------------------------------------------------------------------

def test_submission_file_shape(classified, export):
    out = to_note_classifications_csv(classified)
    assert list(out.columns) == ["shift_id", "category", "note"]
    assert len(out) == len(export["shift_notes"])
    assert out["shift_id"].is_unique
    assert out["category"].isin(CATEGORIES).all()
    assert out["note"].notna().all()


def test_submission_keeps_the_original_note_text(classified, export):
    """They will read a sample by hand, so the text must be untouched."""
    out = to_note_classifications_csv(classified)
    original = export["shift_notes"].set_index("shift_id")["note"].fillna("")
    rejoined = out.set_index("shift_id")["note"]
    assert (rejoined == original.reindex(rejoined.index)).all()


# --------------------------------------------------------------------------
# The cascade: template -> classifier -> LLM -> unclear
# --------------------------------------------------------------------------

def _export_with_notes(export, texts):
    """Copy an export, appending synthetic notes to the end."""
    import copy

    clone = copy.copy(export)
    clone.tables = dict(export.tables)
    extra = pd.DataFrame({
        "shift_id": [f"SYNTH{i}" for i in range(len(texts))],
        "logged_by": ["SUP-99"] * len(texts),
        "note": texts,
    })
    clone.tables["shift_notes"] = pd.concat(
        [export["shift_notes"], extra], ignore_index=True
    )
    return clone


def test_known_notes_never_leave_gate_one(classified):
    """Every note in the shipped corpus is a sentence we already know."""
    assert (classified["source"] == "template").all()
    assert classified["match_score"].min() >= 0.62


def test_classifier_rescues_a_new_wording(export):
    """Gate 2: templates score this 0.52 and give up; the classifier gets it."""
    clone = _export_with_notes(export, ["my relief bunked, i had to stay till morning"])
    out = classify_notes(clone, use_llm=False)
    row = out[out["shift_id"] == "SYNTH0"].iloc[0]
    assert row["match_score"] < 0.62, "should not have matched a template"
    assert row["source"] == "classifier"
    assert row["category"] == "relief_no_show"


def test_genuinely_new_causes_are_not_silently_absorbed(export):
    """Gate 4, and the reason the cascade exists.

    Left alone the classifier files these as `nothing_reported` -- recording a
    real operational failure as a normal shift. With no LLM key they must come
    back `unclear` and be visible instead.
    """
    novel = [
        "roadworks on the N1, whole team arrived two hours late",
        "protest outside the gate, nobody could leave site",
        "load shedding, no power, stayed for security",
    ]
    clone = _export_with_notes(export, novel)
    out = classify_notes(clone, use_llm=False)
    synthetic = out[out["shift_id"].str.startswith("SYNTH")]

    assert (synthetic["category"] == "unclear").all(), synthetic[["note", "category"]]
    assert (synthetic["who_pays"] == "no_signal").all()
    assert not (synthetic["category"] == "nothing_reported").any()


def test_llm_gate_fills_in_when_available(export, monkeypatch, tmp_path):
    """Gate 3, driven by a stub so the test needs no key and no network."""
    from app import llm as llm_module

    class StubLabeller:
        available = True

        def label(self, notes):
            return [llm_module.LLMResult("other", "load_shedding") for _ in notes]

    def fake_build(enabled=True):
        return llm_module.CachedLabeller(
            inner=StubLabeller(), path=tmp_path / "cache.json"
        )

    monkeypatch.setattr(llm_module, "build_labeller", fake_build)

    clone = _export_with_notes(export, ["load shedding, no power, stayed for security"])
    out = classify_notes(clone, use_llm=True)
    row = out[out["shift_id"] == "SYNTH0"].iloc[0]

    # `other` is not in our taxonomy, so it lands in `unclear` -- but the
    # proposed name is kept, which is the useful part.
    assert row["category"] == "unclear"
    assert row["suggested_category"] == "load_shedding"


def test_missing_api_key_does_not_break_anything(export, monkeypatch):
    """Requirement 4: no key must still give a complete, working result.

    Every provider key is cleared AND `.env` loading is stubbed out, so this
    genuinely exercises the no-key path instead of quietly hitting the network
    because a developer happens to have a key on disk.
    """
    from app import llm as llm_module

    for name in ("ANTHROPIC_API_KEY", *llm_module.GeminiLabeller.ENV_KEYS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_module, "load_dotenv", lambda path=None: None)
    clone = _export_with_notes(export, ["something nobody has ever written before qqq"])
    out = classify_notes(clone, use_llm=True)
    assert len(out) == len(export["shift_notes"]) + 1
    assert out["category"].notna().all()
    assert out["who_pays"].notna().all()
