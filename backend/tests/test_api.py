"""The HTTP layer, and requirement 4.

"On Monday the client sends a new export... Loading it must not require a
developer." These tests upload files the way a person would -- including badly,
with wrong names, missing columns and junk attached -- and check the result is
either correct or a sentence a contract manager could read.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app import config as cfg
from app.api import app, assert_no_pii

client = TestClient(app)
DATA = cfg.DATA_DIR
ALL_TABLES = [
    "shifts", "employees", "sites", "shift_notes",
    "public_holidays", "weekly_summary", "payroll_details",
]


def upload(names=ALL_TABLES, rename=None, replace=None, extra=None):
    """POST an export the way a browser would."""
    files = []
    for name in names:
        data = replace.get(name) if replace and name in replace else (DATA / f"{name}.csv").read_bytes()
        filename = rename(name) if rename else f"{name}.csv"
        files.append(("files", (filename, io.BytesIO(data), "text/csv")))
    for filename, data in (extra or {}).items():
        files.append(("files", (filename, io.BytesIO(data), "application/octet-stream")))
    return client.post("/api/analyse", files=files)


@pytest.fixture(scope="module")
def payload():
    response = client.get("/api/analyse")
    assert response.status_code == 200
    return response.json()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_default_analysis_has_every_section(payload):
    for section in ("week", "headline", "at_risk", "reasons", "sites",
                    "data_gaps", "model", "warnings"):
        assert section in payload, section


def test_week_is_derived_from_the_data(payload):
    week = payload["week"]
    assert week["start"] == "2026-08-10"
    assert week["end"] == "2026-08-16"
    assert week["data_through"] == "2026-08-12"
    assert week["days_remaining"] == 4
    # The live week opens on Women's Day, the busiest Monday in the data.
    assert week["public_holidays"] == ["2026-08-10"]


def test_at_risk_is_ranked_and_populated(payload):
    at_risk = payload["at_risk"]
    assert len(at_risk) == 15
    scores = [row["risk_score"] for row in at_risk]
    assert scores == sorted(scores, reverse=True)
    top = at_risk[0]
    assert top["full_name"]
    assert top["projected_hours"] > top["h_sofar"]


def test_reasons_carry_cost(payload):
    reasons = {row["category"]: row for row in payload["reasons"]}
    assert set(reasons) <= set(payload["model"]["features"]) | set(reasons)
    assert reasons["relief_no_show"]["cost"] > 0
    assert reasons["nothing_reported"]["notes"] > 300


def test_sites_rank_avoidable_cost(payload):
    sites = payload["sites"]
    assert len(sites) == 6
    costs = [s["avoidable_cost"] for s in sites]
    assert costs == sorted(costs, reverse=True)
    assert sites[0]["site_name"]
    assert sites[0]["top_cause"]


def test_model_card_reports_baselines_and_uncertainty(payload):
    model = payload["model"]
    assert model["features"] == ["p_h_full", "h_sofar"]
    assert 0.7 < model["roc_auc"] < 1.0
    assert model["baselines"]["model"]["roc_auc"] > \
        model["baselines"]["hours_so_far"]["roc_auc"]
    assert model["backtest_breaches"] > 0


def test_data_gaps_surface_what_the_client_cannot_see(payload):
    gaps = payload["data_gaps"]
    # Three unclosed shifts this week; two of these people read 0.00h in the
    # client's own system, and all three notes describe extra work.
    assert len(gaps["unclosed_shifts"]) == 3
    assert all(row["note"] for row in gaps["unclosed_shifts"])
    assert gaps["capped_shifts"] > 300


# --------------------------------------------------------------------------
# Requirement 4: uploading next week's export
# --------------------------------------------------------------------------

def test_upload_matches_the_default_analysis():
    """The same files uploaded must give the same answer."""
    uploaded = upload().json()
    default = client.get("/api/analyse").json()
    assert uploaded["week"] == default["week"]
    assert uploaded["headline"] == default["headline"]
    assert [r["employee_id"] for r in uploaded["at_risk"]] == \
        [r["employee_id"] for r in default["at_risk"]]


def test_upload_tolerates_real_world_filenames():
    """Exports arrive as `2026-08-17_shifts (1).CSV`, not `shifts.csv`."""
    response = upload(rename=lambda n: f"2026-08-17_{n} (1).CSV")
    assert response.status_code == 200
    assert response.json()["headline"]["employees"] == 213


def test_upload_works_with_only_the_essential_files():
    """Only shifts and employees are required.

    A partial export must still answer requirement 1 -- losing `sites.csv`
    should not cost the contract manager his breach list. The panels that need
    the missing file come back empty, and the absence is reported.
    """
    response = upload(names=["shifts", "employees"])
    assert response.status_code == 200
    body = response.json()

    assert body["headline"]["employees"] == 213
    assert len(body["at_risk"]) == 15          # the graded answer survives
    assert body["model"]["roc_auc"] > 0.7

    assert body["reasons"] == []               # no notes, so no "why"
    assert body["headline"]["notes_total"] == 0
    assert any("shift_notes" in w for w in body["warnings"])
    assert any("payroll_details" in w for w in body["warnings"])

    # Sites still appear, named by id, because they come from the shift data.
    assert len(body["sites"]) == 6
    assert body["sites"][0]["site_name"] == body["sites"][0]["site_id"]


def test_junk_attachments_are_ignored_not_fatal():
    response = upload(extra={"site_photos.zip": b"PK\x03\x04junk"})
    assert response.status_code == 200
    assert any("site_photos.zip" in w for w in response.json()["warnings"])


# --------------------------------------------------------------------------
# Bad uploads must fail readably, never silently
# --------------------------------------------------------------------------

def test_missing_shifts_is_a_readable_error():
    response = upload(names=["employees", "sites"])
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "shifts.csv" in detail
    assert "Traceback" not in detail


def test_missing_column_names_the_column():
    response = upload(replace={"shifts": b"shift_id,employee_id\nS1,E1\n"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "shift_date" in detail and "clock_in_time" in detail


def test_empty_file_is_explained():
    response = upload(replace={"shifts": b""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_no_files_at_all():
    assert client.post("/api/analyse", files=[]).status_code in (400, 422)


def test_complete_week_refuses_rather_than_guessing():
    """Data ending on a Sunday has nothing left to forecast."""
    import pandas as pd

    shifts = pd.read_csv(DATA / "shifts.csv")
    shifts = shifts[shifts["shift_date"] < "2026-08-10"]
    response = upload(replace={"shifts": shifts.to_csv(index=False).encode()})
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "no days remain" in detail or "week in progress" in detail


# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------

def test_no_pii_in_any_response(payload):
    assert_no_pii(payload)
    body = str(payload)
    for column in cfg.PII_COLUMNS:
        assert column not in body


def test_pii_guard_actually_catches_a_leak():
    """Guard against the guard being a no-op."""
    with pytest.raises(RuntimeError, match="account_number"):
        assert_no_pii({"employees": [{"account_number": "123456"}]})


def test_bank_details_never_appear_even_though_payroll_is_loaded(payload):
    """`hourly_rate` is used for costing; nothing else from that file is."""
    assert payload["reasons"][0]["cost"] is not None  # costing did run
    assert "bank_name" not in str(payload)
    assert "tax_number" not in str(payload)


# --------------------------------------------------------------------------
# Validation endpoint
# --------------------------------------------------------------------------

def test_validation_endpoint_reports_intervals():
    body = client.get("/api/validation").json()
    assert body["n"] == 50
    assert body["human_labelled"] == 45
    template = body["scores"]["template_llm"]
    assert template["accuracy"] >= 0.9
    assert template["ci_low"] < 1.0, "a point estimate of 1.0 needs an honest floor"
    assert body["scores"]["rules"]["errors"]


def test_period_is_labelled_so_costs_are_not_misread(payload):
    """`reasons` and `sites` cost the whole export, not the live week.

    Three days of notes cannot answer "where is the avoidable cost
    concentrated", so the figures span everything -- and the payload says so,
    rather than letting a ten-week total read as this week's bill.
    """
    period = payload["period"]
    assert period["from"] == "2026-06-08"
    assert period["to"] == "2026-08-12"
    assert period["weeks"] == 10
