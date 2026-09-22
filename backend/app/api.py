"""The HTTP layer.

Requirement 4: "On Monday the client sends a new export... Loading it must not
require a developer." So the only thing the frontend does is POST seven CSVs to
`/api/analyse`. Everything else -- which week is live, which weekday the data
stops on, how far to train back -- is derived from the upload itself.

Two rules enforced here:

  * `payroll_details.csv` carries bank accounts and tax numbers. Only
    `hourly_rate` is ever read, and nothing from that file reaches a response.
    `assert_no_pii` makes that a runtime guarantee, not a convention.
  * A bad upload must fail with a sentence a contract manager could read, never
    a stack trace and never a silently wrong number.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config as cfg
from .loader import ExportError, load_default_export, load_export
from .notes import (
    CATEGORIES,
    COMPANY_PAYS,
    COST_OF_CATEGORY,
    FIX_FOR_CATEGORY,
    classify_notes,
)
from .actions import recommend, systemic
from .pipeline import Analysis, analyse
from .validation import validate

MAX_UPLOAD_BYTES = 32 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the cache at boot rather than charging it to the first visitor."""
    try:
        default_payload()
    except Exception:  # pragma: no cover - never block startup on this
        pass
    yield


app = FastAPI(
    title="Jem Ops Room",
    description="Who will breach the 10-hour overtime cap by Sunday.",
    version="1.0.0",
    lifespan=lifespan,
)

# The frontend is deployed separately, so it is a cross-origin caller.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------

def assert_no_pii(payload: Any) -> None:
    """Fail loudly if a PII field ever appears in a response.

    Jem builds payroll software. A bank account number leaking into a dashboard
    JSON would be a real incident, so this is checked on every response rather
    than trusted to code review.
    """
    encoded = json.dumps(payload, default=str)
    for column in cfg.PII_COLUMNS:
        if f'"{column}"' in encoded:
            raise RuntimeError(f"PII column '{column}' reached the API response")


def _clean(value: Any) -> Any:
    """NumPy and pandas types are not JSON-serialisable; NaN is not valid JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


# --------------------------------------------------------------------------
# Building the response
# --------------------------------------------------------------------------

def build_payload(analysis: Analysis, top_n: int = 15) -> dict:
    """Everything the dashboard renders, derived once.

    Only `shifts` and `employees` are required. A partial export still produces
    a working answer -- the panels that need the missing file are empty and the
    absence is reported in `warnings`, rather than the whole upload failing.
    An export missing `sites.csv` should not cost the manager his breach list.
    """
    export = analysis.export
    ctx = analysis.context
    employees = export["employees"][
        ["employee_id", "full_name", "role", "primary_site_id", "shift_pattern"]
    ]

    site_table = export.get("sites")
    sites = (
        site_table.set_index("site_id")["site_name"]
        if site_table is not None
        else pd.Series(dtype=object)
    )

    has_notes = export.get("shift_notes") is not None
    classified = classify_notes(export) if has_notes else _empty_classified()
    costed = analysis.costed

    return {
        "week": {
            "start": ctx.week_start.date().isoformat(),
            "end": ctx.week_end.date().isoformat(),
            "data_through": ctx.data_through.date().isoformat(),
            "days_elapsed": ctx.days_elapsed,
            "days_remaining": ctx.days_remaining,
            "public_holidays": [d.date().isoformat() for d in ctx.public_holidays],
        },
        # The cost figures in `reasons` and `sites` cover the WHOLE export, not
        # just the live week. "Where is avoidable cost concentrated" is a
        # pattern question, and three days of notes would not answer it -- but
        # the dashboard must say so rather than let it read as this week's bill.
        "period": {
            "from": _clean(analysis.shifts["shift_date"].min()),
            "to": _clean(analysis.shifts["shift_date"].max()),
            "weeks": int(analysis.shifts["week_start"].nunique()),
        },
        "headline": _headline(analysis, classified, costed, top_n),
        "actions": recommend(analysis, top_n),
        "systemic": systemic(analysis, classified),
        "at_risk": _at_risk(analysis, employees, sites, top_n),
        "reasons": _reasons(classified, costed, export),
        "sites": _sites(classified, costed, export, sites),
        "data_gaps": _data_gaps(analysis, classified, employees),
        "model": _model_card(analysis),
        "warnings": list(export.warnings),
    }


def _empty_classified() -> pd.DataFrame:
    """Stand-in when an export arrives without `shift_notes.csv`.

    Requirement 3 cannot be answered without notes, but requirement 1 can --
    so the breach list still works and the dashboard says why the "why" is blank.
    """
    return pd.DataFrame(columns=[
        "shift_id", "logged_by", "note", "category", "match_score",
        "rules_category", "methods_agree", "source", "suggested_category",
        "who_pays", "suggested_fix",
    ])


def _headline(analysis, classified, costed, top_n) -> dict:
    live = analysis.predictions
    week = costed[costed["week_start"] == analysis.context.week_start]
    failure_notes = classified[classified["who_pays"] == COMPANY_PAYS]

    return {
        "employees": int(len(live)),
        "at_risk": int(min(top_n, (live["risk_score"] > 0).sum())),
        "already_over": int((live["h_sofar"] > cfg.BREACH_TOTAL_HOURS).sum()),
        "hours_so_far": _clean(week["hours"].sum()),
        "wage_cost_so_far": _clean(week["cost"].sum()),
        "premium_cost_so_far": _clean(week["premium_cost"].sum()),
        "notes_total": int(len(classified)),
        "notes_operational_failure": int(len(failure_notes)),
        "notes_client_billable": int((classified["who_pays"] == "client_pays").sum()),
    }


def _at_risk(analysis, employees, sites, top_n) -> list[dict]:
    live = analysis.predictions.head(top_n).merge(employees, on="employee_id", how="left")
    live["site_name"] = live["primary_site_id"].map(sites)
    live["hours_to_spare"] = (cfg.BREACH_TOTAL_HOURS - live["projected_hours"]).round(2)
    columns = [
        "employee_id", "full_name", "role", "primary_site_id", "site_name",
        "shift_pattern", "h_sofar", "p_h_full", "predicted_remaining",
        "projected_hours", "projected_overtime", "hours_to_spare", "risk_score",
        "is_estimated", "unclosed_shifts",
    ]
    return _records(live[[c for c in columns if c in live.columns]])


def _reasons(classified, costed, export) -> list[dict]:
    """Why the hours happened, costed in rands.

    Cost is attributed by joining each note to its shift, so "relief no-shows
    cost R41,000" is derived rather than asserted.
    """
    if classified.empty:
        return []

    joined = classified.merge(
        costed[["shift_id", "cost", "premium_cost", "hours", "week_start"]],
        on="shift_id", how="left",
    )
    grouped = joined.groupby("category", as_index=False).agg(
        notes=("shift_id", "count"),
        hours=("hours", "sum"),
        cost=("cost", "sum"),
        premium_cost=("premium_cost", "sum"),
    )
    grouped["who_pays"] = grouped["category"].map(COST_OF_CATEGORY)
    grouped["suggested_fix"] = grouped["category"].map(FIX_FOR_CATEGORY)
    grouped = grouped.sort_values("cost", ascending=False)
    return _records(grouped)


def _sites(classified, costed, export, sites) -> list[dict]:
    """Where the avoidable cost is concentrated, and what is causing it."""
    joined = (
        classified.merge(
            costed[["shift_id", "site_id", "cost", "hours"]], on="shift_id", how="left"
        ).dropna(subset=["site_id"])
        if not classified.empty
        else pd.DataFrame(columns=["site_id", "cost", "hours", "who_pays", "category"])
    )

    rows = []
    # Every site in the shift data gets a row, even one with no notes, so a
    # missing `shift_notes.csv` leaves the costs visible rather than the site.
    for site_id in sorted(costed["site_id"].dropna().unique()):
        block = joined[joined["site_id"] == site_id]
        failures = block[block["who_pays"] == COMPANY_PAYS]
        by_cause = failures.groupby("category")["cost"].sum().sort_values(ascending=False)
        site_cost = costed.loc[costed["site_id"] == site_id, "cost"].sum()
        rows.append({
            "site_id": site_id,
            "site_name": sites.get(site_id, site_id),
            "total_cost": _clean(site_cost),
            "avoidable_cost": _clean(failures["cost"].sum()),
            "avoidable_hours": _clean(failures["hours"].sum()),
            "billable_cost": _clean(block[block["who_pays"] == "client_pays"]["cost"].sum()),
            "top_cause": by_cause.index[0] if len(by_cause) else None,
            "top_cause_cost": _clean(by_cause.iloc[0]) if len(by_cause) else None,
        })
    return sorted(rows, key=lambda r: r["avoidable_cost"] or 0, reverse=True)


def _data_gaps(analysis, classified, employees) -> dict:
    """What the system cannot see, said out loud.

    The client's own engine drops unclosed shifts, which records them as zero
    hours rather than unknown. Two people this week show 0.00h in their system
    and ~12h in ours. Surfacing that is the point of the product.
    """
    shifts = analysis.shifts
    live_week = shifts[
        (shifts["week_start"] == analysis.context.week_start) & shifts["is_unclosed"]
    ].merge(employees, on="employee_id", how="left")

    note_table = analysis.export.get("shift_notes")
    if note_table is not None:
        live_week["note"] = live_week["shift_id"].map(
            note_table.set_index("shift_id")["note"]
        )
    else:
        live_week["note"] = None

    unclear = classified[classified["category"] == "unclear"]
    suggested = classified["suggested_category"].dropna()

    return {
        "unclosed_shifts": _records(live_week[[
            "shift_id", "employee_id", "full_name", "site_id",
            "shift_date", "clock_in_time", "hours", "note",
        ]]),
        "unclear_notes": int(len(unclear)),
        "unclear_examples": _records(unclear[["shift_id", "note"]].head(10)),
        "suggested_categories": (
            suggested.value_counts().rename_axis("name").reset_index(name="count")
            .to_dict("records")
        ),
        "capped_shifts": int(shifts["is_capped"].sum()),
    }


def _model_card(analysis) -> dict:
    """How well the prediction actually works, including against baselines.

    On the dashboard because a number with no error bar invites more trust than
    it has earned.
    """
    metrics = analysis.metrics or {}
    matched = metrics.get("history_matched", {})
    return {
        "features": analysis.forecast.model.features,
        "typical_error_hours": round(analysis.forecast.model.typical_error, 2),
        "roc_auc": matched.get("roc_auc"),
        "pr_auc": matched.get("pr_auc"),
        "base_rate": matched.get("base_rate"),
        "lift": matched.get("lift"),
        "backtest_weeks": matched.get("n"),
        "backtest_breaches": matched.get("breaches"),
        "top_n": matched.get("top_n", {}).get(10),
        "baselines": analysis.baselines,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.exception_handler(ExportError)
async def export_error_handler(_request, exc: ExportError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# The shipped export never changes, so its analysis is computed once and served
# from memory. Rebuilding it per request cost ~5s, nearly all of it re-classifying
# the same 2,117 notes -- which is exactly what makes a cold start painful on a
# free host. Uploads are unaffected; those always recompute.
_DEFAULT_PAYLOAD: dict | None = None


def default_payload() -> dict:
    global _DEFAULT_PAYLOAD
    if _DEFAULT_PAYLOAD is None:
        payload = build_payload(analyse(load_default_export()))
        assert_no_pii(payload)
        _DEFAULT_PAYLOAD = payload
    return _DEFAULT_PAYLOAD


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "warm": _DEFAULT_PAYLOAD is not None}


@app.get("/api/analyse")
def analyse_default() -> dict:
    """The export that ships with the repo. What the dashboard shows on load."""
    return default_payload()


@app.post("/api/analyse")
async def analyse_upload(files: list[UploadFile] = File(...)) -> dict:
    """Drop in next week's export. No code change, no redeploy, no developer."""
    if not files:
        raise HTTPException(400, "No files were uploaded.")

    contents: dict[str, bytes] = {}
    total = 0
    for upload in files:
        data = await upload.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"That upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)}MB. "
                "Please send the CSV files on their own.",
            )
        contents[upload.filename or "unnamed.csv"] = data

    export = load_export(files=contents)  # ExportError -> 400 with a readable message
    try:
        payload = build_payload(analyse(export))
    except ValueError as exc:
        # e.g. a complete week with nothing left to forecast, or too little history
        raise HTTPException(400, str(exc)) from exc

    assert_no_pii(payload)
    return payload


@app.get("/api/validation")
def validation() -> dict:
    """Evidence that the note sorting works. Not a claim -- a measurement."""
    result = validate(load_default_export())
    scores = {}
    for name, score in result["scores"].items():
        low, high = score.confidence_interval
        scores[name] = {
            "accuracy": score.accuracy,
            "correct": score.correct,
            "n": score.n,
            "ci_low": low,
            "ci_high": high,
            "corpus_reweighted": score.reweighted,
            "errors": _records(score.errors),
        }
    return {
        "n": result["n"],
        "scores": scores,
        "categories": CATEGORIES,
    }
