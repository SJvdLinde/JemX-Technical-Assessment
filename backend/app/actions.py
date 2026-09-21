"""What the contract manager should actually do today.

Requirement 2: "Something the contract manager could act on without opening a
spreadsheet. A suggestion that could have been written without looking at the
data is worth nothing to them."

So every field here is derived. "Monitor overtime closely" is not an action.
"Take Portia off one shift this weekend and give it to Thabo, who is at 19h"
is an action, and both names and both numbers come from the data.

Two kinds of recommendation:

  Per person   How many hours to remove, and who at the same site could absorb
               them without going over themselves.
  Systemic     Where avoidable cost is concentrated and what is causing it --
               the thing that stops next week looking like this one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .notes import COMPANY_PAYS, FIX_FOR_CATEGORY

# Aim below the cap rather than at it -- landing someone on exactly 55.0h leaves
# no room for the model's ~3.5h typical error.
SAFETY_MARGIN_HOURS = 2.0

# A swap candidate must stay this far under the cap after taking the work on.
CANDIDATE_HEADROOM_HOURS = 6.0


def recommend(analysis, top_n: int = 15) -> list[dict]:
    """One recommendation per at-risk employee."""
    live = analysis.predictions
    employees = analysis.export["employees"].set_index("employee_id")
    sites = (
        analysis.export["sites"].set_index("site_id")["site_name"]
        if analysis.export.get("sites") is not None
        else pd.Series(dtype=object)
    )

    # Everyone's projected position this week, so we can find spare capacity.
    projected = live.set_index("employee_id")["projected_hours"]
    typical_error = float(analysis.forecast.model.typical_error)

    out = []
    for _, row in live.head(top_n).iterrows():
        employee_id = row["employee_id"]
        person = employees.loc[employee_id]
        target = cfg.BREACH_TOTAL_HOURS - SAFETY_MARGIN_HOURS
        excess = float(row["projected_hours"]) - target

        shift_length = _typical_shift(analysis.shifts, employee_id)
        shifts_to_cut = int(np.ceil(excess / shift_length)) if excess > 0 else 0

        out.append({
            "employee_id": employee_id,
            "full_name": person["full_name"],
            "role": person["role"],
            "site_id": person["primary_site_id"],
            "site_name": sites.get(person["primary_site_id"], person["primary_site_id"]),
            "risk_score": float(row["risk_score"]),
            "hours_so_far": float(row["h_sofar"]),
            "projected_hours": float(row["projected_hours"]),
            "hours_over_target": round(max(excess, 0.0), 2),
            "shifts_to_cut": shifts_to_cut,
            "typical_shift_hours": round(shift_length, 2),
            "action": _sentence(person, row, excess, shifts_to_cut, typical_error),
            # "Needs action" is not just "projected over". The forecast is
            # typically out by ~3.5h, so anyone sitting inside that margin is one
            # unplanned shift from a breach and is worth a decision today.
            # Suggesting a reshuffle for someone comfortably clear is noise, and
            # noise is what stops a ten-minute dashboard being used.
            "cover_options": (
                _cover_options(
                    analysis, employees, projected, employee_id, person, shift_length
                )
                if _needs_action(row, excess, typical_error)
                else []
            ),
            "needs_action": _needs_action(row, excess, typical_error),
            "headroom_hours": round(
                cfg.BREACH_TOTAL_HOURS - float(row["projected_hours"]), 2
            ),
        })
    return out


def _needs_action(row, excess: float, typical_error: float) -> bool:
    """Projected over, or close enough that one shift would do it."""
    headroom = cfg.BREACH_TOTAL_HOURS - float(row["projected_hours"])
    return excess > 0 or headroom <= typical_error


def _typical_shift(shifts: pd.DataFrame, employee_id: str) -> float:
    """This person's median shift, which is what a cut actually removes."""
    theirs = shifts.loc[
        (shifts["employee_id"] == employee_id) & shifts["hours"].notna(), "hours"
    ]
    if theirs.empty:
        return 9.5
    return float(np.clip(theirs.median(), 4.0, cfg.SHIFT_LENGTH_CEILING))


def _sentence(person, row, excess: float, shifts_to_cut: int, typical_error: float) -> str:
    """The instruction, and nothing else.

    The name, hours worked and projection are separate fields, so the dashboard
    lays them out as figures rather than burying them in a sentence. What is
    left here is the decision: cut something, check something, or add nothing.
    """
    projected = row["projected_hours"]

    if excess > 0:
        plural = "shift" if shifts_to_cut == 1 else "shifts"
        return f"Cut {shifts_to_cut} {plural} ({excess:.1f}h) before Sunday."

    headroom = cfg.BREACH_TOTAL_HOURS - projected
    if headroom <= typical_error:
        return (
            f"{headroom:.1f}h of room, forecast error {typical_error:.1f}h. "
            f"One extra shift breaches. Check the weekend roster."
        )
    return f"{headroom:.1f}h of room. Add no extra hours."


def _cover_options(analysis, employees, projected, employee_id, person, shift_length):
    """Who could take the shift instead, without going over themselves.

    Same site and same role, because a guard post needs a guard. Ranked by how
    much room they have left, so the suggestion is the safest one available.
    """
    same_team = employees[
        (employees["primary_site_id"] == person["primary_site_id"])
        & (employees["role"] == person["role"])
    ].index.drop(employee_id, errors="ignore")

    candidates = projected.reindex(same_team).dropna()
    room = cfg.BREACH_TOTAL_HOURS - CANDIDATE_HEADROOM_HOURS - shift_length
    viable = candidates[candidates <= room].sort_values().head(3)

    return [
        {
            "employee_id": candidate_id,
            "full_name": employees.loc[candidate_id, "full_name"],
            "projected_hours": round(float(hours), 2),
            "projected_after_swap": round(float(hours) + shift_length, 2),
        }
        for candidate_id, hours in viable.items()
    ]


def systemic(analysis, classified, top_n: int = 4) -> list[dict]:
    """What to fix so next week is different.

    Ranked by the rand cost of avoidable overtime, per cause and per site, so
    the manager can see where the money actually goes rather than a list of
    categories.
    """
    if classified.empty:
        return []

    joined = classified.merge(
        analysis.costed[["shift_id", "site_id", "cost", "hours"]],
        on="shift_id", how="left",
    )
    failures = joined[joined["who_pays"] == COMPANY_PAYS].dropna(subset=["cost"])
    if failures.empty:
        return []

    sites = (
        analysis.export["sites"].set_index("site_id")["site_name"]
        if analysis.export.get("sites") is not None
        else pd.Series(dtype=object)
    )

    out = []
    by_cause = failures.groupby("category")["cost"].sum().sort_values(ascending=False)
    for category, cost in by_cause.head(top_n).items():
        block = failures[failures["category"] == category]
        worst_site = block.groupby("site_id")["cost"].sum().sort_values(ascending=False)
        site_id = worst_site.index[0] if len(worst_site) else None
        share = cost / failures["cost"].sum()

        out.append({
            "category": category,
            "cost": round(float(cost), 2),
            "share_of_avoidable": round(float(share), 3),
            "hours": round(float(block["hours"].sum()), 1),
            "notes": int(len(block)),
            "worst_site_id": site_id,
            "worst_site_name": sites.get(site_id, site_id),
            "worst_site_cost": round(float(worst_site.iloc[0]), 2) if site_id else None,
            "fix": FIX_FOR_CATEGORY.get(category),
            # Figures, not prose. The dashboard lays these out as columns.
            "sentence": (
                f"R{cost:,.0f} · {share:.0%} of avoidable overtime · "
                f"worst at {sites.get(site_id, site_id)} R{worst_site.iloc[0]:,.0f}"
            ) if site_id else None,
        })
    return out
