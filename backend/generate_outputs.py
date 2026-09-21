"""Regenerate the submission deliverables.

    python generate_outputs.py                 # uses data/
    python generate_outputs.py --data path/to  # uses another export
"""

from __future__ import annotations

import argparse
import json

from app import config as cfg
from app.loader import load_export
from app.pipeline import analyse, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(cfg.DATA_DIR), help="export folder")
    parser.add_argument("--metrics", action="store_true", help="print backtest metrics")
    args = parser.parse_args()

    analysis = analyse(load_export(directory=args.data))
    ctx = analysis.context

    print(f"Week in progress : {ctx.week_start.date()} -> {ctx.week_end.date()}")
    print(f"Data through     : {ctx.data_through.date()} "
          f"({ctx.days_elapsed} days done, {ctx.days_remaining} to go)")
    if ctx.public_holidays:
        print(f"Public holidays  : {', '.join(str(d.date()) for d in ctx.public_holidays)}")

    predictions = write_predictions(analysis)
    flagged = int(predictions["will_breach"].sum())
    print(f"\nWrote {cfg.PREDICTIONS_CSV.relative_to(cfg.REPO_ROOT)} "
          f"({len(predictions)} rows, {flagged} flagged)")

    top = analysis.predictions.merge(
        analysis.export["employees"][["employee_id", "full_name", "primary_site_id"]],
        on="employee_id",
    ).head(5)
    print("\nHighest risk this week:")
    for _, row in top.iterrows():
        print(f"  {row.risk_score:.2f}  {row.full_name:20} {row.primary_site_id}  "
              f"{row.h_sofar:5.1f}h so far -> {row.projected_hours:5.1f}h projected")

    if args.metrics:
        print("\nBacktest:")
        print(json.dumps(analysis.metrics, indent=2, default=str))
        print("\nBaselines:")
        print(json.dumps(analysis.baselines, indent=2))


if __name__ == "__main__":
    main()
