"""End to end: a client export in, answers out.

One entry point so the API, the tests and the CSV writer all run the same code.
Nothing here is specific to the shipped data -- the live week, the cut-off day
and the training window are all derived from whatever was uploaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config as cfg
from .cost import cost_shifts, cost_weekly
from .features import build_features
from .hours import WeekContext, build_shifts, build_weekly, week_context
from .loader import Export, load_default_export, load_export
from .predict import Forecast, baselines, forecast, to_predictions_csv


@dataclass
class Analysis:
    """Everything the dashboard and the deliverables need."""

    export: Export
    shifts: pd.DataFrame
    weekly: pd.DataFrame
    costed: pd.DataFrame
    context: WeekContext
    frame: pd.DataFrame
    forecast: Forecast
    baselines: dict

    @property
    def predictions(self) -> pd.DataFrame:
        return self.forecast.predictions

    @property
    def metrics(self) -> dict:
        return self.forecast.metrics


def analyse(export: Export) -> Analysis:
    shifts = build_shifts(export)
    context = week_context(shifts, export)
    weekly = build_weekly(shifts, employees=export["employees"])
    costed = cost_shifts(shifts, export)

    # The cut-off is whatever weekday the data stops on. Requirement 4 depends
    # on this being derived rather than hardcoded.
    cutoff = int(context.data_through.weekday())
    frame = build_features(shifts, export["employees"], cutoff)

    result = forecast(frame)
    return Analysis(
        export=export,
        shifts=shifts,
        weekly=weekly,
        costed=costed,
        context=context,
        frame=frame,
        forecast=result,
        baselines=baselines(result.backtest, frame),
    )


def analyse_default() -> Analysis:
    return analyse(load_default_export())


def analyse_upload(files) -> Analysis:
    return analyse(load_export(files=files))


def write_predictions(analysis: Analysis, path: str | Path | None = None) -> pd.DataFrame:
    """Write the submission file. Defaults to `outputs/predictions.csv`."""
    path = Path(path) if path is not None else cfg.PREDICTIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_predictions_csv(analysis.forecast, analysis.export["employees"])
    out.to_csv(path, index=False)
    return out
