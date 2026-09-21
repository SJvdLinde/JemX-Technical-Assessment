# Ops Room — overtime before Sunday

Predicts which employees will exceed the 10-hour weekly overtime cap by Sunday,
using data that stops on Wednesday, and says what to do about it today.

Live week: **2026-08-10 → 2026-08-16**. Data runs to Wednesday 2026-08-12.

## Deliverables

| File | |
|---|---|
| [`outputs/predictions.csv`](outputs/predictions.csv) | 213 rows, one per employee. `employee_id,will_breach,risk_score` |
| [`outputs/note_classifications.csv`](outputs/note_classifications.csv) | 2,117 rows, one per note. `shift_id,category,note` |
| [`NOTES.md`](NOTES.md) | Assumptions, how the note-sorting was checked, the trained-model question |

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
cd backend && ./run_api.sh          # API on :8000

cd web && npm install && npm run dev # dashboard on :3000
```

Regenerate the deliverables:

```bash
cd backend && python generate_outputs.py --metrics
```

Run the tests (133):

```bash
cd backend && python -m pytest
```

No API key is needed for any of the above.

## Layout

```
backend/app/
  loader.py      reads an export from a directory or an upload
  hours.py       shifts -> hours -> employee-week
  cost.py        hours -> rands (the only place payroll is touched)
  features.py    modelling frame, cut-off day parameterised
  predict.py     the model, backtest, predictions.csv
  notes.py       taxonomy, templates, rules, classifier, cascade
  llm.py         optional LLM fallback (not required; see below)
  validation.py  scoring against the hand-labelled sample
  actions.py     what to do about it
  api.py         HTTP layer
web/             Next.js dashboard
labels/          50 hand-labelled notes + the labelling guide
data/            the client export (synthetic)
```

## How it works, briefly

**The prediction** is a ridge regression on two numbers: the employee's historical
mean weekly hours, and hours worked before the cut-off. It predicts the hours
still to come, then converts that to a probability using the distribution of the
model's own past errors. Backtested by replaying each past week as if it were
Wednesday: **ROC 0.85, PR 0.29 against a 4.9% base rate**, versus 0.72 for the
best naive baseline.

**The notes** are sorted by a cascade: template matching on sentences a language
model read and judged (committed to `notes.py`, so every decision is auditable),
then a character n-gram classifier for unseen typos, then an optional LLM for
genuinely new causes, then `unclear`. Validated against 50 hand labels:
**100%, 95% CI [0.93, 1.00]**.

**Next week's export** uploads through the dashboard. The live week, the cut-off
weekday and the training window are all derived from the files.

## The LLM is optional

Gate 3 of the note cascade calls an LLM only for notes nothing else can place.
On this data that is **zero notes**. Without a key those notes come back
`unclear` and are shown on the dashboard rather than silently mislabelled.

Set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to enable it. Answers are
cached to `labels/llm_cache.json` so output stays reproducible.

## Notes on the data

- Weeks run Monday to Sunday. Overtime is hours above 45; a breach is more than
  10 of those, i.e. a week over 55 hours.
- **184 shifts have no clock-out.** The client's system drops them, recording
  zero hours. We estimate them from each employee's own median shift, which
  reveals 13 breaches their system hides. See `NOTES.md`.
- **No shift anywhere exceeds 13.5h**, so hours are a lower bound.
- `payroll_details.csv` contains bank and tax fields. Only `hourly_rate` is read,
  and a runtime check fails the request if any other field reaches a response.
