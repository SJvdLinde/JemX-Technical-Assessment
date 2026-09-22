# JemX Technical Assesment

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
