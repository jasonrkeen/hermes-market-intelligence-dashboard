# Copilot instructions: Hermès International Market Intelligence & Equity Research Pipeline

## Purpose
Python equity research and market intelligence dashboard for Hermès International, featuring valuation, risk modeling, news/geopolitical intelligence, PDF reporting, and Streamlit analytics.

## Tech stack
- Python; dependencies in `requirements.txt`
- Key libraries: `yfinance`, `pandas`, `numpy`, `matplotlib`, `reportlab`, `streamlit`

## Folder structure
- `config/` (2 files)
- `data/` (12 files)
- `outputs/` (136 files)
- `src/` (22 files)
- `README.md`
- `app.py`
- `main.py`
- `requirements.txt`

## Setup, run, test
Commands documented in README.md (referenced files exist):
```bash
python main.py
python -m streamlit run app.py
```
- No automated tests exist yet; if adding them, use pytest under `tests/`.

## Conventions observed
- Functions use type hints; keep annotating new code.
- Random seeds are set for reproducibility (Monte Carlo/simulation); keep seeded, deterministic runs.
- Public data sources referenced: yfinance.
- Write paths relative to the repo root (no absolute local paths).

## Output artifacts
- Generated outputs go to: `outputs/charts/`, `outputs/reports/`, `outputs/snapshots/`, `outputs/tables/`
- A PDF report is produced with `reportlab`; keep it in sync with model changes.
- Streamlit dashboard: `app.py` (run with `streamlit run <file>`).
- Charts are generated with matplotlib.

## Guardrails
- Never commit secrets, API keys, tokens, or `.env` files.
- Do not commit large raw datasets or binaries; keep data fetched or documented with its source.
- Keep analysis source-backed: cite data sources and assumptions; label scenario/forecast outputs as model estimates, not facts.
- Synthetic data must stay synthetic; never introduce real personal or sensitive records.
- Preserve reproducibility: no unseeded randomness, no hidden manual steps.
