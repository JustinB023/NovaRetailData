# NovaRetail Customer Intelligence Dashboard (Streamlit)

Single-file Streamlit dashboard to explore customer segments, revenue patterns, and early-warning indicators.

## Files
- `app.py` — Streamlit app
- `NR_dataset.xlsx` — dataset (must be in repo root next to `app.py`)
- `requirements.txt` — dependencies
- `insights.txt` — sample insights (placeholders)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public).
2. Streamlit Community Cloud → **New app**
3. Select repo + branch
4. Main file path: `app.py`
5. Deploy

## Notes
- No external APIs / no secrets.
- Cross-filtering is supported by clicking in charts (uses `streamlit-plotly-events`).
