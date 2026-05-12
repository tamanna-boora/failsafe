# What I learned building FAILSAFE

## Where I started

Before this project, I knew enough Python to write scripts and had used pandas for basic data cleaning in a course assignment. That was pretty much it. I'd never used git properly, never trained a real ML model end to end, and had no idea what SHAP or XGBoost even were. FAILSAFE was my first time taking a dataset, building a model, and turning it into something someone could actually open in a browser.

## What I picked up along the way

- **Virtual environments** — how to set one up, why it matters, and what happens when you don't (dependency hell)
- **Git basics** — init, add, commit, push, reading diffs; got burned once by not committing early enough
- **XGBoost** — why it handles tabular data with mixed types better than logistic regression, what `scale_pos_weight` does for imbalanced classes
- **SHAP waterfall plots** — what they actually show (not just feature importance, but per-prediction attribution), and why the base value matters
- **Streamlit** — building a multi-section data app without touching HTML or JavaScript; faster than I expected
- **sklearn metrics** — accuracy alone is misleading when classes are imbalanced; F1 and recall on the minority class matter more here
- **PyArrow serialization** — mixed-type columns in a DataFrame will silently break `st.dataframe()`; convert to string first
- **Writing a README** — the kind someone else could actually follow, not just notes to future-me

## Things that surprised me

Setting up the environment and debugging the data pipeline took almost as long as writing the model itself. I assumed the ML part would be the hard bit, but most of the time went into connecting pieces — getting the encoder to save and reload correctly, making SHAP work inside Streamlit, fixing the matplotlib background.

SHAP surprised me genuinely. I expected a feature importance bar chart — instead, waterfall plots show *why this specific student* got flagged, not just what the model cares about globally. That felt meaningful. It made the model feel less like a black box and more like something a faculty member could trust.

The dataset is from Portuguese secondary schools, which is obviously not IITG. But the patterns — absences predicting failure, early period grades being the strongest signal — felt universal enough that the project still makes sense as a proof of concept.

## What I'd do differently

- Start with a minimal working version first, then add features — I spent too long trying to build everything at once
- Set up git from day one, not after the folder already had 20 files in it
- Test on a fresh clone of the repo before submission — assumptions about local paths can break things
- Sketch the dashboard layout on paper before writing any Streamlit code

## What's next

The Streamlit version is fine for a demo but not for real deployment. A proper version would have a FastAPI backend, a React frontend, and a PostgreSQL database storing predictions and intervention logs over time. I'd also want to integrate real IITG data — CGPA, attendance, maybe mess and hostel records — and retrain the model on that. The interventions are currently rule-based; with more data, a proper recommendation layer would be possible.
