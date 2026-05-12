# notes

## dataset
- UCI student performance dataset, id=320, fetched via `ucimlrepo`
- 649 students from portuguese secondary schools
- target = G3 < 10 (out of 20) → at-risk label
- ~15% positive class → imbalanced but not terrible, manageable with scale_pos_weight
- G1, G2 are period grades kept as features — they're legitimately available mid-year
- G3 is the thing we're predicting from, so we drop it or it's perfect leakage

## model choice
- thought about logistic regression but xgboost handles mixed categorical + numeric
  without needing one-hot encoding or imputation
- gives feature importance basically for free
- scale_pos_weight = neg/pos fixes the "predict everything as safe" problem that
  happens with imbalanced data
- 89% accuracy, F1 ~0.70 on held-out test — decent for an MVP
- what matters most is recall on the at-risk class (don't miss kids who actually need help)
- tried a few hyperparams manually, nothing fancy, 200 trees with depth 4 seems stable

## SHAP
- TreeExplainer is fast for XGBoost, works well
- waterfall plots show per-student attribution, not global importance
- G1 and G2 dominate — makes sense, early grades predict finals
- without G1/G2 the model would be less accurate but more useful for *early* intervention
  (before first exam) — worth thinking about for a future version
- faculty won't know what a SHAP waterfall is → added caption explaining left/right bars
- caching the explainer with @st.cache_resource otherwise it rebuilds every rerun

## interventions
- rule-based on raw feature values, not ML-generated
- rules check thresholds: absences > 10, studytime < 2, failures > 0, etc.
- LabelEncoder sorts alphabetically so "no" → 0, "yes" → 1 — easy to forget this
- SHAP top features used to prioritize which rules fire first
- fallback list in case no rules match (shouldn't happen often but safety net)
- judges will probably ask why not use a recommendation model → "rules are interpretable,
  auditable, and work with n=649. an LLM or collaborative filter needs more data and
  is overkill for a demo"

## things I wrestled with
- pyarrow serialization: Profile tab crashed because the Value column had mixed
  python types (int + str). fix = .astype(str) before st.dataframe()
- use_container_width=True: deprecated in streamlit 1.57, replaced by width='stretch'
  (which is actually the new default, so the param was redundant anyway)
- matplotlib on dark theme: charts rendered white boxes on dark background.
  fix = fig.patch.set_alpha(0.0) + ax.set_facecolor("none") + manually set all
  text/spine colors to #FAFAFA / #888888

## todo / future ideas
- "what-if" risk slider: show how risk changes if attendance improves by N%
- risk trajectory over time if we had G1 → G2 → G3 progression data
- real IITG data: CGPA, attendance records, club activity, hostel data
- proper FastAPI + React version with JWT auth and per-faculty dashboards
- retrain the model periodically as new semester data comes in
