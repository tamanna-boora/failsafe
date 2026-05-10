"""
app.py — FAILSAFE Streamlit Dashboard
Predicts student failure risk using XGBoost + SHAP explanations,
and surfaces personalised intervention recommendations.

Run from project root:
    streamlit run app/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import shap
import joblib
import streamlit as st

# ── Import interventions from src/ ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from interventions import generate_interventions

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "students.csv")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAILSAFE — Student Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high   { background:#ffe0e0; border-left:4px solid #d32f2f; padding:8px 12px; border-radius:4px; }
    .risk-medium { background:#fff8e1; border-left:4px solid #f9a825; padding:8px 12px; border-radius:4px; }
    .risk-low    { background:#e8f5e9; border-left:4px solid #388e3c; padding:8px 12px; border-radius:4px; }
    .metric-box  { background:#f0f2f6; border-radius:8px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load model artifacts (cached so they load once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    model_path    = os.path.join(MODELS_DIR, "xgb_model.joblib")
    encoder_path  = os.path.join(MODELS_DIR, "label_encoders.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    metrics_path  = os.path.join(MODELS_DIR, "metrics.joblib")

    missing = [p for p in [model_path, encoder_path, features_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, None

    model         = joblib.load(model_path)
    encoders      = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)
    metrics       = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}
    return model, encoders, feature_names, metrics


@st.cache_resource(show_spinner="Building SHAP explainer …")
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: preprocess uploaded / sample data
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, encoders: dict, feature_names: list) -> pd.DataFrame:
    """
    Apply saved label encoders to categorical columns and return only
    the columns the model was trained on, in the correct order.
    Unknown categories are mapped to 0 (safe fallback for tree models).
    """
    df = df.copy()

    # Drop columns the model doesn't need
    for col in ["G3", "at_risk"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode categoricals using saved encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    # Reindex to match training feature order, fill any missing cols with 0
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helper: SHAP waterfall plot for one student
# ══════════════════════════════════════════════════════════════════════════════
def shap_waterfall(explainer, X_row: pd.DataFrame, max_display: int = 10):
    """Return a matplotlib Figure with the SHAP waterfall for one student."""
    explanation = explainer(X_row)  # Explanation object for 1 row
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.sca(ax)
    shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Helper: get top positive-SHAP features for a student (driving at-risk = 1)
# ══════════════════════════════════════════════════════════════════════════════
def get_top_risk_features(explainer, X_row: pd.DataFrame, n: int = 10) -> list[str]:
    """Return feature names sorted by SHAP value (descending), positive first."""
    explanation = explainer(X_row)
    shap_vals   = explanation[0].values          # 1-D array
    feature_names = list(X_row.columns)
    # Sort by raw SHAP value (positive = increases at-risk prediction)
    sorted_idx  = np.argsort(shap_vals)[::-1]
    return [feature_names[i] for i in sorted_idx[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# Risk level labels
# ══════════════════════════════════════════════════════════════════════════════
def risk_label(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    return "Low"

RISK_COLOR = {"High": "#d32f2f", "Medium": "#f9a825", "Low": "#388e3c"}
RISK_CSS   = {"High": "risk-high",  "Medium": "risk-medium",  "Low": "risk-low"}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎓 FAILSAFE")
        st.caption("Student Failure Risk Predictor")
        st.divider()

        model, encoders, feature_names, metrics = load_artifacts()

        if model is not None and metrics:
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                st.caption(f"Confusion Matrix — TP:{cm[1,1]} FP:{cm[0,1]} FN:{cm[1,0]} TN:{cm[0,0]}")

        st.divider()
        st.subheader("Risk Thresholds")
        st.markdown("🔴 **High** — score ≥ 60%")
        st.markdown("🟡 **Medium** — 30% – 59%")
        st.markdown("🟢 **Low** — score < 30%")
        st.divider()
        st.caption("IIT Guwahati Coding Club · Even Semester 2026")

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎓 FAILSAFE — Student Failure Risk Prediction")
    st.markdown(
        "An **Explainable AI** dashboard that identifies at-risk students "
        "and recommends personalised academic interventions. "
        "Powered by **XGBoost** + **SHAP**."
    )

    # ── Model not trained yet ─────────────────────────────────────────────────
    if model is None:
        st.error(
            "Model artifacts not found in `models/`. "
            "Please run the following commands first:\n"
            "```\n"
            "python download_data.py\n"
            "python src/train_model.py\n"
            "```"
        )
        return

    explainer = get_explainer(model)

    # ── Data Input ────────────────────────────────────────────────────────────
    st.header("1. Load Student Data")
    data_source = st.radio(
        "Data source",
        ["Use built-in sample data", "Upload a CSV file"],
        horizontal=True,
    )

    df_raw = None

    if data_source == "Use built-in sample data":
        if not os.path.exists(DATA_PATH):
            st.error("Sample data not found. Run `python download_data.py` first.")
            return
        full_df = pd.read_csv(DATA_PATH)
        # Use last 30 rows as demo (unseen-looking sample)
        df_raw = full_df.tail(30).reset_index(drop=True)
        st.success(f"Loaded {len(df_raw)} students from the built-in dataset.")

    else:
        uploaded = st.file_uploader(
            "Upload a CSV with student features (same format as students.csv)",
            type="csv",
        )
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Uploaded {len(df_raw)} student records.")
        else:
            st.info("Awaiting CSV upload. The file should contain the same feature columns as the training data.")
            st.markdown(
                "**Expected columns (excerpt):** `school, sex, age, address, famsize, Pstatus, "
                "Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, "
                "schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, "
                "famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2`"
            )
            return

    # ── Preview raw data ─────────────────────────────────────────────────────
    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df_raw, use_container_width=True)

    # ── Run Predictions ───────────────────────────────────────────────────────
    st.header("2. Predictions")

    X_processed = preprocess(df_raw, encoders, feature_names)
    probas      = model.predict_proba(X_processed)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    results_df = df_raw.copy()
    results_df["Risk Score (%)"] = (probas * 100).round(1)
    results_df["Risk Level"]     = [risk_label(p) for p in probas]
    results_df["Predicted"]      = ["At-Risk" if p == 1 else "Safe" for p in predictions]

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    n_total    = len(results_df)
    n_at_risk  = int(predictions.sum())
    n_high     = (results_df["Risk Level"] == "High").sum()
    n_medium   = (results_df["Risk Level"] == "Medium").sum()
    n_low      = (results_df["Risk Level"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", n_total)
    col2.metric("At-Risk (≥50%)", n_at_risk, delta=f"{100*n_at_risk/n_total:.0f}%", delta_color="inverse")
    col3.metric("High Risk", n_high)
    col4.metric("Medium Risk", n_medium)

    # ── Filter ────────────────────────────────────────────────────────────────
    st.subheader("Filter by Risk Level")
    risk_filter = st.multiselect(
        "Show students with risk level:",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    filtered_df = results_df[results_df["Risk Level"].isin(risk_filter)]

    # ── Results table ─────────────────────────────────────────────────────────
    display_cols = ["Risk Score (%)", "Risk Level", "Predicted", "absences", "studytime",
                    "failures", "G1", "G2"]
    available    = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Risk Score (%)": st.column_config.ProgressColumn(
                "Risk Score (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Risk Level": st.column_config.TextColumn("Risk Level"),
        },
    )

    # ── Visualisations ─────────────────────────────────────────────────────────
    st.header("3. Summary Visualisations")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("Risk Distribution")
        risk_counts = results_df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        colors = [RISK_COLOR["High"], RISK_COLOR["Medium"], RISK_COLOR["Low"]]
        wedges, texts, autotexts = ax_pie.pie(
            risk_counts,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax_pie.set_title("Student Risk Distribution")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with viz_col2:
        st.subheader("Top Global Risk Factors (Feature Importance)")
        # Use XGBoost's built-in feature importance
        importance = model.feature_importances_
        imp_series = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(10)
        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
        imp_series[::-1].plot(kind="barh", ax=ax_bar, color="#1976d2")
        ax_bar.set_title("Top 10 Feature Importances")
        ax_bar.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # ── At-Risk Student Deep Dive ─────────────────────────────────────────────
    at_risk_mask = results_df["Risk Level"].isin(["High", "Medium"]) & results_df["Risk Level"].isin(risk_filter)
    at_risk_rows = results_df[at_risk_mask]

    if len(at_risk_rows) == 0:
        st.info("No at-risk students to show based on current filter.")
        return

    st.header(f"4. At-Risk Student Analysis ({len(at_risk_rows)} students)")
    st.markdown(
        "Expand each student card to see an **AI explanation** (SHAP waterfall) "
        "and personalised **intervention recommendations**."
    )

    # Limit deep-dive to avoid overwhelming the page
    MAX_DEEP_DIVE = 20
    if len(at_risk_rows) > MAX_DEEP_DIVE:
        st.warning(f"Showing deep-dive for the top {MAX_DEEP_DIVE} highest-risk students.")
        at_risk_rows = at_risk_rows.nlargest(MAX_DEEP_DIVE, "Risk Score (%)")

    for idx, (orig_idx, row) in enumerate(at_risk_rows.iterrows()):
        score = row["Risk Score (%)"]
        level = row["Risk Level"]
        css   = RISK_CSS[level]

        with st.expander(
            f"Student #{orig_idx + 1}  ·  Risk: {score:.1f}%  ·  Level: {level}",
            expanded=(idx == 0),  # auto-open the first card
        ):
            st.markdown(f'<div class="{css}"><b>{level} Risk Student</b> — predicted risk score: {score:.1f}%</div>',
                        unsafe_allow_html=True)
            st.write("")

            tab_explain, tab_actions, tab_profile = st.tabs(
                ["📊 SHAP Explanation", "💡 Interventions", "👤 Profile"]
            )

            with tab_explain:
                st.markdown("**Why is this student flagged as at-risk?**")
                st.caption(
                    "Bars pointing right increase risk; bars pointing left decrease risk. "
                    "The longer the bar, the stronger the influence."
                )
                try:
                    X_one = X_processed.iloc[[orig_idx]]
                    fig_shap = shap_waterfall(explainer, X_one, max_display=10)
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")

            with tab_actions:
                st.markdown("**Recommended Interventions**")
                try:
                    X_one_row = X_processed.iloc[[orig_idx]]
                    top_feats = get_top_risk_features(explainer, X_one_row, n=10)
                    student_vals = X_processed.iloc[orig_idx].to_dict()
                    interventions = generate_interventions(student_vals, top_feats)

                    for i, action in enumerate(interventions, 1):
                        st.markdown(f"**{i}.** {action}")
                except Exception as e:
                    st.warning(f"Could not generate interventions: {e}")

            with tab_profile:
                st.markdown("**Student Feature Values**")
                # Show original (un-encoded) values where available
                profile_data = {}
                for feat in feature_names:
                    if feat in df_raw.columns:
                        profile_data[feat] = df_raw.iloc[orig_idx][feat]
                profile_series = pd.Series(profile_data)
                st.dataframe(profile_series.to_frame("Value"), use_container_width=True)


if __name__ == "__main__":
    main()
