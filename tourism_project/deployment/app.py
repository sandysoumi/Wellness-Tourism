"""
Streamlit App - Wellness Tourism Package Purchase Predictor

Loads the champion model that the GitHub Actions pipeline committed to the
repository, collects a customer's details into a DataFrame and predicts
whether they will buy the Wellness Tourism Package. Streamlit Community
Cloud redeploys this app automatically whenever the pipeline pushes a new
model to the main branch.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
METRICS_PATH = APP_DIR / "metrics.json"
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

st.set_page_config(page_title="Wellness Tourism Purchase Predictor",
                   page_icon="🧳", layout="wide")

# ── Custom styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 600; }
    div[data-testid="stMetric"] {
        background-color: rgba(46, 158, 91, 0.08);
        border: 1px solid rgba(46, 158, 91, 0.25);
        border-radius: 10px; padding: 12px 16px;
    }
    .app-header {
        padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 1.2rem; color: white;
        background: linear-gradient(135deg, #2E9E5B 0%, #1B6B3E 100%);
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; color: white; }
    .app-header p { margin: 4px 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Loading ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    """Model bundle = fitted pipeline + tuned threshold + feature lists."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    try:
        return json.loads(METRICS_PATH.read_text())
    except FileNotFoundError:
        return None


def category_options(bundle):
    """Dropdown options read from the trained encoder, so they always match
    the categories the model actually saw during training."""
    encoder = (bundle["model"].named_steps["preprocessor"]
               .named_transformers_["cat"].named_steps["onehot"])
    return {col: list(levels) for col, levels in
            zip(bundle["categorical_features"], encoder.categories_)}


def select(label, col, preferred):
    """Dropdown for a categorical feature, defaulting to a typical value."""
    opts = options[col]
    return st.selectbox(label, opts, index=opts.index(preferred) if preferred in opts else 0)


def prepare(df: pd.DataFrame, features) -> pd.DataFrame:
    """Same light cleaning as training; missing columns become NaN and are
    filled by the pipeline's imputers."""
    df = df.copy()
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()
    if "Gender" in df:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    for col in features:
        if col not in df:
            df[col] = np.nan
    return df[features]


def gauge(probability, threshold):
    color = "#2E9E5B" if probability >= threshold else "#C4442E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, threshold * 100], "color": "rgba(196, 68, 46, 0.15)"},
                         {"range": [threshold * 100, 100], "color": "rgba(46, 158, 91, 0.15)"}],
               "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8,
                             "value": threshold * 100}},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    return fig


if not MODEL_PATH.exists():
    st.error("No trained model found. Run the GitHub Actions pipeline first.")
    st.stop()

bundle = load_bundle()
model, threshold, features = bundle["model"], bundle["threshold"], bundle["features"]
options = category_options(bundle)
metrics_info = load_metrics()

# ── Sidebar: model card ────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write("Predicts whether a customer is likely to purchase **Visit with Us's** new "
             "**Wellness Tourism Package**, so the sales team can prioritise outreach.")
    st.subheader("Champion model")
    st.write(f"**Algorithm:** {bundle['model_name']}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    if metrics_info:
        m = metrics_info["metrics"]
        col_a, col_b = st.columns(2)
        col_a.metric("F1 score", f"{m['f1']:.3f}")
        col_b.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        col_a.metric("Precision", f"{m['precision']:.3f}")
        col_b.metric("Recall", f"{m['recall']:.3f}")
        st.caption("Held-out test-set scores. Chosen from 6 tuned algorithms by "
                   f"cross-validated F1 · trained {metrics_info.get('trained_at_utc', '')} UTC")
    st.divider()
    st.caption("🔄 Part of an automated MLOps pipeline: GitHub Actions validates the data, "
               "retrains and tracks the models in MLflow, and commits the champion here.")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧳 Wellness Tourism Package — Purchase Predictor</h1>
    <p>Predict whether a customer is likely to buy, before your sales team makes contact.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_batch, tab_insights = st.tabs(
    ["🔮 Predict", "📋 Batch scoring", "📊 Model insights"])

# ── Tab 1: single prediction ───────────────────────────────────────────────
with tab_predict:
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👤 Customer profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = select("Gender", "Gender", "Male")
            marital_status = select("Marital status", "MaritalStatus", "Married")
            occupation = select("Occupation", "Occupation", "Salaried")
            designation = select("Designation", "Designation", "Executive")
            monthly_income = st.number_input("Monthly income", 1000, 200000, 22000, step=1000)
            st.caption("Typical range in training data: 1,000 – 98,000")
            passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
            own_car = st.selectbox("Owns a car?", ["Yes", "No"])
        with col2:
            st.subheader("🏖️ Trip & pitch details")
            city_tier = st.selectbox("City tier", [1, 2, 3])
            typeof_contact = select("Type of contact", "TypeofContact", "Self Enquiry")
            preferred_star = st.selectbox("Preferred property star", [3, 4, 5])
            num_persons = st.number_input("Number of persons visiting", 1, 10, 3)
            num_children = st.number_input("Children visiting (under 5)", 0, 5, 1)
            num_trips = st.number_input("Trips per year (average)", 1, 30, 3)
            duration = st.number_input("Duration of pitch (minutes)", 1, 180, 15)
            followups = st.number_input("Number of follow-ups", 0, 10, 3)
            satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Collect the inputs into a single-row DataFrame.
        input_df = pd.DataFrame([{
            "Age": age, "TypeofContact": typeof_contact, "CityTier": city_tier,
            "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
            "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": followups,
            "PreferredPropertyStar": preferred_star, "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips, "Passport": int(passport == "Yes"),
            "PitchSatisfactionScore": satisfaction, "OwnCar": int(own_car == "Yes"),
            "NumberOfChildrenVisiting": num_children, "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])
        st.divider()
        st.write("**Customer data sent to the model**")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

        probability = float(model.predict_proba(prepare(input_df, features))[:, 1][0])
        result_col, gauge_col = st.columns(2)
        with result_col:
            st.subheader("Prediction result")
            if probability >= threshold:
                st.success("✅ **Likely to PURCHASE** the Wellness Tourism Package")
                st.balloons()
            else:
                st.warning("❌ **Unlikely to purchase** the Wellness Tourism Package")
            st.metric("Purchase probability", f"{probability:.1%}")
            st.caption(f"Customers at or above {threshold:.0%} are flagged as likely buyers. "
                       "The prediction supports, not replaces, the sales team's judgement.")
        with gauge_col:
            st.plotly_chart(gauge(probability, threshold), use_container_width=True)

# ── Tab 2: batch scoring ───────────────────────────────────────────────────
with tab_batch:
    st.write("Upload a CSV of customers (columns as in the data dictionary). The app "
             "scores every row and returns a **ranked call list**, highest probability first.")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        raw = pd.read_csv(upload)
        proba = model.predict_proba(prepare(raw, features))[:, 1]
        scored = raw.assign(PurchaseProbability=proba.round(4),
                            PredictedBuyer=(proba >= threshold).astype(int))
        scored = scored.sort_values("PurchaseProbability", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers scored", f"{len(scored):,}")
        c2.metric("Likely buyers", f"{int(scored['PredictedBuyer'].sum()):,}")
        c3.metric("Share flagged", f"{scored['PredictedBuyer'].mean():.1%}")
        st.dataframe(scored, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download ranked list", scored.to_csv(index=False),
                           file_name="scored_customers.csv", mime="text/csv")

# ── Tab 3: model insights ──────────────────────────────────────────────────
with tab_insights:
    if metrics_info:
        st.subheader("How the champion model was chosen")
        st.write("Six algorithms were tuned with `GridSearchCV` (stratified 5-fold CV) and every "
                 "parameter combination was tracked in MLflow. The champion, "
                 f"**{metrics_info['best_model']}**, has the best **cross-validated** F1; the "
                 "held-out test set is used only to report final performance.")
        results = pd.DataFrame(metrics_info["all_results"])
        table = results[["model", "cv_f1", "threshold"] + METRIC_COLS].rename(
            columns={"model": "Model", "cv_f1": "CV F1", "threshold": "Threshold"})
        st.dataframe(table.style.highlight_max(subset=["CV F1"] + METRIC_COLS, color="#d4f4dd")
                     .format({c: "{:.3f}" for c in ["CV F1", "Threshold"] + METRIC_COLS}),
                     use_container_width=True, hide_index=True)

        fig = go.Figure([go.Bar(name=m, x=results["model"], y=results[m]) for m in METRIC_COLS])
        fig.update_layout(barmode="group", title="Test-set metrics by model", height=420,
                          yaxis=dict(range=[0, 1]),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        cm = metrics_info.get("confusion_matrix")
        if cm:
            cm_fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: No", "Predicted: Yes"],
                                          y=["Actual: No", "Actual: Yes"], text=cm,
                                          texttemplate="%{text}", colorscale="Greens",
                                          showscale=False))
            cm_fig.update_layout(title=f"Confusion matrix — {metrics_info['best_model']} (test set)",
                                 height=330, yaxis=dict(autorange="reversed"))
            st.plotly_chart(cm_fig, use_container_width=True)
        st.caption(f"Winning hyperparameters: `{metrics_info.get('best_params', {})}`")
    else:
        st.info("metrics.json not found — comparison unavailable.")
