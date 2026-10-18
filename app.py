"""Step 6 - Streamlit demo: enter a film's characteristics, get a hit/flop probability.

Run:  streamlit run app.py        (after `python train_models.py`)
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
import features  # noqa: F401  (required so joblib can unpickle FeatureBuilder / CareerLookup)

st.set_page_config(page_title="Movie Success Predictor", page_icon="🎬", layout="wide")


@st.cache_resource
def load_artifacts(path):
    return joblib.load(path)


def shap_contributions(model, X):
    """Per-feature SHAP contribution for one row (tree models only)."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    return pd.Series(sv[0], index=X.columns)


files = sorted(config.MODELS_DIR.glob("artifacts_*.joblib"))
st.title("🎬 Movie Commercial Success Predictor")
if not files:
    st.error("No trained model found. Run the pipeline first: `python train_models.py`.")
    st.stop()

with st.sidebar:
    st.header("Model")
    default = next((i for i, p in enumerate(files) if p.stem == "artifacts_roi"), 0)
    choice = st.selectbox("Success definition", files, index=default,
                          format_func=lambda p: p.stem.replace("artifacts_", ""))
    art = load_artifacts(choice)
    st.write(f"**Target:** {art['target_text']}")
    st.write(f"**Algorithm:** {art['model_name']}")
    st.write(f"**Trained on films up to:** {art['trained_through']}")
    st.write(f"**Base rate of success:** {art['base_rate']:.0%}")
    st.caption(f"Validation: {art['metrics']}")
    st.caption("Educational demo. Trained on historical TMDB data (revenue/budget in nominal USD); "
               "probabilities are model estimates, not forecasts.")

lookup, builder = art["lookup"], art["builder"]

c1, c2 = st.columns(2)
with c1:
    st.subheader("The film")
    budget_m = st.number_input("Budget (USD millions)", 0.5, 400.0, 60.0, step=5.0)
    runtime = st.number_input("Runtime (minutes)", 60, 240, 110)
    release = st.date_input("Planned release date", pd.Timestamp.today() + pd.Timedelta(days=180))
    genres = st.multiselect("Genres", art["genres"], default=art["genres"][:1], max_selections=4)
    english = st.checkbox("English-language film", value=True)
    sequel = st.checkbox("Sequel / franchise installment", value=False)
    cert = st.selectbox("MPAA rating", ["Unknown", "G", "PG", "PG-13", "R"], index=3)
with c2:
    st.subheader("The team")
    director = st.selectbox("Director (type to search)", [None] + lookup.director_names, index=0,
                            format_func=lambda x: "Unknown / first-time director" if x is None else x)
    cast = st.multiselect("Top billed actors (up to 3)", lookup.actor_names, max_selections=3)
    studios = st.multiselect("Production companies", lookup.studio_names, max_selections=3)
    st.subheader("Critic reception")
    has_critic = st.checkbox("I have an expected critic score (post-screening model)",
                             disabled=art["critic"] is None)
    critic = st.slider("Expected critic score (0-100)", 0, 100, 65, disabled=not has_critic)
    st.caption("Uses critic scores only (Tomatometer / Metascore style). Audience scores are never used "
               "because they are measured after release.")

if st.button("Predict", type="primary"):
    if not genres:
        st.warning("Pick at least one genre.")
        st.stop()
    row = pd.DataFrame([{
        "budget": budget_m * 1e6, "runtime": float(runtime), "release_date": pd.Timestamp(release),
        "original_language": "en" if english else "xx", "genres_list": genres, "studios_list": studios,
        "is_sequel": int(sequel), "us_certification": None if cert == "Unknown" else cert,
        "critic_score": float(critic) if has_critic else np.nan,
        **lookup.history_row(director, cast, studios),
    }])
    spec = art["critic"] if (has_critic and art["critic"]) else art["greenlight"]
    X = builder.transform(row)[spec["features"]]
    prob = float(spec["model"].predict_proba(X)[0, 1])
    verdict = prob >= spec["threshold"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of success", f"{prob:.0%}", f"{prob - art['base_rate']:+.0%} vs. base rate")
    m2.metric("Verdict", "Likely HIT" if verdict else "Likely FLOP")
    m3.metric("Model used", "Post-screening (+critics)" if spec is art["critic"] else "Greenlight")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Decision threshold {spec['threshold']:.2f} (chosen on cross-validated training data).")

    try:
        contrib = shap_contributions(spec["model"], X).sort_values(key=np.abs, ascending=False).head(8)
        st.subheader("What drives this prediction")
        st.bar_chart(contrib.sort_values())
        st.caption("Positive bars push toward success, negative toward flop (SHAP values, log-odds scale).")
    except Exception as exc:                       # explanations are a bonus, never block the prediction
        st.info(f"Explanation unavailable: {exc}")
