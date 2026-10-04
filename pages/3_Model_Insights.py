import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_predictor

st.set_page_config(page_title="Model Insights - IPL Predictor", layout="wide", page_icon="🤖")

st.markdown("""
<style>
.section-card {
    background-color: #101D2F;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #1C314D;
    margin-bottom: 1.5rem;
}
.metric-title {
    color: #A0AAB5;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    color: #FFFFFF;
    font-size: 2.5rem;
    font-weight: bold;
}
.pipeline-step {
    background-color: #14243A;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border-left: 4px solid #1E88E5;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

st.title("🤖 Model Insights")
st.markdown("View verified model performance and project pipeline details.")
st.page_link("app.py", label="← Back to Home", icon="🏠")
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Model Architecture")
    st.markdown('<div class="section-card"><h2 style="margin: 0; color: #00E676;">XGBoost Regressor</h2></div>', unsafe_allow_html=True)
    
    st.markdown("### Verified Test Performance")
    st.markdown('<p style="color: #A0AAB5;">Evaluated on a chronologically held-out test set to avoid temporal leakage.</p>', unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        with st.container(border=True):
            st.markdown('<div class="metric-title">MAE</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">22.68</div>', unsafe_allow_html=True)
    with mc2:
        with st.container(border=True):
            st.markdown('<div class="metric-title">RMSE</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">30.01</div>', unsafe_allow_html=True)
    with mc3:
        with st.container(border=True):
            st.markdown('<div class="metric-title">MAPE</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">12.07%</div>', unsafe_allow_html=True)

    st.markdown("### Project Pipeline")
    steps = [
        "1. Raw Data (IPL.csv)",
        "2. Data Cleaning & Normalization",
        "3. Leakage-Safe Historical Features",
        "4. XGBoost Model Training",
        "5. Real-Time Prediction Layer",
        "6. Dashboard UI"
    ]
    for step in steps:
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### Feature Importance")
    st.markdown('<p style="color: #A0AAB5;">Top 10 most influential features extracted from the trained model.</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        try:
            xgb_model = predictor.pipeline.named_steps['model']
            preprocessor = predictor.pipeline.named_steps['preprocessor']
            from scripts.predict import CATEGORICAL_FEATURES, LIVE_FEATURES, HISTORICAL_FEATURES
            cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            all_feature_names = cat_features_out + LIVE_FEATURES + HISTORICAL_FEATURES
            importances = xgb_model.feature_importances_
            
            feat_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#101D2F')
            ax.set_facecolor('#101D2F')
            
            ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#00E676', height=0.6)
            ax.tick_params(colors='#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1C314D')
            ax.spines['left'].set_color('#1C314D')
            ax.set_xlabel("Relative Importance", color='#A0AAB5')
            
            st.pyplot(fig)
        except Exception:
            st.info("Feature importance is not currently exposed by the pipeline.")
