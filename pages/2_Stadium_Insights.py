import streamlit as st
import datetime
import matplotlib.pyplot as plt

from utils import load_historical_data, calculate_stadium_insights

st.set_page_config(page_title="Stadium Insights - IPL Predictor", layout="wide", page_icon="🏟️")

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
</style>
""", unsafe_allow_html=True)

try:
    hist_df = load_historical_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.title("🏟️ Stadium Insights")
st.markdown("Explore historical scoring patterns and trends by stadium.")
st.page_link("app.py", label="← Back to Home", icon="🏠")
st.write("---")

venues = sorted(hist_df['venue_clean'].dropna().unique())

col_main, col_date = st.columns([3, 1])
with col_main:
    venue = st.selectbox("Select Stadium", venues, index=venues.index('Wankhede Stadium') if 'Wankhede Stadium' in venues else 0)
with col_date:
    match_date = st.date_input("View Data Prior To", value=datetime.date.today())

st.write("")

metrics, trend_df = calculate_stadium_insights(hist_df, venue, match_date)

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        with st.container(border=True):
            st.markdown('<div class="metric-title">Matches Analyzed</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{metrics["matches_analyzed"]}</div>', unsafe_allow_html=True)
        
    with c2:
        with st.container(border=True):
            st.markdown('<div class="metric-title">Average Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{metrics["avg_score"]:.1f}</div>', unsafe_allow_html=True)
        
    with c3:
        with st.container(border=True):
            st.markdown('<div class="metric-title">Average Run Rate</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{metrics["avg_run_rate"]:.2f}</div>', unsafe_allow_html=True)
        
    with c4:
        with st.container(border=True):
            st.markdown('<div class="metric-title">Highest Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{metrics["high_score"]:.0f}</div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Historical Score Trend")
    st.markdown("First Innings Scores over the last 20 matches.")
    
    if len(trend_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#07111F')
        ax.set_facecolor('#101D2F')
        
        ax.plot(range(1, len(trend_df)+1), trend_df['final_score'], marker='o', color='#1E88E5', linewidth=2)
        ax.axhline(y=metrics['avg_score'], color='#00E676', linestyle='--', label='Average Score', linewidth=2)
        
        ax.set_xlabel("Recent Matches (Oldest → Newest)", color='#A0AAB5')
        ax.set_ylabel("First Innings Score", color='#A0AAB5')
        ax.tick_params(colors='#A0AAB5')
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#A0AAB5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1C314D')
        ax.spines['left'].set_color('#1C314D')
        ax.legend(facecolor='#101D2F', edgecolor='#1C314D', labelcolor='#FFFFFF')
        
        st.pyplot(fig)
else:
    st.info("No historical data available for this venue prior to the selected date. (Cold Start)")
