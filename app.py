import streamlit as st

st.set_page_config(
    page_title="FUTURE XI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for extra layout improvements
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:nth-child(1) a span:last-child { display: none; }
[data-testid="stSidebarNav"] ul li:nth-child(1) a::after { content: "🏏 FUTURE XI"; margin-left: 0.5rem; font-weight: 600; }
.stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.hero-header {
    text-align: center;
    padding-top: 1rem;
    padding-bottom: 1rem;
    margin: 0 auto;
}
.hero-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0;
    color: #FFFFFF;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #A0AAB5;
    margin-top: -10px;
    margin-bottom: 1rem;
}
.match-card {
    background-color: #101D2F;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #1C314D;
}
.match-teams {
    font-size: 2rem;
    font-weight: bold;
    color: #FFFFFF;
}
.match-venue {
    font-size: 1.1rem;
    color: #A0AAB5;
    margin-top: 0.5rem;
}
.score-value {
    font-size: 4rem;
    font-weight: 800;
    color: #FFFFFF;
}
.predicted-value {
    font-size: 4rem;
    font-weight: 800;
    color: #00E676; /* Accent color */
}
.score-label {
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #A0AAB5;
}
.sub-stat {
    font-size: 1.2rem;
    color: #A0AAB5;
}
</style>
""", unsafe_allow_html=True)

# Container for Hero Section
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🏏</div>
    <div class="hero-title">FUTURE XI</div>
    <div class="hero-subtitle">Real-Time Cricket Analytics & Score Forecasting</div>
</div>
""", unsafe_allow_html=True)

# Fetch from session state if available, else use a demo state
demo_bat = st.session_state.get('last_batting_team', "Chennai Super Kings")
demo_bowl = st.session_state.get('last_bowling_team', "Mumbai Indians")
demo_venue = st.session_state.get('last_venue', "Arun Jaitley Stadium, Delhi")
demo_score = st.session_state.get('last_score', 124)
demo_wickets = st.session_state.get('last_wickets', 4)
demo_overs = st.session_state.get('last_overs', 15.2)
demo_pred = st.session_state.get('last_prediction', 181)

st.markdown(f"""
<div class="match-card">
    <div class="match-teams">{demo_bat} <span style="color: #4A6583;">VS</span> {demo_bowl}</div>
    <div class="match-venue">{demo_venue}</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 4, 4, 1])

with col2:
    st.markdown(f"""
    <div style="text-align: center; background-color: #101D2F; padding: 2rem; border-radius: 12px; border: 1px solid #1C314D; height: 100%;">
        <div class="score-label">CURRENT SCORE</div>
        <div class="score-value">{demo_score} / {demo_wickets}</div>
        <div class="sub-stat">{demo_overs:.1f} overs</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="text-align: center; background-color: #14243A; padding: 2rem; border-radius: 12px; border: 1px solid #1E88E5; height: 100%;">
        <div class="score-label" style="color: #00E676;">PREDICTED FINAL</div>
        <div class="predicted-value">{demo_pred:.0f}</div>
        <div class="sub-stat">final score</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.write("---")
st.write("")

nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### 🏏 Match Prediction")
    st.markdown("Predict the final score from the current innings state.")
    st.page_link("pages/1_Match_Prediction.py", label="Open Match Prediction", icon="🏏", use_container_width=True)

with nav2:
    st.markdown("### 🏟️ Stadium Insights")
    st.markdown("Explore historical scoring patterns by stadium.")
    st.page_link("pages/2_Stadium_Insights.py", label="Open Stadium Insights", icon="🏟️", use_container_width=True)

with nav3:
    st.markdown("### 🤖 Model Insights")
    st.markdown("View model performance and feature importance.")
    st.page_link("pages/3_Model_Insights.py", label="Open Model Insights", icon="🤖", use_container_width=True)
