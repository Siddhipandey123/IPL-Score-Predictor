import streamlit as st
import datetime
import matplotlib.pyplot as plt

from utils import (
    load_predictor, 
    load_historical_data, 
    convert_overs_to_balls, 
    get_latest_historical_features
)

st.set_page_config(page_title="Match Prediction - IPL Predictor", layout="wide", page_icon="🏏")

# Common CSS
st.markdown("""
<style>
.section-title {
    color: #A0AAB5;
    text-transform: uppercase;
    font-size: 0.9rem;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

try:
    predictor = load_predictor()
    hist_df = load_historical_data()
except Exception as e:
    st.error(f"Failed to load pipeline or data: {e}")
    st.stop()

teams = sorted(hist_df['batting_team_clean'].dropna().unique())
venues = sorted(hist_df['venue_clean'].dropna().unique())

st.title("🏏 Match Prediction")
st.markdown("Enter the current match state to predict the final innings score.")

st.page_link("app.py", label="← Back to Home", icon="🏠")
st.write("---")

col_main, col_side = st.columns([2, 1])

with col_main:
    with st.container(border=True):
        st.markdown('<div class="section-title">MATCH DETAILS</div>', unsafe_allow_html=True)
        
        col_d, col_i = st.columns(2)
        with col_d:
            match_date = st.date_input("Match Date", value=datetime.date.today())
        with col_i:
            st.selectbox("Innings", ["1st Innings (Supported)"], disabled=True)
            innings_val = 1

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            batting_team = st.selectbox("Batting Team", teams, index=teams.index('Chennai Super Kings') if 'Chennai Super Kings' in teams else 0)
        with col_t2:
            bowling_team = st.selectbox("Bowling Team", teams, index=teams.index('Mumbai Indians') if 'Mumbai Indians' in teams else 1)
        
        venue = st.selectbox("Stadium/Venue", venues, index=venues.index('Wankhede Stadium') if 'Wankhede Stadium' in venues else 0)
        
    with st.container(border=True):
        st.markdown('<div class="section-title">MATCH STATE</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            current_score = st.number_input("Current Score", min_value=0, max_value=400, value=100, step=1)
        with col_s2:
            wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1)
        with col_s3:
            valid_overs = []
            for i in range(20):
                for j in range(6):
                    valid_overs.append(f"{i}.{j}")
            valid_overs.append("20.0")
            overs_completed_str = st.selectbox("Overs Completed", valid_overs, index=valid_overs.index("12.0"))
        
    with st.container(border=True):
        st.markdown('<div class="section-title">MOMENTUM</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            runs_last_6 = st.number_input("Runs in Last 6 Balls", min_value=0, max_value=36, value=8)
        with col_m2:
            runs_last_12 = st.number_input("Runs in Last 12 Balls", min_value=0, max_value=72, value=15)
        with col_m3:
            runs_last_18 = st.number_input("Runs in Last 18 Balls", min_value=0, max_value=108, value=25)

with col_side:
    with st.container(border=True):
        st.markdown('<div class="section-title">PREDICTION</div>', unsafe_allow_html=True)
        
        overs_completed = float(overs_completed_str)
        
        balls_completed, err = convert_overs_to_balls(overs_completed)
        if err:
            st.error(err)
            st.stop()
            
        balls_remaining = 120 - balls_completed
        current_run_rate = (current_score / balls_completed) * 6 if balls_completed > 0 else 0
        
        st.markdown(f"**Current Run Rate:** {current_run_rate:.2f}")
        st.markdown(f"**Balls Remaining:** {balls_remaining}")
        st.write("")
        
        predict_btn = st.button("PREDICT FINAL SCORE", type="primary", use_container_width=True)
        
        if predict_btn:
            # Validations
            if batting_team == bowling_team:
                st.error("Batting and Bowling teams must be different.")
            elif runs_last_6 > runs_last_12 or runs_last_12 > runs_last_18:
                st.error("Invalid momentum: Runs in last 6 balls must be <= last 12 balls <= last 18 balls.")
            elif runs_last_18 > current_score:
                st.error("Invalid momentum: Runs in last 18 balls cannot exceed the current score.")
            elif balls_completed < 18 and runs_last_18 > 0 and (runs_last_18 > (balls_completed * 6)): # basic sanity
                 st.error("Invalid momentum: Runs exceed maximum possible for balls bowled.")
            elif balls_remaining == 0:
                st.info(f"Innings is complete! Final Score: {current_score}")
            else:
                hist_features = get_latest_historical_features(hist_df, batting_team, bowling_team, venue, match_date)
                
                current_state = {
                    'current_score': current_score,
                    'wickets_lost': wickets_lost,
                    'balls_completed': balls_completed,
                    'balls_remaining': balls_remaining,
                    'current_run_rate': current_run_rate,
                    'runs_last_6_balls': runs_last_6,
                    'runs_last_12_balls': runs_last_12,
                    'runs_last_18_balls': runs_last_18,
                    'innings': innings_val, 
                    'batting_team_clean': batting_team,
                    'bowling_team_clean': bowling_team,
                    'venue_clean': venue,
                }
                current_state.update(hist_features)
                
                try:
                    raw_model_prediction = predictor.predict(current_state)
                    
                    # Cricket-domain plausibility constraint for late-innings predictions:
                    # Prevents the model from forecasting more runs than theoretically possible (6 runs * balls remaining)
                    maximum_projection = current_score + (6 * balls_remaining)
                    
                    final_pred = min(raw_model_prediction, maximum_projection)
                    final_pred = max(final_pred, current_score)
                    
                    # Update Session State for Home Page
                    st.session_state['last_batting_team'] = batting_team
                    st.session_state['last_bowling_team'] = bowling_team
                    st.session_state['last_venue'] = venue
                    st.session_state['last_score'] = current_score
                    st.session_state['last_wickets'] = wickets_lost
                    st.session_state['last_overs'] = overs_completed
                    st.session_state['last_prediction'] = final_pred
                    
                    st.write("")
                    st.markdown(f"<h1 style='text-align: center; color: #00E676; font-size: 4.5rem; margin-bottom: 0;'>{final_pred:.0f}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: #A0AAB5;'>PROJECTED RUNS</p>", unsafe_allow_html=True)
                    
                    st.write("---")
                    # Score Forecast Chart
                    fig, ax = plt.subplots(figsize=(4, 3))
                    fig.patch.set_facecolor('#14243A')
                    ax.set_facecolor('#14243A')
                    x_vals = [f"{overs_completed:.1f} ov", "20.0 ov"]
                    y_vals = [current_score, final_pred]
                    
                    ax.plot(x_vals, y_vals, marker='o', color='#00E676', linewidth=2, markersize=8)
                    ax.fill_between(x_vals, 0, y_vals, color='#00E676', alpha=0.1)
                    ax.set_ylim(0, max(final_pred + 20, 250))
                    ax.tick_params(colors='#A0AAB5')
                    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#A0AAB5')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('#1C314D')
                    ax.spines['left'].set_color('#1C314D')
                    
                    for i, v in enumerate(y_vals):
                        ax.text(i, v + 8, f"{v:.0f}", ha='center', color='white', fontweight='bold')
                        
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
