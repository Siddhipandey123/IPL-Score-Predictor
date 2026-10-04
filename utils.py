import pandas as pd
import os
import sys
import datetime
import streamlit as st

# Ensure scripts module can be imported correctly regardless of where the file is run
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from scripts.predict import IPLScorePredictor, HISTORICAL_FEATURES

# Phase 3 Fallback Constants
PRIOR_INNINGS_SCORE = 160.0
PRIOR_RUN_RATE = 8.0

@st.cache_resource
def load_predictor():
    return IPLScorePredictor()

@st.cache_data
def load_historical_data():
    df = pd.read_csv("data/processed/innings_features.csv", low_memory=False)
    extra_cols = ['match_id', 'innings', 'final_score', 'venue_highest_score', 'venue_lowest_score']
    cols = ['date', 'batting_team_clean', 'bowling_team_clean', 'venue_clean'] + HISTORICAL_FEATURES + extra_cols
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].sort_values('date')
    return df

def convert_overs_to_balls(overs_completed):
    full_overs = int(overs_completed)
    balls_in_partial = int(round((overs_completed - full_overs) * 10))
    if balls_in_partial > 5:
        return None, "Invalid overs input. Fractional part cannot exceed .5 (e.g., 12.3 means 12 overs and 3 balls, 12.6 is invalid)."
    balls_completed = (full_overs * 6) + balls_in_partial
    if balls_completed > 120:
        return None, "Balls completed cannot exceed 120 (20 overs)."
    return balls_completed, None

def get_latest_historical_features(df, bat_team, bowl_team, venue, match_date):
    hist = {}
    past_df = df[df['date'] < str(match_date)]
    
    # Venue features
    v_df = past_df[past_df['venue_clean'] == venue]
    if len(v_df) > 0:
        hist['venue_avg_score'] = v_df.iloc[-1]['venue_avg_score']
        hist['venue_avg_run_rate'] = v_df.iloc[-1]['venue_avg_run_rate']
    else:
        hist['venue_avg_score'] = PRIOR_INNINGS_SCORE
        hist['venue_avg_run_rate'] = PRIOR_RUN_RATE
        
    # Batting team features
    bat_df = past_df[past_df['batting_team_clean'] == bat_team]
    if len(bat_df) > 0:
        hist['batting_team_avg_score_before'] = bat_df.iloc[-1]['batting_team_avg_score_before']
        hist['batting_team_avg_run_rate_before'] = bat_df.iloc[-1]['batting_team_avg_run_rate_before']
    else:
        hist['batting_team_avg_score_before'] = PRIOR_INNINGS_SCORE
        hist['batting_team_avg_run_rate_before'] = PRIOR_RUN_RATE
        
    # Bowling team features
    bowl_df = past_df[past_df['bowling_team_clean'] == bowl_team]
    if len(bowl_df) > 0:
        hist['bowling_team_avg_runs_conceded_before'] = bowl_df.iloc[-1]['bowling_team_avg_runs_conceded_before']
        hist['bowling_team_avg_run_rate_conceded_before'] = bowl_df.iloc[-1]['bowling_team_avg_run_rate_conceded_before']
    else:
        hist['bowling_team_avg_runs_conceded_before'] = PRIOR_INNINGS_SCORE
        hist['bowling_team_avg_run_rate_conceded_before'] = PRIOR_RUN_RATE
        
    # Batting team at Venue features
    bat_v_df = past_df[(past_df['batting_team_clean'] == bat_team) & (past_df['venue_clean'] == venue)]
    if len(bat_v_df) > 0:
        hist['batting_team_venue_avg_score_before'] = bat_v_df.iloc[-1]['batting_team_venue_avg_score_before']
    else:
        hist['batting_team_venue_avg_score_before'] = hist['venue_avg_score']
        
    return hist

def calculate_stadium_insights(hist_df, venue, match_date):
    past_df = hist_df[(hist_df['venue_clean'] == venue) & (hist_df['date'] < str(match_date))]
    if len(past_df) == 0:
        return None, None
    
    inn1_df = past_df[past_df['innings'] == 1].drop_duplicates(subset=['match_id'])
    matches_analyzed = len(inn1_df)
    
    if matches_analyzed == 0:
        return None, None
        
    avg_score = inn1_df['final_score'].mean()
    high_score = inn1_df['final_score'].max()
    low_score = inn1_df['final_score'].min()
    
    avg_run_rate = (inn1_df['final_score'] / 20.0).mean()
    
    metrics = {
        'matches_analyzed': matches_analyzed,
        'avg_score': avg_score,
        'high_score': high_score,
        'low_score': low_score,
        'avg_run_rate': avg_run_rate
    }
    
    trend_df = inn1_df[['date', 'final_score']].sort_values('date').tail(20)
    
    return metrics, trend_df
