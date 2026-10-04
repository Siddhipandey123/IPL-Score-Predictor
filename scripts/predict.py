import os
import joblib
import pandas as pd
import sys

# Define expected features based on training script
LIVE_FEATURES = [
    'current_score', 'wickets_lost', 'balls_completed', 'balls_remaining',
    'current_run_rate', 'runs_last_6_balls', 'runs_last_12_balls', 'runs_last_18_balls',
    'innings'
]

CATEGORICAL_FEATURES = ['batting_team_clean', 'bowling_team_clean', 'venue_clean']

HISTORICAL_FEATURES = [
    'venue_avg_score', 'venue_avg_run_rate', 
    'batting_team_avg_score_before', 'batting_team_avg_run_rate_before',
    'bowling_team_avg_runs_conceded_before', 'bowling_team_avg_run_rate_conceded_before',
    'batting_team_venue_avg_score_before'
]

ALL_FEATURES = LIVE_FEATURES + CATEGORICAL_FEATURES + HISTORICAL_FEATURES

class IPLScorePredictor:
    """
    A real-time prediction interface that wraps the trained Phase 4 ML pipeline.
    This class is designed to be imported and used by the Phase 5 dashboard.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            # Default to the models directory relative to this script
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'xgboost_pipeline.pkl')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
            
        self.pipeline = joblib.load(model_path)

    def predict(self, current_state: dict) -> float:
        """
        Predicts the final score given the current state of the innings.
        
        Args:
            current_state (dict): Dictionary containing the live, categorical, and historical features.
            
        Returns:
            float: The predicted final score.
        """
        # Validate input features
        missing_features = [f for f in ALL_FEATURES if f not in current_state]
        if missing_features:
            raise ValueError(f"Missing required features in input: {missing_features}")
            
        # Convert dictionary to single-row DataFrame
        df = pd.DataFrame([current_state])
        
        # Enforce exact same types and preprocessing as training
        df[HISTORICAL_FEATURES] = df[HISTORICAL_FEATURES].fillna(0)
        df[LIVE_FEATURES] = df[LIVE_FEATURES].fillna(0)
        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
        
        # Ensure column order matches ALL_FEATURES exactly
        df = df[ALL_FEATURES]
        
        # Predict using the loaded sklearn Pipeline (handles one-hot encoding internally)
        prediction = self.pipeline.predict(df)[0]
        
        return float(prediction)

if __name__ == "__main__":
    print("--- IPL Score Predictor CLI Demo ---\n")
    try:
        predictor = IPLScorePredictor()
        
        # A valid example prediction state
        sample_state = {
            # Live Match-State
            'current_score': 100,
            'wickets_lost': 2,
            'balls_completed': 72,
            'balls_remaining': 48,
            'current_run_rate': 8.33,
            'runs_last_6_balls': 8,
            'runs_last_12_balls': 15,
            'runs_last_18_balls': 25,
            'innings': 1,
            
            # Categorical Match Identity
            'batting_team_clean': 'Chennai Super Kings',
            'bowling_team_clean': 'Mumbai Indians',
            'venue_clean': 'Wankhede Stadium',
            
            # Phase 3 Historical Features (Pre-computed for venue/teams)
            'venue_avg_score': 160.0,
            'venue_avg_run_rate': 8.0,
            'batting_team_avg_score_before': 165.0,
            'batting_team_avg_run_rate_before': 8.2,
            'bowling_team_avg_runs_conceded_before': 155.0,
            'bowling_team_avg_run_rate_conceded_before': 7.7,
            'batting_team_venue_avg_score_before': 162.0
        }
        
        print(f"Input State:")
        for k, v in sample_state.items():
            print(f"  {k}: {v}")
            
        print("\nPredicting...")
        predicted_score = predictor.predict(sample_state)
        
        print(f"\n[SUCCESS] Predicted Final Score: {predicted_score:.1f} runs")
        
    except Exception as e:
        print(f"[ERROR] running prediction: {e}")
        sys.exit(1)
