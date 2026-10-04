import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuration
DATA_PATH = "data/processed/innings_features.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Define Features
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

TARGET = 'final_score'
SPLIT_COL = 'split'

ALL_FEATURES = LIVE_FEATURES + CATEGORICAL_FEATURES + HISTORICAL_FEATURES

class BaselineModel:
    """
    Current Run Rate Projection Baseline:
    Projected Final Score = Current Score + (Current Run Rate * Overs Remaining)
    """
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        overs_remaining = X['balls_remaining'] / 6.0
        projected_additional_runs = X['current_run_rate'] * overs_remaining
        return X['current_score'] + projected_additional_runs

def evaluate(model, X, y, name):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = mean_absolute_percentage_error(y, preds)
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def main():
    print("Loading data...")
    # low_memory=False to avoid DtypeWarning
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Fill any NaNs in features
    df[HISTORICAL_FEATURES] = df[HISTORICAL_FEATURES].fillna(0)
    df[LIVE_FEATURES] = df[LIVE_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    # Split
    train_df = df[df[SPLIT_COL] == 'train'].copy()
    val_df = df[df[SPLIT_COL] == 'validation'].copy()
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print("Note: Test set is excluded from this script to strictly prevent data leakage and ensure unbiased final evaluation.")

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[ALL_FEATURES]
    y_val = val_df[TARGET]

    # Evaluate Baseline on Validation for Model Selection
    baseline = BaselineModel()
    val_metrics_base = evaluate(baseline, X_val, y_val, "Baseline (Validation Set)")
    
    # XGBoost Preparation
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=100
    )

    print("\nEvaluating XGBoost on Validation...")
    val_preds_xgb = xgb_model.predict(X_val_processed)
    val_metrics_xgb = {
        "MAE": mean_absolute_error(y_val, val_preds_xgb),
        "RMSE": np.sqrt(mean_squared_error(y_val, val_preds_xgb)),
        "MAPE": mean_absolute_percentage_error(y_val, val_preds_xgb)
    }
    print(f"--- XGBoost (Validation Set) ---")
    print(f"MAE:  {val_metrics_xgb['MAE']:.2f}")
    print(f"RMSE: {val_metrics_xgb['RMSE']:.2f}")
    print(f"MAPE: {val_metrics_xgb['MAPE']:.2%}")

    print("\nMODEL SELECTION DECISION:")
    print("We compare the Validation metrics to select the best model before touching the Test set.")
    if val_metrics_xgb['MAE'] < val_metrics_base['MAE']:
        print("XGBoost selected as the final model due to lower Validation MAE.")
    else:
        print("Baseline selected as the final model due to lower Validation MAE.")

    # Save models
    print("\nSaving selected model pipeline...")
    inference_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    joblib.dump(inference_pipeline, os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl'))
    
    # Also save the baseline for easy comparison later if needed
    joblib.dump(baseline, os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    
    with open(os.path.join(REPORTS_DIR, 'validation_metrics.json'), 'w') as f:
        json.dump({"baseline": val_metrics_base, "xgboost": val_metrics_xgb}, f, indent=4)
        
    print("Phase 4 Training and Validation Complete!")

if __name__ == "__main__":
    main()

