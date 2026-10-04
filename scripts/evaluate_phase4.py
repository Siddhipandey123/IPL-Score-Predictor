import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from train_phase4 import DATA_PATH, ALL_FEATURES, TARGET, SPLIT_COL, MODELS_DIR, REPORTS_DIR

def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

def analyze_by_stage(df, y_true, y_pred, model_name):
    print(f"\n--- Analysis by Innings Stage ({model_name}) ---")
    stages = [
        ("Powerplay (Overs 0-5)", df['overs_completed'] <= 5),
        ("Middle (Overs 6-15)", (df['overs_completed'] > 5) & (df['overs_completed'] <= 15)),
        ("Death (Overs 16-19)", df['overs_completed'] > 15)
    ]
    
    results = {}
    for stage_name, condition in stages:
        if condition.sum() > 0:
            m = get_metrics(y_true[condition], y_pred[condition])
            results[stage_name] = m
            print(f"{stage_name} -> MAE: {m['MAE']:.2f}, RMSE: {m['RMSE']:.2f}, MAPE: {m['MAPE']:.2%}")
    return results

def main():
    print("Loading test data for final unbiased evaluation...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    from train_phase4 import HISTORICAL_FEATURES, LIVE_FEATURES, CATEGORICAL_FEATURES
    df[HISTORICAL_FEATURES] = df[HISTORICAL_FEATURES].fillna(0)
    df[LIVE_FEATURES] = df[LIVE_FEATURES].fillna(0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    # Strictly use ONLY the test set
    test_df = df[df[SPLIT_COL] == 'test'].copy()
    print(f"Test size: {len(test_df)}")
    
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]

    report = {}

    # Evaluate Baseline (for benchmark comparison on test set)
    from train_phase4 import BaselineModel
    baseline = BaselineModel()
    y_pred_base = baseline.predict(X_test)
    metrics_base = get_metrics(y_test, y_pred_base)
    print("\n--- Baseline Model (FINAL TEST SET) ---")
    print(f"MAE:  {metrics_base['MAE']:.2f}")
    print(f"RMSE: {metrics_base['RMSE']:.2f}")
    print(f"MAPE: {metrics_base['MAPE']:.2%}")
    base_stage_metrics = analyze_by_stage(test_df, y_test, y_pred_base, "Baseline")
    report['baseline_test'] = {"overall": metrics_base, "stages": base_stage_metrics}

    # Evaluate Selected XGBoost Model
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_pipeline.pkl')
    if os.path.exists(xgb_path):
        xgb_pipeline = joblib.load(xgb_path)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        metrics_xgb = get_metrics(y_test, y_pred_xgb)
        print("\n--- Selected XGBoost Model (FINAL TEST SET) ---")
        print(f"MAE:  {metrics_xgb['MAE']:.2f}")
        print(f"RMSE: {metrics_xgb['RMSE']:.2f}")
        print(f"MAPE: {metrics_xgb['MAPE']:.2%}")
        xgb_stage_metrics = analyze_by_stage(test_df, y_test, y_pred_xgb, "XGBoost")
        report['xgboost_test'] = {"overall": metrics_xgb, "stages": xgb_stage_metrics}
        
    if report:
        with open(os.path.join(REPORTS_DIR, 'test_metrics.json'), 'w') as f:
            json.dump(report, f, indent=4)
        print("\nSaved test evaluation metrics to reports/test_metrics.json")
    else:
        print("\nNo models found to evaluate. Please run train_phase4.py first.")

if __name__ == "__main__":
    main()
