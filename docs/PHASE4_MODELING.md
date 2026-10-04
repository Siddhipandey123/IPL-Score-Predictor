# Phase 4: Model Training and Evaluation

This document summarizes the Phase 4 machine learning pipeline for real-time IPL score prediction.

## 1. Objective
Build a final-score prediction system from the current innings state, strictly adhering to the chronological data splits and using only the validation set for model selection, while reserving the test set exclusively for final unbiased evaluation. No target leakage or future-data leakage is permitted.

## 2. Models Implemented
Given the project constraints and the tabular nature of the data, we implemented a practical tabular ML model alongside a mathematically naive baseline to prove learning efficacy.

### Baseline Model: Current Run Rate Projection
- **Logic**: Projected Score = Current Score + (Current Run Rate * Overs Remaining)
- **Purpose**: A naive benchmark to ensure the ML model captures complex match state.

### Primary Model: XGBoost Regressor
- **Framework**: `xgboost` with `scikit-learn` Pipeline.
- **Handling Categoricals**: Used `OneHotEncoder` for `batting_team_clean`, `bowling_team_clean`, and `venue_clean`.
- **Hyperparameters**:
  - `n_estimators`: 500 (with early stopping on validation set)
  - `learning_rate`: 0.05
  - `max_depth`: 6
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8

## 3. Dataset and Split
- **Dataset**: `data/processed/innings_features.csv`
- **Split mechanism**: Strict chronological split defined in Phase 2/3.
- **Train Size**: 222,213 deliveries (Used purely for training).
- **Validation Size**: 17,307 deliveries (Used strictly for model tuning and selection).
- **Test Size**: 33,818 deliveries (Completely untouched until the final model was chosen).

## 4. Features
*A full feature audit is available in `PHASE4_FEATURE_AUDIT.md`.*
- **Live State**: Score, wickets, run rates, balls remaining, recent momentum.
- **Categorical**: Batting team, bowling team, venue.
- **Historical**: Leakage-safe rolling averages (from Phase 3) for venues and teams prior to the match.

## 5. Model Selection (Using Validation Set)
**Crucially, the test set was excluded during this phase.** We evaluated the trained models on the Validation Set to make our final model selection decision.

| Model | Validation MAE | Validation RMSE | Validation MAPE |
|-------|----------------|-----------------|-----------------|
| **Baseline** | 28.62 | 46.09 | 16.71% |
| **XGBoost** | 18.55 | 24.31 | 10.74% |

**Decision**: Because XGBoost achieved a significantly lower MAE (18.55) compared to the Baseline (28.62) on the validation set, **XGBoost was selected as the final model**.

## 6. Final Unbiased Evaluation (Using Test Set)
After selecting XGBoost, we evaluated it—and the baseline, for comparison—once on the completely untouched Test Set.

### Overall Performance

| Model | Test MAE | Test RMSE | Test MAPE |
|-------|----------|-----------|-----------|
| **Baseline** | 29.81 | 48.01 | 17.02% |
| **XGBoost** | 22.68 | 30.01 | 12.07% |

### Performance by Innings Stage

| Stage | Baseline Test MAE | XGBoost Test MAE |
|-------|-------------------|------------------|
| **Powerplay (Overs 0-5)** | 59.79 | 31.98 |
| **Middle (Overs 6-15)** | 23.55 | 22.37 |
| **Death (Overs 16-19)** | 9.25 | 12.57 |

*Insights:* XGBoost learns the nuanced dynamics of the innings, realizing that high Powerplay run rates usually drop. The baseline simply extrapolates the current run rate, causing a massive ~60 run error during the first 5 overs. The baseline is mathematically highly accurate in the final overs when few balls remain.

## 7. Real-Time Pipeline
The selected final model (**XGBoost**) is saved at `models/xgboost_pipeline.pkl`.
A reusable prediction script is available in `scripts/predict.py`. It requires only leakage-free live state and historical features to make instantaneous predictions.

