# FUTURE XI — Real-Time IPL Score Forecasting & Stadium Analytics

## 1. Project Overview
FUTURE XI is an end-to-end machine learning system and interactive web application that forecasts the final score of a first-innings IPL batting team based on the current match state. Using historical IPL ball-by-ball data, the model processes real-time match dynamics to provide an accurate estimate of the projected total.

## 2. Problem Statement
During a T20 cricket match, predicting the final score is notoriously difficult due to the volatile nature of the format. Traditional run-rate extrapolations often fail because they do not account for critical contextual factors such as pitch history, wickets lost, batting team strength, and bowling team quality. This project aims to solve that by building a data-driven machine learning model that understands historical patterns and current momentum.

## 3. Objectives
- Build a machine learning model capable of accurately forecasting a first-innings IPL score.
- Ensure strict temporal separation during training to prevent data leakage from future matches.
- Develop a visually appealing, interactive dashboard for real-time inference and analytics.
- Honestly evaluate the model against a standard current-run-rate baseline.

## 4. Key Features
- **Real-Time Match Forecasting:** Predicts the final score based on overs completed, wickets lost, current score, and recent momentum.
- **Cricket Plausibility Constraint:** Ensures predictions remain mathematically and logically possible (never exceeding the theoretical maximum of 6 runs per remaining ball).
- **Stadium Analytics:** Explores historical scoring trends, average run rates, and past match statistics by venue.
- **Model Insights:** Transparently displays model evaluation metrics and global feature importance.

## 5. Machine Learning Approach
- **Algorithm:** XGBoost Regressor
- **Validation Strategy:** Strict chronological train/validation/test split to prevent data leakage.
  - **Training:** Matches through 2022
  - **Validation:** Matches in 2023
  - **Testing:** Matches in 2024–2025
- **Baseline Comparison:** The XGBoost model is rigorously evaluated against a standard current-run-rate extrapolation baseline.

## 6. Data & Preprocessing
The model was trained on a large IPL ball-by-ball historical dataset covering 12+ years of matches.
- **Phase 1 & 2:** Initial data understanding, canonicalization of team/venue names, and outlier removal.
- **Phase 3:** Leakage-safe historical feature engineering.
- **Phase 4:** Assembly of the model-ready dataset and pipeline construction.

## 7. Feature Engineering
The model utilizes a blend of match-state and historical features:
- **Match State:** Current score, wickets lost, balls completed, balls remaining, current run rate.
- **Momentum:** Runs scored in the last 6, 12, and 18 balls.
- **Historical (Leakage-Safe):** Team batting averages, team bowling averages, and venue averages up to (but not including) the current match date. 

## 8. Model Evaluation
The pipeline was evaluated purely on the chronologically held-out test set (2024–2025 matches) to ensure an unbiased estimation of real-world performance. 

## 9. Results
Overall Test Performance:
- **XGBoost Test MAE:** ~22.7 runs
- **XGBoost Test RMSE:** ~30.0 runs
- **XGBoost Test MAPE:** ~12.1%
- **Baseline Test MAE:** ~29.8 runs

Powerplay vs Baseline:
- **XGBoost Powerplay MAE:** ~32.0 runs
- **Baseline Powerplay MAE:** ~59.8 runs

*Note: The XGBoost model demonstrates massive superiority over the baseline during the volatile Powerplay and middle overs.*

## 10. Dashboard
The interactive Streamlit dashboard is split into three main sections:
1. **Match Prediction:** First-innings forecasting only. The user enters team and match state data (automatically handling valid cricket over notation conversions). The UI is protected by strict domain logic validation.
2. **Stadium Insights:** Historical venue performance, average scores, run rates, high scores, and trend analysis.
3. **Model Insights:** Displays test evaluation metrics (MAE, RMSE, MAPE) and visualizes the most influential features driving the predictions.

## 11. Project Pipeline

```text
Raw IPL Ball-by-Ball Data
           ↓
Phase 1 — Initial Data Understanding
           ↓
Phase 2 — Data Cleaning & Canonicalization
           ↓
Phase 3 — Historical Feature Engineering
           ↓
Phase 4 — Model-Ready Dataset
           ↓
Chronological Train / Validation / Test Split
           ↓
XGBoost Regressor
           ↓
Prediction + Cricket Plausibility Constraint
           ↓
Streamlit Dashboard
```

## 12. Repository Structure
```
IPL-Score-Predictor/
├── .streamlit/
│   └── config.toml                # Dashboard theme configuration
├── data/
│   └── processed/
│       ├── innings_clean.csv      # Cleaned Phase 2 data
│       └── innings_features.csv   # Model-ready Phase 3 data
├── docs/                          # Detailed phase documentation
├── models/
│   └── xgboost_pipeline.pkl       # Trained XGBoost model pipeline
├── pages/
│   ├── 1_Match_Prediction.py      # Match prediction UI
│   ├── 2_Stadium_Insights.py      # Venue analytics UI
│   └── 3_Model_Insights.py        # Model evaluation UI
├── reports/                       # Evaluation JSON metric reports
├── scripts/
│   ├── prepare_phase2.py          # Data cleaning script
│   ├── prepare_phase3.py          # Feature engineering script
│   ├── train_phase4.py            # Model training script
│   ├── evaluate_phase4.py         # Evaluation script
│   └── predict.py                 # Prediction inference module
├── .gitignore                     # Git tracking rules
├── app.py                         # Main Streamlit landing page
├── IPL.csv                        # Raw historical dataset
├── requirements.txt               # Project dependencies
└── utils.py                       # Shared backend and data utilities
```

## 13. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/IPL-Score-Predictor.git
   cd IPL-Score-Predictor
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 14. Running the Application
To start the FUTURE XI dashboard locally, simply run:
```bash
streamlit run app.py
```
This will launch the application in your default web browser.

## 15. Limitations
**Death Overs Forecasting:** During the final death overs of an innings, the simple current-run-rate baseline occasionally outperforms the XGBoost model. This is an observed machine learning limitation likely due to the extreme variance and chaotic nature of T20 death batting, which is difficult for standard regressors to capture smoothly without under-predicting explosive finishes.

## 16. Future Improvements
- **Model Blending:** Blending the XGBoost model with the baseline run-rate strictly during the death overs.
- **Prediction Uncertainty:** Outputting confidence intervals alongside the point estimate.
- **Momentum Granularity:** Improving momentum features (e.g., separating boundary count vs standard runs).
- **Deployment:** Containerizing the application using Docker and scaling it via cloud deployment.
- **Continuous Learning:** Adding more recent IPL data sequentially to keep the model updated with modern scoring trends (e.g., the "Impact Player" rule effect).

## 17. Skills / Technologies
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Joblib
- **Frontend / UI:** Streamlit
- **Machine Learning:** Regression, Feature Engineering, Temporal Cross-Validation, Pipeline Construction
- **Domain:** Sports Analytics, Cricket (T20)

## 18. Author
**Siddhi Kumari**
*(Update with your contact links, LinkedIn, or portfolio)*
