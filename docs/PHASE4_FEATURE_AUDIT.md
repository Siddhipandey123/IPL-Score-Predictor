# Phase 4: Feature Audit and Leakage Prevention

This document audits all features used in the Phase 4 Model Training to ensure zero data leakage from future events or post-match information.

## ✅ Included Features (Leakage-Safe)

### 1. Live Match-State Features
These features are updated ball-by-ball and represent exactly what is known at the moment of prediction.
- `current_score`: The batting team's score before the current delivery is bowled.
- `wickets_lost`: Wickets lost before the current delivery.
- `balls_completed` & `overs_completed`: Progress of the innings.
- `balls_remaining`: Number of legal deliveries left in the 20-over innings.
- `current_run_rate`: Run rate at the current point in the innings.
- `runs_last_6_balls`, `runs_last_12_balls`, `runs_last_18_balls`: Rolling sums of runs scored in the recent window.
- `innings`: The current innings (1 or 2).

### 2. Categorical / Identity Features
These are known before the match begins.
- `batting_team_clean`: The team batting.
- `bowling_team_clean`: The team bowling.
- `venue_clean`: The stadium/venue.

### 3. Historical Features (from Phase 3)
Calculated using strictly prior matches relative to the current match date.
- `venue_avg_score`: Average score at the venue in all prior matches.
- `venue_avg_run_rate`: Average run rate at the venue.
- `batting_team_avg_score_before`: Average score of the batting team historically.
- `batting_team_avg_run_rate_before`: Average run rate of the batting team historically.
- `bowling_team_avg_runs_conceded_before`: Average score conceded by bowling team historically.
- `bowling_team_avg_run_rate_conceded_before`: Average run rate conceded.
- `batting_team_venue_avg_score_before`: Team's historical average score at this specific venue.

---

## 🚫 Excluded Features (To Prevent Leakage)

The following columns present in `innings_features.csv` or raw data are explicitly **excluded** to prevent data leakage:

- `final_score`: The target variable. Cannot be used as a feature.
- `runs_total`, `runs_batter`, `runs_extras`, `runs_bowler`: Represents runs scored *on the current delivery* (happens in the future relative to the prediction point).
- `player_out`, `wicket_kind`: Represents a wicket falling *on the current delivery*.
- `team_runs`, `team_balls`, `team_wicket`: Final innings aggregates.
- `global_avg_score_before`, etc.: Some are redundant; we select the most predictive subsets to prevent multicollinearity without leakage.
- Post-match data (if any existed, e.g., result, player of the match): Excluded.

