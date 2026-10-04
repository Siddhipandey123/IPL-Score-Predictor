# Phase 2: data cleaning and leakage-safe dataset

This phase builds a **primary 20-over innings dataset** for real-time-style final-score prediction. Raw `IPL.csv` is **never modified**.

Re-run:

```text
python scripts/prepare_phase2.py
```

Output: `data/processed/innings_clean.csv`

---

## What was cleaned

- Dropped the source index column `Unnamed: 0`.
- Kept only **innings 1 and 2** (scheduled T20 innings).
- Removed **entire matches** that were **D/L** or **no result**, because those finals are not standard complete 20-over innings.
- Super-over innings (`innings` 3–6) were dropped from this dataset. The parent matches’ innings 1–2 were **kept** (they finished as ties, not D/L/no-result).
- Added canonical team and venue names **without deleting** the original `batting_team`, `bowling_team`, and `venue` columns.
- Added leakage-safe live-state features and a per-innings `final_score` target.
- Assigned a **match-level** chronological train / validation / test split.

## What was excluded (from the processed file only)

| Exclusion | Rule | Counts (from `python scripts/prepare_phase2.py`) |
|---|---|---|
| Super overs | `innings > 2` | 171 balls, 32 mini-innings, 15 matches (innings 1–2 of those matches were kept) |
| D/L | `method` non-null on the match | 22 matches, 3,890 balls |
| No result | `result_type == "no result"` | 8 matches, 806 balls |

Net processed size: **273,338 balls**, **1,139 matches** (from 278,205 / 1,169). D/L and no-result matches do not overlap.

These rows remain in `IPL.csv`.

Post-match / future-information columns were **not copied** into `innings_clean.csv` so they cannot be used accidentally as features:

- `player_of_match`, `match_won_by`, `win_outcome`, `result_type`, `superover_winner`, `method`, `runs_target`

`runs_target` is the completed first-innings total plus one. It is omitted from this primary file (including innings 2) to keep the first modelling pass free of chase-target leakage. A later optional chase model can re-join it from rules that only apply after innings 1 is finished.

## Team mappings (`batting_team_clean` / `bowling_team_clean`)

Same franchise, renamed or spelling variants:

| Raw name | Canonical name |
|---|---|
| Royal Challengers Bangalore | Royal Challengers Bengaluru |
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Rising Pune Supergiants | Rising Pune Supergiant |

Names already in canonical form are left unchanged.

**Not merged** (different franchises):

- Deccan Chargers ≠ Sunrisers Hyderabad
- Gujarat Lions ≠ Gujarat Titans
- Pune Warriors ≠ Rising Pune Supergiant
- Kochi Tuskers Kerala remains its own team

## Venue mappings (`venue_clean`)

Normalized **punctuation, missing city suffixes, and official same-ground renames** that are well documented:

| Raw examples | Canonical venue |
|---|---|
| Wankhede Stadium / Wankhede Stadium, Mumbai | Wankhede Stadium, Mumbai |
| M Chinnaswamy Stadium / M.Chinnaswamy Stadium / M Chinnaswamy Stadium, Bengaluru | M Chinnaswamy Stadium, Bengaluru |
| Eden Gardens / Eden Gardens, Kolkata | Eden Gardens, Kolkata |
| Feroz Shah Kotla / Arun Jaitley Stadium / Arun Jaitley Stadium, Delhi | Arun Jaitley Stadium, Delhi |
| MA Chidambaram Stadium (Chepauk variants) | MA Chidambaram Stadium, Chepauk, Chennai |
| Rajiv Gandhi International Stadium (Uppal variants) | Rajiv Gandhi International Stadium, Uppal, Hyderabad |
| Brabourne Stadium variants | Brabourne Stadium, Mumbai |
| Dr DY Patil Sports Academy variants | Dr DY Patil Sports Academy, Mumbai |
| ACA-VDCA Visakhapatnam variants | …Visakhapatnam |
| HPCA Dharamsala variants | …Dharamsala |
| Sawai Mansingh Stadium variants | Sawai Mansingh Stadium, Jaipur |
| PCA Mohali / IS Bindra Stadium variants | Punjab Cricket Association IS Bindra Stadium, Mohali |
| Maharaja Yadavindra Singh Stadium, Mullanpur / New Chandigarh | …New Chandigarh |
| Sheikh Zayed Stadium / Zayed Cricket Stadium, Abu Dhabi | Zayed Cricket Stadium, Abu Dhabi |
| Subrata Roy Sahara Stadium / Maharashtra Cricket Association Stadium(, Pune) | Maharashtra Cricket Association Stadium, Pune |

**Not merged:** `Sardar Patel Stadium, Motera` vs `Narendra Modi Stadium, Ahmedabad` (Motera was demolished and rebuilt). Raw `venue` is always retained for audit.

## How `final_score` was created

For every `(match_id, innings)`:

1. Order deliveries by original file order (`_source_order`), then `delivery_seq = 1..n`.
2. `final_score = last(team_runs)` in that order.

`ball_no` is **not** unique inside an over when extras occur (e.g. a wide and the legal ball can both be labelled `0.3`), so chronology uses `delivery_seq`, not `ball_no` uniqueness.

## Real-time features (information up to and including the current ball)

| Feature | Definition |
|---|---|
| `current_score` | `team_runs` after this delivery |
| `wickets_lost` | `team_wicket` after this delivery |
| `balls_completed` | `team_balls` (legal balls) |
| `balls_remaining` | `max(0, 120 - balls_completed)` |
| `overs_completed` | `balls_completed / 6` |
| `current_run_rate` | `(current_score * 6) / balls_completed` if `balls_completed > 0`, else `0` |
| `runs_last_6_balls` | rolling sum of `runs_total` over the last 6 **deliveries** in this innings (`min_periods=1`) |
| `runs_last_12_balls` | same, window 12 |
| `runs_last_18_balls` | same, window 18 |

Windows never look ahead and never cross into another innings or match.

Recent-run windows count **deliveries** (rows), including extras, because extras still add runs and occupy a row in the ball-by-ball log.

## Leakage prevention

- No post-match result fields in the processed table.
- No current-match future balls in any feature.
- `final_score` is the **target only**, not a feature.
- No historical venue/team averages in this phase (those belong in Phase 3 and must use **strictly earlier matches**).
- Split is by **match year**, not by random balls. Every ball of a match has the same `split`.

## Train / validation / test

Using the match `year` column:

| Split | Years | Rule |
|---|---|---|
| `train` | 2008–2022 | inclusive |
| `validation` | 2023 | |
| `test` | 2024–2025 | |

Season labels such as `2007/08` and `2020/21` follow calendar `year` (2008 and 2020), so they fall in **train**.

## Intended later model columns

**Candidate features:** `batting_team_clean`, `bowling_team_clean`, `venue_clean`, `innings`, `current_score`, `wickets_lost`, `balls_completed`, `balls_remaining`, `overs_completed`, `current_run_rate`, `runs_last_6_balls`, `runs_last_12_balls`, `runs_last_18_balls`, plus toss fields if used.

**Target:** `final_score`

**Not features:** `final_score` (except as y), raw result columns (already dropped), any Phase 3 history built from the current or future matches.

## Validation the script runs

- No `innings > 2`
- Unique `(match_id, innings, delivery_seq)`
- `final_score` present
- `current_score <= final_score`
- `balls_remaining >= 0`
- `wickets_lost` in 0–10
- One split per match; train/validation/test match IDs do not overlap
- SHA-256 of `IPL.csv` unchanged after the run
