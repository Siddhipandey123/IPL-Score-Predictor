"""
Phase 2: leakage-safe IPL innings dataset.

Reads raw IPL.csv (never writes to it). Writes data/processed/innings_clean.csv.
Does not train models.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "IPL.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "innings_clean.csv"

LEGAL_BALLS_PER_T20_INNINGS = 120

TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Formatting / city-suffix / punctuation only, plus a few well-documented
# same-ground official renames. Rebuilt or distinct grounds are left separate.
VENUE_NAME_MAP = {
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens": "Eden Gardens, Kolkata",
    "Eden Gardens, Kolkata": "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium, Delhi",
    # Official rename of the same Delhi ground (Feroz Shah Kotla).
    "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk, Chennai",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk, Chennai",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Brabourne Stadium": "Brabourne Stadium, Mumbai",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium, Mumbai",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy, Mumbai",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy, Mumbai",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": (
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"
    ),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": (
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"
    ),
    "Himachal Pradesh Cricket Association Stadium": (
        "Himachal Pradesh Cricket Association Stadium, Dharamsala"
    ),
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": (
        "Himachal Pradesh Cricket Association Stadium, Dharamsala"
    ),
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium, Pune",
    # Same Gahunje/Pune ground; former commercial name.
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali"
    ),
    "Punjab Cricket Association IS Bindra Stadium": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali"
    ),
    "Punjab Cricket Association IS Bindra Stadium, Mohali": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali"
    ),
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali"
    ),
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": (
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh"
    ),
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": (
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh"
    ),
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium, Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "Zayed Cricket Stadium, Abu Dhabi",
}

# Intentionally NOT merged (rebuilt / not treated as the same playing venue):
#   Sardar Patel Stadium, Motera  vs  Narendra Modi Stadium, Ahmedabad
UNMERGED_VENUE_NOTES = [
    "Sardar Patel Stadium, Motera vs Narendra Modi Stadium, Ahmedabad: Motera was demolished and rebuilt; kept separate.",
]

LEAKAGE_COLUMNS = [
    "player_of_match",
    "match_won_by",
    "win_outcome",
    "result_type",
    "superover_winner",
    "method",
    "runs_target",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_split(year: int) -> str:
    if year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    if year >= 2024:
        return "test"
    raise ValueError(f"Unexpected year: {year}")


def rolling_runs(group: pd.Series, window: int) -> pd.Series:
    return group.rolling(window=window, min_periods=1).sum()


def prepare() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    hash_before = sha256_file(RAW_PATH)
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    n_rows_raw = len(raw)
    n_matches_raw = raw["match_id"].nunique()

    # Preserve original file order for correct innings chronology.
    raw = raw.copy()
    raw["_source_order"] = range(len(raw))

    super_over_mask = raw["innings"] > 2
    n_super_over_rows = int(super_over_mask.sum())
    n_super_over_matches = int(raw.loc[super_over_mask, "match_id"].nunique())
    n_super_over_innings = int(
        raw.loc[super_over_mask, ["match_id", "innings"]].drop_duplicates().shape[0]
    )

    dl_match_ids = set(raw.loc[raw["method"].notna(), "match_id"].unique())
    no_result_match_ids = set(raw.loc[raw["result_type"] == "no result", "match_id"].unique())
    overlap_dl_nr = dl_match_ids & no_result_match_ids

    n_dl_matches = len(dl_match_ids)
    n_nr_matches = len(no_result_match_ids)
    n_dl_rows = int(raw["match_id"].isin(dl_match_ids).sum())
    n_nr_rows = int(raw["match_id"].isin(no_result_match_ids).sum())
    n_dl_rows_main = int(
        raw.loc[raw["match_id"].isin(dl_match_ids) & (raw["innings"] <= 2)].shape[0]
    )
    n_nr_rows_main = int(
        raw.loc[raw["match_id"].isin(no_result_match_ids) & (raw["innings"] <= 2)].shape[0]
    )

    work = raw.loc[~super_over_mask].copy()
    excluded_match_ids = dl_match_ids | no_result_match_ids
    work = work.loc[~work["match_id"].isin(excluded_match_ids)].copy()

    work = work.sort_values(["match_id", "innings", "_source_order"], kind="mergesort")
    work["delivery_seq"] = work.groupby(["match_id", "innings"], sort=False).cumcount() + 1

    last_score = (
        work.sort_values(["match_id", "innings", "delivery_seq"], kind="mergesort")
        .groupby(["match_id", "innings"], sort=False)["team_runs"]
        .transform("last")
    )
    work["final_score"] = last_score.astype(int)

    work["batting_team_clean"] = work["batting_team"].map(lambda x: TEAM_NAME_MAP.get(x, x))
    work["bowling_team_clean"] = work["bowling_team"].map(lambda x: TEAM_NAME_MAP.get(x, x))
    work["venue_clean"] = work["venue"].map(lambda x: VENUE_NAME_MAP.get(x, x))

    work["current_score"] = work["team_runs"].astype(int)
    work["wickets_lost"] = work["team_wicket"].astype(int)
    work["balls_completed"] = work["team_balls"].astype(int)
    work["balls_remaining"] = (LEGAL_BALLS_PER_T20_INNINGS - work["balls_completed"]).clip(lower=0)
    work["overs_completed"] = (work["balls_completed"] / 6.0).round(4)
    work["current_run_rate"] = 0.0
    nonzero_balls = work["balls_completed"] > 0
    work.loc[nonzero_balls, "current_run_rate"] = (
        work.loc[nonzero_balls, "current_score"] * 6.0 / work.loc[nonzero_balls, "balls_completed"]
    ).round(4)

    grouped_runs = work.groupby(["match_id", "innings"], sort=False)["runs_total"]
    work["runs_last_6_balls"] = grouped_runs.transform(lambda s: rolling_runs(s, 6)).astype(int)
    work["runs_last_12_balls"] = grouped_runs.transform(lambda s: rolling_runs(s, 12)).astype(int)
    work["runs_last_18_balls"] = grouped_runs.transform(lambda s: rolling_runs(s, 18)).astype(int)

    match_year = work.groupby("match_id")["year"].transform("first")
    work["split"] = match_year.map(assign_split)

    drop_cols = [c for c in LEAKAGE_COLUMNS if c in work.columns]
    drop_cols += ["Unnamed: 0", "_source_order"]
    work = work.drop(columns=drop_cols, errors="ignore")

    front = [
        "match_id",
        "date",
        "year",
        "season",
        "split",
        "innings",
        "delivery_seq",
        "batting_team",
        "batting_team_clean",
        "bowling_team",
        "bowling_team_clean",
        "venue",
        "venue_clean",
        "city",
        "over",
        "ball",
        "ball_no",
        "current_score",
        "wickets_lost",
        "balls_completed",
        "balls_remaining",
        "overs_completed",
        "current_run_rate",
        "runs_last_6_balls",
        "runs_last_12_balls",
        "runs_last_18_balls",
        "final_score",
        "team_runs",
        "team_balls",
        "team_wicket",
        "runs_total",
        "valid_ball",
    ]
    front = [c for c in front if c in work.columns]
    rest = [c for c in work.columns if c not in front]
    work = work[front + rest]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work.to_csv(OUT_PATH, index=False)

    hash_after = sha256_file(RAW_PATH)
    raw_unchanged = hash_before == hash_after

    # --- validation ---
    checks = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    record("innings_le_2", bool((work["innings"] <= 2).all()), f"max innings={work['innings'].max()}")
    record(
        "no_super_over_innings",
        bool((work["innings"] >= 1).all() and set(work["innings"].unique()) <= {1, 2}),
        f"unique innings={sorted(work['innings'].unique().tolist())}",
    )

    dup_ball_no = int(work.duplicated(["match_id", "innings", "ball_no"]).sum())
    # Extras can reuse the same ball_no (e.g. wide then legal 0.3). Chronology is delivery_seq.
    record(
        "unique_match_innings_delivery_seq",
        not work.duplicated(["match_id", "innings", "delivery_seq"]).any(),
        f"duplicate delivery_seq={int(work.duplicated(['match_id','innings','delivery_seq']).sum())}; "
        f"duplicate (match_id,innings,ball_no) extra-related={dup_ball_no}",
    )
    record("final_score_present", bool(work["final_score"].notna().all()), f"na={int(work['final_score'].isna().sum())}")
    score_ok = bool((work["current_score"] <= work["final_score"]).all())
    n_score_bad = int((work["current_score"] > work["final_score"]).sum())
    record("current_score_le_final_score", score_ok, f"violations={n_score_bad}")
    record(
        "balls_remaining_non_negative",
        bool((work["balls_remaining"] >= 0).all()),
        f"min={int(work['balls_remaining'].min())}",
    )
    wicket_ok = bool(work["wickets_lost"].between(0, 10).all())
    record(
        "wickets_lost_0_to_10",
        wicket_ok,
        f"min={int(work['wickets_lost'].min())} max={int(work['wickets_lost'].max())}",
    )

    match_split = work.groupby("match_id")["split"].nunique()
    record("one_split_per_match", bool((match_split == 1).all()), f"matches with mixed split={int((match_split > 1).sum())}")

    ids = {
        s: set(work.loc[work["split"] == s, "match_id"].unique())
        for s in ["train", "validation", "test"]
    }
    overlap = (
        len(ids["train"] & ids["validation"])
        + len(ids["train"] & ids["test"])
        + len(ids["validation"] & ids["test"])
    )
    record("split_matches_no_overlap", overlap == 0, f"overlapping_match_ids={overlap}")
    record("ipl_csv_unchanged", raw_unchanged, f"sha256={hash_after}")
    record("no_dl_method_column", "method" not in work.columns, "")
    record("no_result_type_column", "result_type" not in work.columns, "")
    record("no_runs_target_column", "runs_target" not in work.columns, "")

    n_rows_out = len(work)
    n_matches_out = work["match_id"].nunique()
    split_rows = work["split"].value_counts().to_dict()
    split_matches = work.groupby("split")["match_id"].nunique().to_dict()

    print("=" * 72)
    print("PHASE 2 VALIDATION REPORT")
    print("=" * 72)
    print(f"raw_file: {RAW_PATH}")
    print(f"processed_file: {OUT_PATH}")
    print(f"rows_before_cleaning: {n_rows_raw}")
    print(f"matches_before_cleaning: {n_matches_raw}")
    print(f"rows_after_cleaning: {n_rows_out}")
    print(f"matches_after_cleaning: {n_matches_out}")
    print("--- exclusions (from raw IPL.csv; file itself was not edited) ---")
    print(
        f"super_over_innings_gt_2: rows={n_super_over_rows} "
        f"innings_units={n_super_over_innings} matches_containing_them={n_super_over_matches}"
    )
    print(
        f"D/L matches excluded: matches={n_dl_matches} "
        f"all_rows={n_dl_rows} innings_1_2_rows={n_dl_rows_main}"
    )
    print(
        f"no-result matches excluded: matches={n_nr_matches} "
        f"all_rows={n_nr_rows} innings_1_2_rows={n_nr_rows_main}"
    )
    print(f"D/L and no-result match overlap: {len(overlap_dl_nr)}")
    print("--- split (by match year; all balls of a match stay together) ---")
    for name in ["train", "validation", "test"]:
        print(f"{name}: matches={split_matches.get(name, 0)} rows={split_rows.get(name, 0)}")
    print("--- checks ---")
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        extra = f" ({detail})" if detail else ""
        print(f"{status}: {name}{extra}")
    print(f"canonical_teams: {sorted(set(work['batting_team_clean']) | set(work['bowling_team_clean']))}")
    print(f"canonical_venues: {work['venue_clean'].nunique()} (raw venues in output={work['venue'].nunique()})")
    print("=" * 72)
    if not all_ok:
        raise SystemExit("Phase 2 validation failed.")
    print("Phase 2 validation passed. IPL.csv was not modified.")


if __name__ == "__main__":
    prepare()
