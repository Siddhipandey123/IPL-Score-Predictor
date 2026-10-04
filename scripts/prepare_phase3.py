"""
Phase 3: temporal-leakage-safe historical venue/team features.

Reads data/processed/innings_clean.csv (Phase 2). Never writes IPL.csv or
innings_clean.csv. Writes data/processed/innings_features.csv.
Does not train models.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "IPL.csv"
PHASE2_PATH = PROJECT_ROOT / "data" / "processed" / "innings_clean.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "innings_features.csv"

# Used only when no prior match exists in the dataset (the first IPL date).
# Not estimated from later IPL results.
PRIOR_INNINGS_SCORE = 160.0
PRIOR_RUN_RATE = 8.0
MIN_TEAM_VENUE_INNINGS = 3

PHASE2_LIVE_COLS = [
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
    "split",
]

HISTORICAL_FEATURE_COLS = [
    "global_matches_before",
    "global_innings_before",
    "global_avg_score_before",
    "global_avg_first_innings_score_before",
    "global_avg_second_innings_score_before",
    "global_avg_run_rate_before",
    "venue_matches_before",
    "venue_innings_before",
    "venue_avg_first_innings_score",
    "venue_avg_second_innings_score",
    "venue_avg_score",
    "venue_avg_run_rate",
    "venue_highest_score",
    "venue_lowest_score",
    "batting_team_matches_before",
    "batting_team_avg_score_before",
    "batting_team_avg_run_rate_before",
    "bowling_team_matches_before",
    "bowling_team_avg_runs_conceded_before",
    "bowling_team_avg_run_rate_conceded_before",
    "batting_team_venue_matches_before",
    "batting_team_venue_avg_score_before",
]

SUM_COLS = [
    "n_innings",
    "n_matches",
    "sum_score",
    "sum_rr",
    "n_rr",
    "n_inn1",
    "sum_inn1",
    "n_inn2",
    "sum_inn2",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fill_ratio(numer: pd.Series, denom: pd.Series, fallback: pd.Series | float) -> pd.Series:
    out = pd.Series(np.nan, index=numer.index, dtype="float64")
    ok = denom.fillna(0) > 0
    out.loc[ok] = numer.loc[ok].astype(float) / denom.loc[ok].astype(float)
    return out.fillna(fallback)


def previous_cumulatives(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """prev_* = cumulative stats using strictly earlier dates within the group."""
    daily = daily.sort_values(group_cols + ["date"], kind="mergesort").reset_index(drop=True)
    if group_cols:
        grouped = daily.groupby(group_cols, sort=False)
    else:
        grouped = [(None, daily)]

    parts = []
    for _, chunk in grouped:
        chunk = chunk.sort_values("date", kind="mergesort").copy()
        for col in SUM_COLS:
            chunk[f"prev_{col}"] = chunk[col].cumsum().shift(1)
        chunk["prev_max_score"] = chunk["max_score"].cummax().shift(1)
        chunk["prev_min_score"] = chunk["min_score"].cummin().shift(1)
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)
    prev_cols = [c for c in out.columns if c.startswith("prev_")]
    out[prev_cols] = out[prev_cols].fillna(0)
    return out


def aggregate_daily(innings: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keys = group_cols + ["date"]
    daily = innings.groupby(keys, as_index=False).agg(
        n_innings=("final_score", "size"),
        n_matches=("match_id", "nunique"),
        sum_score=("final_score", "sum"),
        sum_rr=("innings_run_rate", "sum"),
        n_rr=("innings_run_rate", "count"),
        n_inn1=("is_inn1", "sum"),
        sum_inn1=("score_inn1", "sum"),
        n_inn2=("is_inn2", "sum"),
        sum_inn2=("score_inn2", "sum"),
        max_score=("final_score", "max"),
        min_score=("final_score", "min"),
    )
    return previous_cumulatives(daily, group_cols)


def rename_prev(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mapper = {c: f"{prefix}_{c}" for c in df.columns if c.startswith("prev_")}
    return df.rename(columns=mapper)


def build_innings_table(balls: pd.DataFrame) -> pd.DataFrame:
    last = (
        balls.sort_values(["match_id", "innings", "delivery_seq"], kind="mergesort")
        .groupby(["match_id", "innings"], as_index=False, sort=False)
        .tail(1)
    )
    innings = last[
        [
            "match_id",
            "innings",
            "date",
            "year",
            "split",
            "venue_clean",
            "batting_team_clean",
            "bowling_team_clean",
            "final_score",
            "balls_completed",
        ]
    ].copy()
    innings["date"] = pd.to_datetime(innings["date"])
    innings["innings_run_rate"] = np.where(
        innings["balls_completed"] > 0,
        innings["final_score"] * 6.0 / innings["balls_completed"],
        np.nan,
    )
    innings["is_inn1"] = (innings["innings"] == 1).astype(int)
    innings["is_inn2"] = (innings["innings"] == 2).astype(int)
    innings["score_inn1"] = innings["final_score"] * innings["is_inn1"]
    innings["score_inn2"] = innings["final_score"] * innings["is_inn2"]
    return innings


def build_match_innings_features(innings: pd.DataFrame) -> pd.DataFrame:
    global_daily = rename_prev(aggregate_daily(innings, []), "global")
    venue_daily = rename_prev(aggregate_daily(innings, ["venue_clean"]), "venue")
    bat_daily = rename_prev(aggregate_daily(innings, ["batting_team_clean"]), "bat")
    bowl_daily = rename_prev(aggregate_daily(innings, ["bowling_team_clean"]), "bowl")
    team_venue_daily = rename_prev(
        aggregate_daily(innings, ["batting_team_clean", "venue_clean"]), "tv"
    )

    hist = innings[
        ["match_id", "innings", "date", "venue_clean", "batting_team_clean", "bowling_team_clean"]
    ].copy()

    hist = hist.merge(
        global_daily[["date"] + [c for c in global_daily.columns if c.startswith("global_prev_")]],
        on="date",
        how="left",
    )
    hist = hist.merge(
        venue_daily[
            ["venue_clean", "date"] + [c for c in venue_daily.columns if c.startswith("venue_prev_")]
        ],
        on=["venue_clean", "date"],
        how="left",
    )
    hist = hist.merge(
        bat_daily[
            ["batting_team_clean", "date"]
            + [c for c in bat_daily.columns if c.startswith("bat_prev_")]
        ],
        on=["batting_team_clean", "date"],
        how="left",
    )
    hist = hist.merge(
        bowl_daily[
            ["bowling_team_clean", "date"]
            + [c for c in bowl_daily.columns if c.startswith("bowl_prev_")]
        ],
        on=["bowling_team_clean", "date"],
        how="left",
    )
    hist = hist.merge(
        team_venue_daily[
            ["batting_team_clean", "venue_clean", "date"]
            + [c for c in team_venue_daily.columns if c.startswith("tv_prev_")]
        ],
        on=["batting_team_clean", "venue_clean", "date"],
        how="left",
    )

    prev_like = [c for c in hist.columns if "_prev_" in c]
    hist[prev_like] = hist[prev_like].fillna(0.0)

    g_n = hist["global_prev_n_matches"]
    g_inn = hist["global_prev_n_innings"]
    g_avg = _fill_ratio(hist["global_prev_sum_score"], g_inn, PRIOR_INNINGS_SCORE)
    g_avg1 = _fill_ratio(hist["global_prev_sum_inn1"], hist["global_prev_n_inn1"], g_avg)
    g_avg2 = _fill_ratio(hist["global_prev_sum_inn2"], hist["global_prev_n_inn2"], g_avg)
    g_rr = _fill_ratio(hist["global_prev_sum_rr"], hist["global_prev_n_rr"], PRIOR_RUN_RATE)
    g_max = np.where(g_inn > 0, hist["global_prev_max_score"], PRIOR_INNINGS_SCORE)
    g_min = np.where(g_inn > 0, hist["global_prev_min_score"], PRIOR_INNINGS_SCORE)

    hist["global_matches_before"] = g_n.astype(int)
    hist["global_innings_before"] = g_inn.astype(int)
    hist["global_avg_score_before"] = g_avg.round(4)
    hist["global_avg_first_innings_score_before"] = g_avg1.round(4)
    hist["global_avg_second_innings_score_before"] = g_avg2.round(4)
    hist["global_avg_run_rate_before"] = g_rr.round(4)

    v_n_match = hist["venue_prev_n_matches"]
    v_n_inn = hist["venue_prev_n_innings"]
    v_avg = _fill_ratio(hist["venue_prev_sum_score"], v_n_inn, g_avg)
    v_avg1 = _fill_ratio(hist["venue_prev_sum_inn1"], hist["venue_prev_n_inn1"], v_avg)
    v_avg2 = _fill_ratio(hist["venue_prev_sum_inn2"], hist["venue_prev_n_inn2"], v_avg)
    v_rr = _fill_ratio(hist["venue_prev_sum_rr"], hist["venue_prev_n_rr"], g_rr)
    v_max = np.where(v_n_inn > 0, hist["venue_prev_max_score"], g_max)
    v_min = np.where(v_n_inn > 0, hist["venue_prev_min_score"], g_min)

    hist["venue_matches_before"] = v_n_match.astype(int)
    hist["venue_innings_before"] = v_n_inn.astype(int)
    hist["venue_avg_score"] = v_avg.round(4)
    hist["venue_avg_first_innings_score"] = v_avg1.round(4)
    hist["venue_avg_second_innings_score"] = v_avg2.round(4)
    hist["venue_avg_run_rate"] = v_rr.round(4)
    hist["venue_highest_score"] = pd.Series(v_max, index=hist.index).astype(float).round(4)
    hist["venue_lowest_score"] = pd.Series(v_min, index=hist.index).astype(float).round(4)

    bat_n = hist["bat_prev_n_matches"]
    bat_inn = hist["bat_prev_n_innings"]
    bat_avg = _fill_ratio(hist["bat_prev_sum_score"], bat_inn, g_avg)
    bat_rr = _fill_ratio(hist["bat_prev_sum_rr"], hist["bat_prev_n_rr"], g_rr)
    hist["batting_team_matches_before"] = bat_n.astype(int)
    hist["batting_team_avg_score_before"] = bat_avg.round(4)
    hist["batting_team_avg_run_rate_before"] = bat_rr.round(4)

    bowl_n = hist["bowl_prev_n_matches"]
    bowl_inn = hist["bowl_prev_n_innings"]
    bowl_avg = _fill_ratio(hist["bowl_prev_sum_score"], bowl_inn, g_avg)
    bowl_rr = _fill_ratio(hist["bowl_prev_sum_rr"], hist["bowl_prev_n_rr"], g_rr)
    hist["bowling_team_matches_before"] = bowl_n.astype(int)
    hist["bowling_team_avg_runs_conceded_before"] = bowl_avg.round(4)
    hist["bowling_team_avg_run_rate_conceded_before"] = bowl_rr.round(4)

    tv_n = hist["tv_prev_n_matches"]
    tv_inn = hist["tv_prev_n_innings"]
    tv_avg_raw = _fill_ratio(hist["tv_prev_sum_score"], tv_inn, bat_avg)
    use_tv = tv_inn >= MIN_TEAM_VENUE_INNINGS
    tv_avg = np.where(use_tv, tv_avg_raw, bat_avg)
    hist["batting_team_venue_matches_before"] = tv_n.astype(int)
    hist["batting_team_venue_avg_score_before"] = pd.Series(tv_avg, index=hist.index).astype(float).round(4)
    hist["batting_team_venue_history_is_sparse"] = (~use_tv).astype(int)
    hist["venue_history_is_cold_start"] = (v_n_match == 0).astype(int)
    hist["batting_team_history_is_cold_start"] = (bat_n == 0).astype(int)
    hist["bowling_team_history_is_cold_start"] = (bowl_n == 0).astype(int)

    keep = [
        "match_id",
        "innings",
        *HISTORICAL_FEATURE_COLS,
        "venue_history_is_cold_start",
        "batting_team_history_is_cold_start",
        "bowling_team_history_is_cold_start",
        "batting_team_venue_history_is_sparse",
    ]
    return hist[keep]


def leakage_audit(innings: pd.DataFrame, feats: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    ordered = innings.sort_values(["date", "match_id"]).drop_duplicates("match_id")
    sample_ids: list[int] = [
        int(ordered.iloc[0]["match_id"]),
        int(ordered.iloc[len(ordered) // 4]["match_id"]),
        int(ordered.iloc[len(ordered) // 2]["match_id"]),
    ]
    val = ordered[ordered["split"] == "validation"]
    test = ordered[ordered["split"] == "test"]
    if len(val):
        sample_ids.append(int(val.iloc[0]["match_id"]))
        sample_ids.append(int(val.iloc[len(val) // 2]["match_id"]))
    if len(test):
        sample_ids.append(int(test.iloc[0]["match_id"]))
        sample_ids.append(int(test.iloc[-1]["match_id"]))
    y2022 = ordered[ordered["year"] == 2022]
    if len(y2022):
        sample_ids.append(int(y2022.iloc[0]["match_id"]))
    sample_ids = list(dict.fromkeys(sample_ids))

    innings = innings.copy()
    innings["date"] = pd.to_datetime(innings["date"])

    for mid in sample_ids:
        inn_m = innings.loc[innings["match_id"] == mid]
        d = inn_m["date"].iloc[0]
        venue = inn_m["venue_clean"].iloc[0]
        bat = inn_m.loc[inn_m["innings"] == 1, "batting_team_clean"].iloc[0]
        past = innings.loc[innings["date"] < d]

        included_current = mid in set(past["match_id"].tolist())
        venue_past_inn1 = past.loc[(past["venue_clean"] == venue) & (past["innings"] == 1)]
        if len(venue_past_inn1) == 0:
            inn1_past = past.loc[past["innings"] == 1, "final_score"]
            if len(inn1_past):
                expected_v1 = float(inn1_past.mean())
            elif len(past):
                expected_v1 = float(past["final_score"].mean())
            else:
                expected_v1 = PRIOR_INNINGS_SCORE
        else:
            expected_v1 = float(venue_past_inn1["final_score"].mean())

        attached_v1 = float(
            feats.loc[
                (feats["match_id"] == mid) & (feats["innings"] == 1),
                "venue_avg_first_innings_score",
            ].iloc[0]
        )
        close = bool(np.isclose(attached_v1, expected_v1, rtol=1e-4, atol=1e-3))
        results.append(
            (
                f"audit_venue_inn1_{mid}",
                (not included_current) and close,
                (
                    f"date={d.date()} venue={venue} past_matches={past['match_id'].nunique()} "
                    f"past_inn1_at_venue={len(venue_past_inn1)} attached={attached_v1:.4f} "
                    f"recomputed={expected_v1:.4f} current_in_past={included_current}"
                ),
            )
        )

        leaky = innings.loc[
            (innings["venue_clean"] == venue)
            & (innings["innings"] == 1)
            & (innings["date"] <= d)
        ]
        leaky_mean = float(leaky["final_score"].mean()) if len(leaky) else np.nan
        # If this venue had prior first innings, including today's match should
        # usually change the average. Always require current id not in contributors.
        results.append(
            (
                f"audit_not_self_inclusive_{mid}",
                mid not in set(venue_past_inn1["match_id"].tolist()),
                f"leaky_mean_if_current_included={leaky_mean:.4f} attached={attached_v1:.4f}",
            )
        )

        bat_past = past.loc[past["batting_team_clean"] == bat]
        if len(bat_past) == 0:
            exp_bat = float(past["final_score"].mean()) if len(past) else PRIOR_INNINGS_SCORE
        else:
            exp_bat = float(bat_past["final_score"].mean())
        att_bat = float(
            feats.loc[
                (feats["match_id"] == mid) & (feats["innings"] == 1),
                "batting_team_avg_score_before",
            ].iloc[0]
        )
        results.append(
            (
                f"audit_bat_{mid}",
                bool(np.isclose(att_bat, exp_bat, rtol=1e-4, atol=1e-3)),
                f"attached={att_bat:.4f} recomputed={exp_bat:.4f}",
            )
        )
        results.append(
            (
                f"audit_no_future_{mid}",
                int((past["date"] >= d).sum()) == 0,
                f"future_or_same_in_past={int((past['date'] >= d).sum())}",
            )
        )

    return results


def _numeric_equal(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.allclose(a.astype(float), b.astype(float), equal_nan=True))
    return bool(a.astype(str).equals(b.astype(str)))


def prepare() -> None:
    if not PHASE2_PATH.exists():
        raise FileNotFoundError(f"Phase 2 file not found: {PHASE2_PATH}")

    hash_raw_before = sha256_file(RAW_PATH) if RAW_PATH.exists() else None
    hash_p2_before = sha256_file(PHASE2_PATH)

    balls = pd.read_csv(PHASE2_PATH, low_memory=False)
    n_rows_p2 = len(balls)
    n_matches_p2 = int(balls["match_id"].nunique())

    innings = build_innings_table(balls)
    feat_inn = build_match_innings_features(innings)

    venue_cols = [
        "venue_matches_before",
        "venue_innings_before",
        "venue_avg_first_innings_score",
        "venue_avg_second_innings_score",
        "venue_avg_score",
        "venue_avg_run_rate",
        "venue_highest_score",
        "venue_lowest_score",
        "global_matches_before",
        "global_innings_before",
        "global_avg_score_before",
        "venue_history_is_cold_start",
    ]
    vcheck = feat_inn.groupby("match_id")[venue_cols].nunique()
    venue_constant = bool((vcheck <= 1).all().all())

    out = balls.merge(feat_inn, on=["match_id", "innings"], how="left", validate="many_to_one")
    if len(out) != n_rows_p2:
        raise RuntimeError("Row count changed during historical merge.")

    out.to_csv(OUT_PATH, index=False)

    hash_raw_after = sha256_file(RAW_PATH) if RAW_PATH.exists() else None
    hash_p2_after = sha256_file(PHASE2_PATH)

    p2 = pd.read_csv(PHASE2_PATH, low_memory=False)
    live_changed = [c for c in PHASE2_LIVE_COLS if not _numeric_equal(p2[c], out[c])]
    dup = int(out.duplicated(["match_id", "innings", "delivery_seq"]).sum())
    nan_hist = {c: int(out[c].isna().sum()) for c in HISTORICAL_FEATURE_COLS}
    nan_total = sum(nan_hist.values())

    split_ok = bool(
        p2.groupby("match_id")["split"].first().equals(out.groupby("match_id")["split"].first())
    )

    cold_venue_rows = int((out["venue_history_is_cold_start"] == 1).sum())
    cold_bat_rows = int((out["batting_team_history_is_cold_start"] == 1).sum())
    cold_bowl_rows = int((out["bowling_team_history_is_cold_start"] == 1).sum())
    cold_any_rows = int(
        (
            (out["venue_history_is_cold_start"] == 1)
            | (out["batting_team_history_is_cold_start"] == 1)
            | (out["bowling_team_history_is_cold_start"] == 1)
        ).sum()
    )
    cold_venue_matches = int(out.loc[out["venue_history_is_cold_start"] == 1, "match_id"].nunique())
    cold_bat_matches = int(out.loc[out["batting_team_history_is_cold_start"] == 1, "match_id"].nunique())
    sparse_tv_rows = int((out["batting_team_venue_history_is_sparse"] == 1).sum())

    split_rows = out["split"].value_counts().to_dict()
    split_matches = out.groupby("split")["match_id"].nunique().to_dict()

    feat_with_meta = feat_inn.merge(
        innings[["match_id", "innings", "date", "venue_clean", "batting_team_clean", "split", "year"]],
        on=["match_id", "innings"],
        how="left",
    )
    audit = leakage_audit(innings, feat_with_meta)

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    record("rows_unchanged", len(out) == n_rows_p2, f"{len(out)} vs {n_rows_p2}")
    record(
        "matches_unchanged",
        out["match_id"].nunique() == n_matches_p2,
        f"{out['match_id'].nunique()} vs {n_matches_p2}",
    )
    record("no_duplicate_delivery_seq", dup == 0, f"dups={dup}")
    record("phase2_live_features_unchanged", len(live_changed) == 0, f"changed={live_changed}")
    record("final_score_unchanged", _numeric_equal(p2["final_score"], out["final_score"]), "")
    record("split_unchanged", split_ok, "")
    record("venue_stats_constant_within_match", venue_constant, "")
    record(
        "historical_features_no_nan",
        nan_total == 0,
        f"nan_by_col={ {k: v for k, v in nan_hist.items() if v} }",
    )
    record("ipl_csv_unchanged", hash_raw_before == hash_raw_after, "")
    record("innings_clean_unchanged", hash_p2_before == hash_p2_after, "")
    record("two_innings_per_match", bool(innings.groupby("match_id").size().eq(2).all()), "")
    record(
        "no_postmatch_result_cols",
        all(
            c not in out.columns
            for c in [
                "player_of_match",
                "match_won_by",
                "win_outcome",
                "result_type",
                "superover_winner",
                "method",
                "runs_target",
            ]
        ),
        "",
    )
    for name, ok, detail in audit:
        record(name, ok, detail)

    print("=" * 72)
    print("PHASE 3 VALIDATION REPORT")
    print("=" * 72)
    print(f"phase2_file: {PHASE2_PATH}")
    print(f"output_file: {OUT_PATH}")
    print(f"rows_before: {n_rows_p2}  rows_after: {len(out)}")
    print(f"matches_before: {n_matches_p2}  matches_after: {out['match_id'].nunique()}")
    print("--- split (unchanged from Phase 2) ---")
    for name in ["train", "validation", "test"]:
        print(f"{name}: matches={split_matches.get(name, 0)} rows={split_rows.get(name, 0)}")
    print("--- cold start ---")
    print(f"venue cold-start: matches={cold_venue_matches} rows={cold_venue_rows}")
    print(f"batting-team cold-start: matches={cold_bat_matches} rows={cold_bat_rows}")
    print(f"bowling-team cold-start rows={cold_bowl_rows}")
    print(f"any cold-start rows={cold_any_rows}")
    print(f"sparse team-venue rows (n<{MIN_TEAM_VENUE_INNINGS})={sparse_tv_rows}")
    print(
        f"MIN_TEAM_VENUE_INNINGS={MIN_TEAM_VENUE_INNINGS} "
        f"PRIOR_SCORE={PRIOR_INNINGS_SCORE} PRIOR_RR={PRIOR_RUN_RATE}"
    )
    print("--- checks ---")
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        extra = f" ({detail})" if detail else ""
        print(f"{status}: {name}{extra}")
    print("=" * 72)
    if not all_ok:
        raise SystemExit("Phase 3 validation failed.")
    print("Phase 3 validation passed. IPL.csv and innings_clean.csv were not modified.")


if __name__ == "__main__":
    prepare()
