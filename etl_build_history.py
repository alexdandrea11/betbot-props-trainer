# etl_build_history.py
# Builds per-market training tables from nfl_data_py weekly data.
# Outputs:
#   data/train_player_receptions.parquet
#   data/train_player_reception_yds.parquet
#   data/train_player_rush_yds.parquet
#   data/train_player_pass_yds.parquet

import os
from datetime import datetime
import pandas as pd
import numpy as np

# ---------------------- robust env parsing ----------------------
def _int_env(name: str, default: int) -> int:
    val = os.getenv(name, "")
    if val is None:
        return int(default)
    val = str(val).strip()
    if not val:
        return int(default)
    try:
        return int(val)
    except ValueError:
        return int(default)

SEASON_START = _int_env("SEASON_START", 2015)
SEASON_END   = _int_env("SEASON_END", datetime.utcnow().year)  # default: current year

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Markets configuration (aligns with train_models_regression.py)
MARKETS = {
    "player_receptions": {
        "y_col": "receptions",
        "aux_feats": ["targets"],
        "prefix": "receptions",
    },
    "player_reception_yds": {
        "y_col": "receiving_yards",
        "aux_feats": ["receptions","targets"],
        "prefix": "receiving_yards",
    },
    "player_rush_yds": {
        "y_col": "rushing_yards",
        "aux_feats": ["carries"],
        "prefix": "rushing_yards",
    },
    "player_pass_yds": {
        "y_col": "passing_yards",
        "aux_feats": [],
        "prefix": "passing_yards",
    },
}

# ---------------------- helpers ----------------------
def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _coerce_int(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="Int64")
    return safe_num(df[col]).astype("Int64").astype(float).astype(int)

def _fetch_weekly_by_year(year: int) -> pd.DataFrame:
    """Try to fetch weekly data for a single season; return empty DataFrame if unavailable."""
    import nfl_data_py as nfl
    try:
        df = nfl.import_weekly_data([year])
        df["season"] = _coerce_int(df, "season")
        df["week"]   = _coerce_int(df, "week")
        return df
    except Exception as e:
        print(f"[ETL] ⚠️ Skipping weekly for {year}: {e}")
        return pd.DataFrame()

def _fetch_schedule_by_year(year: int) -> pd.DataFrame:
    """Try to fetch schedule for a single season; return empty DataFrame if unavailable."""
    import nfl_data_py as nfl
    try:
        sch = nfl.import_schedules([year])
        return sch
    except Exception as e:
        print(f"[ETL] ⚠️ Skipping schedule for {year}: {e}")
        return pd.DataFrame()

def load_weekly(seasons):
    """Load weekly player data (year-by-year) and attach schedule info for days_rest and is_home."""
    # 1) Pull weekly per-year, skipping missing seasons
    weekly_parts = []
    available_years = []
    for y in seasons:
        df_y = _fetch_weekly_by_year(y)
        if not df_y.empty:
            weekly_parts.append(df_y)
            available_years.append(y)

    if not weekly_parts:
        raise RuntimeError("No weekly data available for any requested season.")

    df = pd.concat(weekly_parts, ignore_index=True)
    print(f"[ETL] Weekly rows across {len(available_years)} season(s): {len(df):,}")

    # 2) Standardize column names
    rename = {
        "player_display_name": "player",
        "player_name": "player",
        "recent_team": "team",
        "team": "team",
        "opponent_team": "opp",
        "opponent": "opp",
        "position": "position",
        "week": "week",
        "season": "season",
    }
    for k, v in rename.items():
        if k in df.columns and v != k:
            df = df.rename(columns={k: v})

    # Ensure core numeric stat columns exist
    base_stats = [
        "receptions", "receiving_yards", "targets",
        "rushing_yards", "carries",
        "passing_yards",
    ]
    for c in base_stats:
        if c in df.columns:
            df[c] = safe_num(df[c]).fillna(0.0)
        else:
            df[c] = 0.0

    # week / season as ints (already coerced but ensure)
    df["week"] = _coerce_int(df, "week")
    df["season"] = _coerce_int(df, "season")

    # 3) Pull schedules per available year (skip missing)
    sch_parts = []
    for y in available_years:
        sch_y = _fetch_schedule_by_year(y)
        if not sch_y.empty:
            sch_parts.append(sch_y)

    if sch_parts:
        sch = pd.concat(sch_parts, ignore_index=True)
    else:
        sch = pd.DataFrame()

    # 4) Attach schedule info for dates/home/away if available
    if not sch.empty:
        sch = sch.rename(columns={"home_team": "home", "away_team": "away"})
        # Pick a valid date column if present
        dtcol = None
        for candidate in ("gameday", "game_date", "start_time", "game_time", "game_date_time"):
            if candidate in sch.columns:
                dtcol = candidate
                break

        if dtcol:
            sch["game_date"] = pd.to_datetime(sch[dtcol], errors="coerce")
        else:
            sch["game_date"] = pd.NaT

        sch_small = sch[["season", "week", "home", "away", "game_date"]].copy()
        tmp = df.merge(sch_small, on=["season", "week"], how="left")

        # is_home flag
        tmp["is_home"] = np.where(tmp["team"] == tmp["home"], 1,
                           np.where(tmp["team"] == tmp["away"], 0, np.nan))

        # Derive opponent if missing
        if "opp" not in tmp.columns or tmp["opp"].isna().all():
            tmp["opp"] = np.where(tmp["is_home"] == 1, tmp["away"],
                           np.where(tmp["is_home"] == 0, tmp["home"], tmp.get("opp", "")))

        # days_rest
        tmp["game_date"] = pd.to_datetime(tmp["game_date"], errors="coerce")
        tmp = tmp.sort_values(["player", "season", "week"])
        tmp["prev_date"] = tmp.groupby("player")["game_date"].shift(1)
        tmp["days_rest"] = (tmp["game_date"] - tmp["prev_date"]).dt.days
        tmp["days_rest"] = tmp["days_rest"].fillna(7).clip(lower=3, upper=21)

        tmp["is_home"] = pd.to_numeric(tmp["is_home"], errors="coerce")
        tmp["days_rest"] = pd.to_numeric(tmp["days_rest"], errors="coerce").fillna(7)

        keep_cols = ["player", "team", "opp", "season", "week", "position", "is_home", "days_rest"] + base_stats
        tmp = tmp[keep_cols]
        return tmp.reset_index(drop=True)

    # 5) Fallback without schedules
    print("[ETL] ⚠️ No schedule data available; using defaults (is_home NaN, days_rest=7).")
    df["is_home"] = np.nan
    df["days_rest"] = 7
    if "opp" not in df.columns:
        df["opp"] = ""
    keep_cols = ["player", "team", "opp", "season", "week", "position", "is_home", "days_rest"] + base_stats
    return df[keep_cols].reset_index(drop=True)

def make_rolling_feats(df, col, groups=["player"], windows=(3, 5, 8), suffixes=("l3", "l5", "l8")):
    """Create shifted rolling means so current-week label isn't leaked."""
    out = df.copy().sort_values(groups + ["season", "week"])
    for w, suf in zip(windows, suffixes):
        out[f"{col}_{suf}"] = (
            out.groupby(groups)[col]
               .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
               .reset_index(level=groups, drop=True)
        )
    return out

def build_defense_allowed(df, y_col):
    """
    For each defense (team), compute rolling mean of the stat *allowed*,
    aggregating opponent players' y_col vs that defense by week.
    """
    tmp = df.copy()
    tmp["def_team"] = tmp["opp"]
    allowed = tmp.groupby(["season", "week", "def_team"], as_index=False)[y_col].sum()
    allowed = allowed.sort_values(["def_team", "season", "week"]).reset_index(drop=True)
    for w, suf in [(3, "l3"), (5, "l5"), (8, "l8")]:
        allowed[f"{y_col}_allowed_{suf}"] = (
            allowed.groupby("def_team")[y_col]
                   .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                   .reset_index(level="def_team", drop=True)
        )
    keep = ["season", "week", "def_team"] + [f"{y_col}_allowed_{suf}" for suf in ["l3", "l5", "l8"]]
    return allowed[keep]

def build_h2h_mean(df, y_col):
    """
    Player vs opponent: expanding mean of y_col (prior-only: shifted by 1).
    """
    tmp = df.sort_values(["player", "opp", "season", "week"]).copy()
    grp = tmp.groupby(["player", "opp"])
    tmp["h2h_mean_alltime"] = grp[y_col].apply(lambda s: s.shift(1).expanding(1).mean()).reset_index(level=[0,1], drop=True)
    return tmp[["player", "opp", "season", "week", "h2h_mean_alltime"]]

def build_table_for_market(df, market_key, spec):
    """
    Build feature table for one market.
    Produces columns your trainer expects (see train_models_regression.py).
    """
    y_col = spec["y_col"]
    out = df.copy()

    # Rolling means on target
    out = make_rolling_feats(out, y_col)

    # Rolling means on aux features (e.g., targets_l5)
    for aux in spec.get("aux_feats", []):
        if aux in out.columns:
            out = make_rolling_feats(out, aux)

    # Defense allowed rolling
    allowed = build_defense_allowed(out, y_col)
    out = out.merge(
        allowed,
        left_on=["season", "week", "opp"],
        right_on=["season", "week", "def_team"],
        how="left"
    ).drop(columns=["def_team"])

    # Head-to-head expanding mean
    h2h = build_h2h_mean(out, y_col)
    out = out.merge(h2h, on=["player", "opp", "season", "week"], how="left")

    # Clean NA for generated features
    for c in list(out.columns):
        if c.endswith(("_l3", "_l5", "_l8", "_alltime")):
            vals = pd.to_numeric(out[c], errors="coerce")
            if np.isfinite(vals).any():
                out[c] = vals.fillna(vals.median())
            else:
                out[c] = 0.0

    # Final selection + rename target to y
    out["y"] = pd.to_numeric(out[y_col], errors="coerce").fillna(0.0).astype(float)

    keep = [
        "player", "team", "opp", "season", "week", "position", "is_home", "days_rest", "y",
        f"{y_col}_l3", f"{y_col}_l5", f"{y_col}_l8",
        "h2h_mean_alltime",
    ]

    # Aux (we only need the L5 version per trainer config)
    for aux in spec.get("aux_feats", []):
        k = f"{aux}_l5"
        if k in out.columns:
            keep.append(k)

    # Defense allowed
    for suf in ["l3", "l5", "l8"]:
        k = f"{y_col}_allowed_{suf}"
        if k in out.columns:
            keep.append(k)

    out = out[keep].dropna(subset=["y"]).reset_index(drop=True)
    return out

# ---------------------- main ----------------------
def main():
    seasons = list(range(SEASON_START, SEASON_END + 1))
    print(f"[ETL] Loading weekly data for seasons {seasons} ...", flush=True)
    df = load_weekly(seasons)

    print(f"[ETL] Rows loaded: {len(df):,}", flush=True)

    for mk, spec in MARKETS.items():
        print(f"[ETL] Building table for {mk} ...", flush=True)
        t = build_table_for_market(df, mk, spec)
        outp = os.path.join(OUT_DIR, f"train_{mk}.parquet")
        t.to_parquet(outp, index=False)
        print(f"[ETL]   wrote {outp} [{len(t):,} rows]", flush=True)

    print("[ETL] Complete.", flush=True)

if __name__ == "__main__":
    main()
