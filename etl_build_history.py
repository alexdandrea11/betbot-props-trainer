# etl_build_history.py
# Builds per-market training tables from nfl_data_py weekly data.

import os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

SEASON_START = int(os.getenv("SEASON_START", "2015"))
SEASON_END   = int(os.getenv("SEASON_END", str(datetime.utcnow().year)))  # up to current year
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

MARKETS = {
    "player_receptions": {
        "y_col": "receptions",
        "aux_feats": ["targets"],   # we’ll also make targets_l5
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

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def load_weekly(seasons):
    import nfl_data_py as nfl
    df = nfl.import_weekly_data(seasons)
    # Standardize columns we need
    rename = {
        "player_name":"player",
        "player_display_name":"player",
        "recent_team":"team",
        "team":"team",
        "opponent_team":"opp",
        "opponent":"opp",
        "week":"week",
        "season":"season",
        "position":"position"
    }
    for k,v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    # Coerce ints/floats
    for col in ["week","season"]:
        if col in df.columns:
            df[col] = safe_num(df[col]).astype("Int64").astype(float).astype(int)
    # Basic stat columns we might use
    for c in [
        "receptions","receiving_yards","targets",
        "rushing_yards","carries",
        "passing_yards"
    ]:
        if c in df.columns:
            df[c] = safe_num(df[c]).fillna(0.0)
        else:
            df[c] = 0.0

    # schedule date for rest calc
    try:
        sch = nfl.import_schedules(seasons)
        sch = sch.rename(columns={"home_team":"home","away_team":"away"})
        dtcol = "gameday" if "gameday" in sch.columns else ("game_date" if "game_date" in sch.columns else None)
        if dtcol:
            sch["game_date"] = pd.to_datetime(sch[dtcol], errors="coerce")
        else:
            sch["game_date"] = pd.NaT
        sch_small = sch[["season","week","home","away","game_date"]].copy()
        # merge to get date; figure out if player is home
        tmp = df.merge(sch_small, on=["season","week"], how="left")
        tmp["is_home"] = np.where(tmp["team"]==tmp["home"], 1, np.where(tmp["team"]==tmp["away"], 0, np.nan))
        tmp["game_date"] = pd.to_datetime(tmp["game_date"], errors="coerce")
        # days rest per player
        tmp = tmp.sort_values(["player","season","week"])
        tmp["prev_date"] = tmp.groupby("player")["game_date"].shift(1)
        tmp["days_rest"] = (tmp["game_date"] - tmp["prev_date"]).dt.days.fillna(7).clip(3, 21)
        # opponent column as team abbr
        if "opp" not in tmp.columns:
            tmp["opp"] = np.where(tmp["is_home"]==1, tmp["away"], tmp["home"])
        return tmp
    except Exception as e:
        # fallback without dates
        df["is_home"] = np.nan
        df["days_rest"] = 7
        if "opp" not in df.columns:
            df["opp"] = ""
        return df

def make_rolling_feats(df, col, groups=["player"], windows=(3,5,8), suffixes=("l3","l5","l8")):
    out = df.copy()
    out = out.sort_values(groups+["season","week"])
    for w, suf in zip(windows, suffixes):
        out[f"{col}_{suf}"] = (
            out.groupby(groups)[col]
               .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
               .reset_index(level=groups, drop=True)
        )
    return out

def build_defense_allowed(df, y_col):
    # team-week allowed: sum of opponent players' stat vs a defense
    tmp = df.copy()
    # For each row, defense = the opponent team
    tmp["def_team"] = tmp["opp"]
    # aggregate stat by defense per (season, week)
    allowed = tmp.groupby(["season","week","def_team"], as_index=False)[y_col].sum()
    allowed = allowed.sort_values(["def_team","season","week"])
    for w, suf in [(3,"l3"), (5,"l5"), (8,"l8")]:
        allowed[f"{y_col}_allowed_{suf}"] = (
            allowed.groupby("def_team")[y_col]
                   .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                   .reset_index(level="def_team", drop=True)
        )
    keep = ["season","week","def_team"] + [f"{y_col}_allowed_{suf}" for suf in ["l3","l5","l8"]]
    return allowed[keep]

def build_h2h_mean(df, y_col):
    # prior average for this player vs this opponent, rolling expanding mean
    tmp = df.sort_values(["player","opp","season","week"]).copy()
    grp = tmp.groupby(["player","opp"])
    tmp["h2h_mean_alltime"] = grp[y_col].apply(lambda s: s.shift(1).expanding(1).mean()).reset_index(level=[0,1], drop=True)
    return tmp[["player","opp","season","week","h2h_mean_alltime"]]

def build_table_for_market(df, market_key, spec):
    y_col = spec["y_col"]
    out = df.copy()

    # base rolling on outcome and auxiliaries
    out = make_rolling_feats(out, y_col)
    for aux in spec.get("aux_feats", []):
        if aux in out.columns:
            out = make_rolling_feats(out, aux)

    # defense allowed rolling
    allowed = build_defense_allowed(out, y_col)
    out = out.merge(allowed, left_on=["season","week","opp"], right_on=["season","week","def_team"], how="left").drop(columns=["def_team"])

    # h2h rolling mean
    h2h = build_h2h_mean(out, y_col)
    out = out.merge(h2h, on=["player","opp","season","week"], how="left")

    # clean NA
    for c in out.columns:
        if c.endswith(("_l3","_l5","_l8","_alltime")):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(out[c].median() if np.isfinite(pd.to_numeric(out[c], errors="coerce")).any() else 0.0)

    # final selection + rename target as y
    out["y"] = out[y_col].astype(float)
    keep = [
        "player","team","opp","season","week","position","is_home","days_rest","y",
        f"{y_col}_l3", f"{y_col}_l5", f"{y_col}_l8",
        "h2h_mean_alltime",
    ]
    # aux rolls
    for aux in spec.get("aux_feats", []):
        keep += [f"{aux}_l5"] if f"{aux}_l5" in out.columns else []

    # defense allowed
    for suf in ["l3","l5","l8"]:
        k = f"{y_col}_allowed_{suf}"
        if k in out.columns: keep.append(k)

    out = out[keep].dropna(subset=["y"]).reset_index(drop=True)
    return out

def main():
    seasons = list(range(SEASON_START, SEASON_END+1))
    print(f"Loading weekly data for seasons {seasons} ...", flush=True)
    df = load_weekly(seasons)

    tables = {}
    for mk, spec in MARKETS.items():
        print(f"Building table for {mk} ...", flush=True)
        t = build_table_for_market(df, mk, spec)
        outp = os.path.join(OUT_DIR, f"train_{mk}.parquet")
        t.to_parquet(outp, index=False)
        tables[mk] = outp
        print(f"  wrote {outp} [{len(t):,} rows]")

    print("ETL complete.")
    return tables

if __name__ == "__main__":
    main()
