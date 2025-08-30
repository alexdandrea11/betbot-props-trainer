# train_models_regression.py
# Trains one regression per market: predicts mean stat μ; saves sigma=RMSE.

import os, json, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump as joblib_dump

IN_DIR = "data"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

MARKETS = {
    "player_receptions": {
        "file": "train_player_receptions.parquet",
        "y_col": "y",
        "features": [
            "is_home","days_rest",
            "receptions_l3","receptions_l5","receptions_l8",
            "targets_l5",
            "receptions_allowed_l3","receptions_allowed_l5","receptions_allowed_l8",
            "h2h_mean_alltime",
        ],
        "sigma_floor": 0.9,
    },
    "player_reception_yds": {
        "file": "train_player_reception_yds.parquet",
        "y_col": "y",
        "features": [
            "is_home","days_rest",
            "receiving_yards_l3","receiving_yards_l5","receiving_yards_l8",
            "receptions_l5","targets_l5",
            "receiving_yards_allowed_l3","receiving_yards_allowed_l5","receiving_yards_allowed_l8",
            "h2h_mean_alltime",
        ],
        "sigma_floor": 12.0,
    },
    "player_rush_yds": {
        "file": "train_player_rush_yds.parquet",
        "y_col": "y",
        "features": [
            "is_home","days_rest",
            "rushing_yards_l3","rushing_yards_l5","rushing_yards_l8",
            "carries_l5",
            "rushing_yards_allowed_l3","rushing_yards_allowed_l5","rushing_yards_allowed_l8",
            "h2h_mean_alltime",
        ],
        "sigma_floor": 9.0,
    },
    "player_pass_yds": {
        "file": "train_player_pass_yds.parquet",
        "y_col": "y",
        "features": [
            "is_home","days_rest",
            "passing_yards_l3","passing_yards_l5","passing_yards_l8",
            "passing_yards_allowed_l3","passing_yards_allowed_l5","passing_yards_allowed_l8",
            "h2h_mean_alltime",
        ],
        "sigma_floor": 35.0,
    },
}

def clean_feats(df, feats):
    X = df.copy()
    for c in feats:
        if c not in X.columns:
            X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X[feats].astype(float)

def train_one(market_key, spec):
    path = os.path.join(IN_DIR, spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing training parquet: {path} (run etl_build_history.py first)")

    df = pd.read_parquet(path)
    feats = spec["features"]
    ycol = spec["y_col"]
    X = clean_feats(df, feats)
    y = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0).astype(float)

    # split by season if possible (keep generalization to recent)
    if "season" in df.columns:
        recent_mask = df["season"] >= (df["season"].max() - 1)   # last season as test if available
        X_train = X[~recent_mask]; y_train = y[~recent_mask]
        X_test  = X[ recent_mask]; y_test  = y[ recent_mask]
        if len(X_test) < 200:  # fallback to random split if too small
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # simple ridge
    model = Ridge(alpha=2.0, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    sigma = float(max(rmse, spec.get("sigma_floor", 1.0)))  # floor for stability

    # save artifacts
    base = f"props_reg_{market_key}"
    mpath = os.path.join(OUT_DIR, f"{base}.joblib")
    jpath = os.path.join(OUT_DIR, f"{base}.meta.json")
    joblib_dump(model, mpath)
    meta = {
        "features": feats,
        "sigma": sigma,
        "built_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Ridge(alpha=2.0)",
        "target": spec["y_col"],
    }
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{market_key}] saved {mpath} and {jpath} (sigma={sigma:.2f})")

def main():
    for mk, spec in MARKETS.items():
        train_one(mk, spec)
    print("Training complete.")

if __name__ == "__main__":
    main()
