"""Forecast the coming month for every account and write one row per account to SQLite.

Run from the repo root:  python -m inference.predict           (refuses to write the same month twice)
                         python -m inference.predict --force   (rerun: upsert, never duplicates)
"""
import argparse
import json
import logging

import pandas as pd

from src import db
from src.artifacts import load_artifacts
from src.config import load_config
from src.features import build_features, load_transactions

log = logging.getLogger("predict")


def main(config_path="config.yaml", force=False):
    cfg = load_config(config_path)
    p = cfg["paths"]
    model, features, meta = load_artifacts(p["models"])        # 1. version check + files

    tx = load_transactions(p["transactions"])
    last_closed = tx["month"].max()                              # 2. latest closed month in the data
    target_month = last_closed + pd.DateOffset(months=1)
    tm = target_month.strftime("%Y-%m")

    con = db.init_db(p["db"])                                    # 3. already forecast?
    done = con.execute("SELECT COUNT(*) FROM predictions WHERE target_month = ? AND model_version = ?",
                       (tm, meta["model_version"])).fetchone()[0]
    if done and not force:
        print(f"{done} rows for {tm} / {meta['model_version']} already exist. Use --force to rerun.")
        return 0

    # 4. features with the shared function; cutoff = end of the latest closed month
    feat = build_features(tx, last_month=last_closed, add_next_month=True)
    rows = feat[feat["month"] == target_month]
    missing = [c for c in features if c not in rows]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")

    # 5. accounts with no balance_lag1 have no history to forecast from
    has_history = rows["balance_lag1"].notna()
    scored, skipped = rows[has_history], rows[~has_history]

    # 6. predicted balance = balance_lag1 + predicted change
    change = pd.Series(model.predict_change(scored), index=scored.index)
    drift_move = meta["params"].get("median_move") if meta["model_type"] == "drift" else None
    if drift_move is None:                                       # keep a drift column next to every forecast
        from src.models import DriftModel
        hist = feat[feat["month"] <= last_closed]
        drift_move = DriftModel().fit(hist[hist["balance_lag1"].notna()]).median_move_

    out = [{"account_id": r.account_id, "target_month": tm, "balance_lag1": r.balance_lag1,
            "predicted_change": float(change[i]), "predicted_balance": float(r.balance_lag1 + change[i]),
            "drift_balance": float(r.balance_lag1 + drift_move), "model_used": meta["model_type"],
            "model_version": meta["model_version"], "split": "forecast", "actual_balance": None,
            "features_json": json.dumps({c: db.none_if_nan(getattr(r, c)) for c in features})}
           for i, r in zip(scored.index, scored.itertuples())]
    out += [{"account_id": r.account_id, "target_month": tm, "balance_lag1": None, "predicted_change": None,
             "predicted_balance": None, "drift_balance": None, "model_used": "none",
             "model_version": meta["model_version"], "split": "forecast", "actual_balance": None,
             "features_json": None} for r in skipped.itertuples()]

    db.upsert_predictions(con, out)                              # 7. one transaction, upsert
    print(f"forecast {tm}: {len(scored)} accounts scored, {len(skipped)} skipped (no history), "
          f"{len(out)} rows written | model {meta['model_type']} {meta['model_version']}")   # 8.
    return len(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.config, a.force)
