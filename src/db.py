"""SQLite: one file, two tables, one view. Built-in sqlite3, no server, no passwords.

Run `python -m src.db fill_actuals` when a forecast month has closed, to copy in the real balances.
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    model_version   TEXT PRIMARY KEY,
    trained_at      TEXT NOT NULL,
    model_type      TEXT NOT NULL,            -- hgb or drift (whichever was promoted)
    params_json     TEXT,
    train_months    TEXT,
    test_month      TEXT,
    train_mae       REAL,
    cv_mae          REAL,
    cv_naive_mae    REAL,
    cv_drift_mae    REAL,
    test_mae        REAL,                     -- NULL until test.py runs (never 0)
    test_naive_mae  REAL,
    test_drift_mae  REAL,
    skill           REAL,
    test_scored_at  TEXT,
    promoted        INTEGER NOT NULL CHECK (promoted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY,
    account_id        TEXT NOT NULL,
    target_month      TEXT NOT NULL,           -- YYYY-MM
    balance_lag1      REAL,                    -- also the naive forecast
    predicted_change  REAL,
    predicted_balance REAL,
    drift_balance     REAL,                    -- the drift baseline, to compare later
    model_used        TEXT NOT NULL,           -- hgb, drift, or none (no history)
    model_version     TEXT NOT NULL REFERENCES model_runs(model_version),
    split             TEXT NOT NULL CHECK (split IN ('cv', 'test', 'forecast')),
    actual_balance    REAL,                    -- NULL until known
    created_at        TEXT NOT NULL,
    features_json     TEXT,
    UNIQUE (account_id, target_month, model_version)
);

CREATE TABLE IF NOT EXISTS test_month_log (    -- when a test month was FIRST scored by any model
    test_month       TEXT PRIMARY KEY,
    first_scored_at  TEXT NOT NULL,
    note             TEXT
);

-- errors are computed in a view, never stored, so they never go stale
CREATE VIEW IF NOT EXISTS monthly_scores AS
SELECT target_month, model_version, split,
       COUNT(*)                                   AS rows,
       AVG(ABS(actual_balance - predicted_balance)) AS model_mae,
       AVG(ABS(actual_balance - balance_lag1))      AS naive_mae,
       AVG(ABS(actual_balance - drift_balance))     AS drift_mae
FROM predictions
WHERE split IN ('test', 'forecast') AND actual_balance IS NOT NULL AND predicted_balance IS NOT NULL
GROUP BY target_month, model_version, split;
"""

UPSERT = """
INSERT INTO predictions (account_id, target_month, balance_lag1, predicted_change, predicted_balance,
                         drift_balance, model_used, model_version, split, actual_balance, created_at,
                         features_json)
VALUES (:account_id, :target_month, :balance_lag1, :predicted_change, :predicted_balance,
        :drift_balance, :model_used, :model_version, :split, :actual_balance, :created_at, :features_json)
ON CONFLICT (account_id, target_month, model_version) DO UPDATE SET
    balance_lag1 = excluded.balance_lag1,
    predicted_change = excluded.predicted_change,
    predicted_balance = excluded.predicted_balance,
    drift_balance = excluded.drift_balance,
    model_used = excluded.model_used,
    split = excluded.split,
    created_at = excluded.created_at,
    features_json = excluded.features_json
    -- actual_balance is NOT in this list: a rerun never wipes known actuals
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(path):
    con = connect(path)
    con.executescript(SCHEMA)
    con.commit()
    return con


def none_if_nan(x):
    return None if x is None or x != x else float(x)


def upsert_predictions(con, rows):
    """rows: list of dicts with the predictions columns. One transaction: all or nothing."""
    with con:
        con.executemany(UPSERT, [{**r, "created_at": r.get("created_at", now())} for r in rows])


def insert_model_run(con, run):
    cols = ", ".join(run)
    marks = ", ".join(f":{c}" for c in run)
    with con:
        con.execute(f"INSERT OR REPLACE INTO model_runs ({cols}) VALUES ({marks})", run)


def update_model_run(con, version, **fields):
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with con:
        con.execute(f"UPDATE model_runs SET {sets} WHERE model_version = :v", {**fields, "v": version})


def log_test_month(con, month, note=""):
    """Returns the date the month was first scored if it was already seen, else None."""
    row = con.execute("SELECT first_scored_at FROM test_month_log WHERE test_month = ?", (month,)).fetchone()
    if row:
        return row[0]
    with con:
        con.execute("INSERT INTO test_month_log VALUES (?, ?, ?)", (month, now(), note))
    return None


def fill_actuals(con, transactions_path):
    """Copy the real end-of-month balance into rows whose month has now closed."""
    from src.features import build_features, load_transactions
    tx = load_transactions(transactions_path)
    feat = build_features(tx)
    actual = {(r.account_id, r.month.strftime("%Y-%m")): r.target_balance
              for r in feat[["account_id", "month", "target_balance"]].itertuples()}
    rows = con.execute("SELECT id, account_id, target_month FROM predictions WHERE actual_balance IS NULL").fetchall()
    updates = [(actual[(a, m)], i) for i, a, m in rows if (a, m) in actual]
    with con:
        con.executemany("UPDATE predictions SET actual_balance = ? WHERE id = ?", updates)
    return len(updates)


if __name__ == "__main__":
    from src.config import load_config
    if len(sys.argv) > 1 and sys.argv[1] == "fill_actuals":
        cfg = load_config()
        con = init_db(cfg["paths"]["db"])
        n = fill_actuals(con, cfg["paths"]["transactions"])
        print(f"filled actual_balance in {n} rows")
    else:
        print("usage: python -m src.db fill_actuals")
