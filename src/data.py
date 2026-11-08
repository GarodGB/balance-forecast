"""Data checks: prove the feature table is consistent before any model sees it."""
import numpy as np
import pandas as pd

EXPECTED_MISSING = {  # column -> how many first months may be empty (no history yet)
    "balance_lag1": 1, "balance_lag2": 2, "balance_lag3": 3, "balance_mean_3m": 1,
    "balance_std_3m": 2, "credit_lag1": 1, "debit_lag1": 1, "net_lag1": 1,
    "credit_avg_3m": 1, "debit_avg_3m": 1, "net_avg_3m": 1, "txn_count_lag1": 1,
}


class DataCheckError(Exception):
    pass


def run_checks(feat, tx, stage4=None):
    """Hard checks stop training. Warnings are only reported."""
    report, hard_fail, warnings = {}, [], []

    dup = int(feat.duplicated(["account_id", "month"]).sum())
    report["duplicate_account_months"] = dup
    if dup:
        hard_fail.append(f"{dup} duplicated account-months")

    months = sorted(feat["month"].unique())
    per_acc = feat.groupby("account_id")["month"].nunique()
    gaps = per_acc[per_acc != len(months)].index.tolist()
    report["accounts_missing_months"] = len(gaps)
    if gaps:
        hard_fail.append(f"accounts with missing months: {gaps[:10]}")

    prev_target = feat.groupby("account_id")["target_balance"].shift(1)
    bad_lag = int(((feat["balance_lag1"] - prev_target).abs() > 0.01).sum())
    report["lag1_not_equal_previous_target"] = bad_lag
    if bad_lag:
        hard_fail.append(f"{bad_lag} rows where balance_lag1 != last month's target")

    # the move of each month must equal that month's transactions (credits - debits)
    net = tx.groupby(["account_id", "month"])["billing_amount"].sum()
    move = feat["target_balance"] - feat.groupby("account_id")["target_balance"].shift(1).fillna(0)
    real = feat.set_index(["account_id", "month"]).index.map(net).to_series().fillna(0).to_numpy()
    known = feat["target_balance"].notna().to_numpy()
    bad_move = int((np.abs(move.to_numpy() - real) > 0.01)[known].sum())
    report["move_not_equal_transactions"] = bad_move
    if bad_move:
        hard_fail.append(f"{bad_move} months where the balance move != the month's transactions")

    first = feat["month"].min()
    rank = feat["month"].map({m: i for i, m in enumerate(months)})
    unexpected = {}
    for col, n_first in EXPECTED_MISSING.items():
        if col in feat:
            n = int(feat.loc[rank >= n_first, col].isna().sum())
            if n:
                unexpected[col] = n
    report["unexpected_missing"] = unexpected
    if unexpected:
        warnings.append(f"missing values outside the first months: {unexpected}")

    if stage4 is not None:  # features.py must rebuild exactly what Stage 4 (Spark) produced
        both = feat.merge(stage4, on=["account_id", "month"], suffixes=("", "_s4"))
        diffs = {}
        for col in EXPECTED_MISSING:
            if f"{col}_s4" in both:
                d = (both[col] - both[f"{col}_s4"]).abs()
                diffs[col] = float(d.max(skipna=True)) if d.notna().any() else 0.0
        report["max_diff_vs_stage4"] = diffs
        off = {c: v for c, v in diffs.items() if v > 0.01}
        if off:
            hard_fail.append(f"features.py differs from Stage 4: {off}")

    report["rows"] = len(feat)
    report["months"] = [str(m)[:7] for m in months]
    report["first_month"] = str(first)[:7]
    report["hard_fail"] = hard_fail
    report["warnings"] = warnings
    return report


def load_stage4(path):
    return pd.read_csv(path, parse_dates=["month"], dtype={"account_id": str})
