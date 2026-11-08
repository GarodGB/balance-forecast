"""Build the monthly feature table from the clean transactions.

One function, used by train.py AND predict.py, so the model always sees features built the same way.
Rules: a transaction counts in the month of its transaction date; every feature for month M
uses only transactions BEFORE month M (the cutoff). Only the target uses month M itself.
"""
import pandas as pd

LAG_SOURCES = ["credit", "debit", "net"]


def load_transactions(path):
    tx = pd.read_csv(path, usecols=["account_id", "txn_date_time", "billing_amount"],
                     parse_dates=["txn_date_time"], dtype={"account_id": str})
    tx["month"] = tx["txn_date_time"].dt.to_period("M").dt.to_timestamp()
    return tx


def monthly_summary(tx):
    """One row per account per month that HAS transactions: credit, debit, net, count."""
    amt = tx["billing_amount"]
    return (tx.assign(credit=amt.clip(lower=0), debit=(-amt).clip(lower=0))
              .groupby(["account_id", "month"])
              .agg(credit=("credit", "sum"), debit=("debit", "sum"),
                   net=("billing_amount", "sum"), txn_count=("billing_amount", "size"))
              .reset_index())


def build_features(tx, last_month=None, add_next_month=False):
    """Feature table for every account and every month up to `last_month` (the cutoff).

    tx: transactions with a `month` column.
    last_month: the latest CLOSED month to use. Transactions after it are dropped first.
    add_next_month: also add the row for the month after `last_month` (what predict.py forecasts).
                    Its target is unknown (NaN); its features only use months up to `last_month`.
    """
    if last_month is None:
        last_month = tx["month"].max()
    last_month = pd.Timestamp(last_month)
    tx = tx[tx["month"] <= last_month]                                   # step 1: the cutoff

    months = pd.date_range(tx["month"].min(), last_month, freq="MS")
    if add_next_month:
        months = months.append(pd.DatetimeIndex([last_month + pd.DateOffset(months=1)]))
    accounts = sorted(tx["account_id"].unique())
    grid = pd.MultiIndex.from_product([accounts, months], names=["account_id", "month"]).to_frame(index=False)

    m = grid.merge(monthly_summary(tx), on=["account_id", "month"], how="left")
    m[["credit", "debit", "net", "txn_count"]] = m[["credit", "debit", "net", "txn_count"]].fillna(0)
    m = m.sort_values(["account_id", "month"]).reset_index(drop=True)

    g = m.groupby("account_id")
    m["balance"] = g["net"].cumsum()                                     # running total from 0

    shifted_balance = g["balance"].shift(1)                              # end of the PREVIOUS month
    m["balance_lag1"] = shifted_balance
    m["balance_lag2"] = g["balance"].shift(2)
    m["balance_lag3"] = g["balance"].shift(3)
    roll = shifted_balance.groupby(m["account_id"]).rolling(3, min_periods=1)
    m["balance_mean_3m"] = roll.mean().reset_index(level=0, drop=True)
    m["balance_std_3m"] = (shifted_balance.groupby(m["account_id"]).rolling(3, min_periods=2)
                           .std().reset_index(level=0, drop=True))
    for col in LAG_SOURCES:
        prev = g[col].shift(1)
        m[f"{col}_lag1"] = prev
        m[f"{col}_avg_3m"] = (prev.groupby(m["account_id"]).rolling(3, min_periods=1)
                              .mean().reset_index(level=0, drop=True))
    m["txn_count_lag1"] = g["txn_count"].shift(1)

    m["target_balance"] = m["balance"]
    if add_next_month:                                                    # the future is unknown
        m.loc[m["month"] > last_month, "target_balance"] = float("nan")
    return m.drop(columns=["credit", "debit", "net", "txn_count", "balance"])
