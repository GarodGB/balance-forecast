import pandas as pd

from src.config import load_config
from src.features import build_features
from src.split import make_split
from tests.conftest import fake_transactions


def _tx():
    tx = fake_transactions()
    tx["month"] = tx["txn_date_time"].dt.to_period("M").dt.to_timestamp()
    return tx


def test_lag1_equals_last_months_target():
    f = build_features(_tx())
    prev = f.groupby("account_id")["target_balance"].shift(1)
    assert ((f["balance_lag1"] - prev).abs().dropna() < 1e-9).all()


def test_no_feature_equals_the_target_months_balance():
    f = build_features(_tx())
    later = f[f["month"] > f["month"].min()]
    for col in [c for c in f.columns if c.endswith(("_lag1", "_3m"))]:
        assert not (later[col] - later["target_balance"]).abs().lt(1e-9).all(), col


def test_training_months_come_before_the_test_month(project):
    cfg = load_config(project)
    train_m, test_m, folds = make_split(build_features(_tx()), cfg)
    assert max(train_m) < min(test_m)
    assert all(m in train_m for m in folds)


def test_features_do_not_change_when_later_transactions_are_added():
    tx = _tx()
    cutoff = pd.Timestamp("2022-07-01")
    before = build_features(tx[tx["month"] <= cutoff], last_month=cutoff)
    full = build_features(tx, last_month=cutoff)                  # later rows exist but must be cut off
    cols = [c for c in before.columns if c not in ("account_id", "month")]
    pd.testing.assert_frame_equal(before[cols].reset_index(drop=True), full[cols].reset_index(drop=True))
