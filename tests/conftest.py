"""A tiny fake transactions file, so the tests never need the real (private) data."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def fake_transactions(n_accounts=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_accounts):
        acc = str(500000000 + a)
        for day in pd.date_range("2022-03-01", "2022-09-30", freq="D"):
            if rng.random() < 0.05:
                amt = -round(float(rng.gamma(2, 80)), 2) if rng.random() < 0.9 else round(float(rng.gamma(2, 60)), 2)
                rows.append((acc, day + pd.Timedelta(hours=int(rng.integers(0, 24))), amt))
        rows.append((acc, pd.Timestamp("2022-03-02 10:00"), -10.0))   # every account active in March
    return pd.DataFrame(rows, columns=["account_id", "txn_date_time", "billing_amount"])


@pytest.fixture
def project(tmp_path):
    """A temporary copy of the project config pointing at fake data and empty output folders."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    tx_path = tmp_path / "tx.csv"
    fake_transactions().to_csv(tx_path, index=False)
    cfg["paths"] = {"transactions": str(tx_path), "stage4_features": "",
                    "models": str(tmp_path / "models"), "reports": str(tmp_path / "reports"),
                    "db": str(tmp_path / "db" / "forecasts.db")}
    cfg["tuning"]["n_trials"] = 2
    cfg["evaluation"]["bootstrap_samples"] = 100
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
