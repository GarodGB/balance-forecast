import json
from pathlib import Path

import numpy as np
import yaml

from inference import predict
from train import train


def test_train_then_predict_on_tiny_fake_data(project):
    meta = train.main(str(project), n_trials=2)
    cfg = yaml.safe_load(Path(project).read_text())
    models = Path(cfg["paths"]["models"])
    for f in ["model.pkl", "optuna_study.pkl", "feature_columns.json", "metadata.json"]:
        assert (models / f).exists(), f
    assert json.loads((models / "metadata.json").read_text())["model_version"] == meta["model_version"]

    n = predict.main(str(project))
    assert n == 30                                               # one row per account
    import sqlite3
    con = sqlite3.connect(cfg["paths"]["db"])
    vals = [r[0] for r in con.execute("SELECT predicted_balance FROM predictions WHERE split='forecast'")]
    assert len(vals) == 30 and np.isfinite(vals).all()
    assert predict.main(str(project)) == 0                       # a second run refuses, no duplicates
