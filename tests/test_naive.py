import pandas as pd
import pytest

from src.artifacts import ModelMissing, load_artifacts
from src.models import DriftModel, NaiveModel


def test_naive_forecast_equals_balance_lag1():
    df = pd.DataFrame({"balance_lag1": [-500.0, 0.0, 120.5]})
    assert NaiveModel().predict(df).tolist() == [-500.0, 0.0, 120.5]


def test_predicted_balance_is_lag1_plus_change_for_negative_and_zero_balances():
    df = pd.DataFrame({"balance_lag1": [-500.0, 0.0], "target_balance": [-600.0, -100.0]})
    model = DriftModel().fit(df)                                  # median move = -100
    assert (model.predict(df) - (df["balance_lag1"] + model.predict_change(df))).abs().max() < 1e-12
    assert model.predict(df).tolist() == [-600.0, -100.0]


def test_predict_stops_with_a_clear_message_when_the_model_is_missing(tmp_path):
    with pytest.raises(ModelMissing, match="run `python -m train.train` first"):
        load_artifacts(tmp_path)
