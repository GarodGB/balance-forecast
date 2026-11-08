"""Models. Every model predicts the CHANGE this month; the balance = balance_lag1 + change.
They are small classes (not lambdas) so joblib can save them."""
from sklearn.ensemble import HistGradientBoostingRegressor


def move(df, target="target_balance"):
    return df[target] - df["balance_lag1"]


class NaiveModel:
    """The balance stays the same as last month (change = 0)."""
    name = "naive"

    def fit(self, df):
        return self

    def predict_change(self, df):
        return df["balance_lag1"] * 0.0

    def predict(self, df):
        return df["balance_lag1"] + self.predict_change(df)


class DriftModel(NaiveModel):
    """Last balance + the median monthly move of all accounts in the training months."""
    name = "drift"

    def fit(self, df):
        self.median_move_ = float(move(df).median())
        return self

    def predict_change(self, df):
        return df["balance_lag1"] * 0.0 + self.median_move_


class HGBChangeModel(NaiveModel):
    """Gradient boosting on the change, absolute-error loss (the metric is MAE). NaN stays NaN."""
    name = "hgb"

    def __init__(self, features, params, loss="absolute_error", seed=0):
        self.features, self.params, self.loss, self.seed = list(features), dict(params), loss, seed

    def fit(self, df):
        # a column that is empty in every training row (e.g. balance_lag3 when training on Apr-May only)
        # carries nothing and breaks some scikit-learn versions, so it is left out for this fit
        self.used_ = [c for c in self.features if df[c].notna().any()]
        self.model_ = HistGradientBoostingRegressor(loss=self.loss, random_state=self.seed, **self.params)
        self.model_.fit(df[self.used_], move(df))
        return self

    def predict_change(self, df):
        return self.model_.predict(df[self.used_])


class ThreeMonthAverageModel(NaiveModel):
    """The baseline the brief names: the average balance of the 3 months before."""
    name = "3-month average"

    def predict_change(self, df):
        return df["balance_mean_3m"] - df["balance_lag1"]


class MedianLinearModel(NaiveModel):
    """A fair linear reference: median (quantile) regression on the change, no month_of_year.
    Filled columns get a 0/1 flag; scaler + model live in one Pipeline, fitted on training rows only."""
    name = "median linear"

    def __init__(self, features):
        self.features = [c for c in features if c not in ("month_of_year", "debit_lag1", "debit_avg_3m")]

    def _x(self, df):
        x = df[self.features].copy()
        for c in self.na_cols_:
            x[c + "_missing"] = x[c].isna().astype(int)
        return x.fillna(0)

    def fit(self, df):
        from sklearn.linear_model import QuantileRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.na_cols_ = [c for c in self.features if df[c].isna().any()]
        self.pipe_ = make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, alpha=0, solver="highs"))
        self.pipe_.fit(self._x(df), move(df))
        return self

    def predict_change(self, df):
        return self.pipe_.predict(self._x(df))
