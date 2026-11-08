"""One set of metrics, used by train.py and test.py."""
import numpy as np
from sklearn.metrics import r2_score

from src.models import DriftModel, NaiveModel
from src.split import folds


def metrics(df, pred, pred_naive, pred_drift, target="target_balance"):
    y, lag1 = df[target], df["balance_lag1"]
    err = (y - pred).abs()
    mae = float(err.mean())
    return {
        "MAE": mae,
        "skill_vs_naive": 1 - mae / float((y - pred_naive).abs().mean()),
        "skill_vs_drift": 1 - mae / float((y - pred_drift).abs().mean()),
        "R2_change": float(r2_score(y - lag1, pred - lag1)),
        "R2_level_FLATTERING": float(r2_score(y, pred)),
        "median_AE": float(err.median()),
        "bias": float((y - pred).mean()),
        "within_100": float((err <= 100).mean()),
        "within_500": float((err <= 500).mean()),
    }


def paired_bootstrap(diff, n=1000, seed=0):
    """diff = error(model B) - error(model A), row by row. Returns mean and 95% interval."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def fold_predictions(make_model, train, fold_months, target="target_balance"):
    """Walk-forward: for each fold, fit a NEW model on the months before, predict the fold month.
    Returns one row per validated account-month (with naive and drift next to the model)."""
    rows = []
    for m, tr, va in folds(train, fold_months):
        model = make_model().fit(tr)
        out = va[["account_id", "month", "balance_lag1", target]].copy()
        out["pred"] = model.predict(va)
        out["pred_naive"] = NaiveModel().fit(tr).predict(va)
        out["pred_drift"] = DriftModel().fit(tr).predict(va)
        out["train_mae"] = float((tr[target] - model.predict(tr)).abs().mean())
        rows.append(out)
    import pandas as pd
    return pd.concat(rows, ignore_index=True)


def fold_table(pred_rows, name, target="target_balance"):
    """Per-fold scores: training MAE next to validation MAE, naive and drift."""
    import pandas as pd
    out = []
    for m, g in pred_rows.groupby("month"):
        y = g[target]
        out.append({"model": name, "month": str(m)[:7],
                    "train_MAE": g["train_mae"].iloc[0],
                    "val_MAE": float((y - g["pred"]).abs().mean()),
                    "naive_MAE": float((y - g["pred_naive"]).abs().mean()),
                    "drift_MAE": float((y - g["pred_drift"]).abs().mean())})
    return pd.DataFrame(out)


def errors_by_segment(df, pred, pred_drift, target="target_balance", active_share=None):
    """Where the model loses. For understanding only, never for choosing features."""
    import pandas as pd
    ev = df[["account_id", "balance_lag1", target, "debit_avg_3m"]].copy()
    ev["real_move"] = ev[target] - ev["balance_lag1"]
    ev["err_model"] = (ev[target] - pred).abs()
    ev["err_drift"] = (ev[target] - pred_drift).abs()
    ev["err_naive"] = ev["real_move"].abs()
    ev["month_kind"] = np.where(ev["real_move"] == 0, "quiet",
                                np.where(ev["real_move"] > 0, "refund (up)", "spending"))
    ev["spender"] = pd.qcut(ev["debit_avg_3m"].rank(method="first"), 3, labels=["low", "medium", "high"])
    if active_share is not None:
        ev["user"] = np.where(ev["account_id"].map(active_share) >= 0.67, "active", "regular")
    parts = []
    for col in [c for c in ["month_kind", "spender", "user"] if c in ev]:
        t = ev.groupby(col, observed=True)[["err_model", "err_drift", "err_naive"]].mean()
        t.insert(0, "rows", ev.groupby(col, observed=True).size())
        t = t.reset_index().rename(columns={col: "segment"})
        t.insert(0, "split_by", col)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)
