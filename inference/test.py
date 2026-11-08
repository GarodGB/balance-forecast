"""Score the saved model on the test month ONCE, write an honest report. Never tunes the saved model.

Run from the repo root:  python -m inference.test
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import db
from src.artifacts import load_artifacts
from src.config import load_config
from src.evaluate import errors_by_segment, fold_predictions, metrics, paired_bootstrap
from src.features import build_features, load_transactions
from src.models import DriftModel, HGBChangeModel, NaiveModel, ThreeMonthAverageModel
from src.split import make_split
from src.tuning import tune

log = logging.getLogger("test")


def nested_walk_forward(feat, cfg, features, n_trials=20):
    """For each month: choose the settings with ONLY the months before it, then score that month.
    This is the number to quote, because the test month has already been looked at."""
    mcfg, months = cfg["model"], sorted(feat["month"].unique())[cfg["data"]["drop_first_months"]:]
    min_train = cfg["split"]["min_train_months"]
    eval_months = months[min_train:]
    rows, diffs = [], []
    for m in eval_months:
        hist, now = feat[feat["month"] < m], feat[feat["month"] == m]
        hist = hist[hist["month"].isin(months)]
        inner = [v for v in eval_months if v < m]
        params, tuned = dict(mcfg["default_params"]), False
        if inner:                                   # enough history to tune on
            study = tune(hist, inner, features, cfg, n_trials=n_trials)
            default = fold_predictions(lambda: HGBChangeModel(features, params, mcfg["loss"], mcfg["seed"]),
                                       hist, inner)
            default_mae = float((default["target_balance"] - default["pred"]).abs().mean())
            if study.best_value <= default_mae - cfg["tuning"]["min_gain_over_default"]:
                params, tuned = study.best_params, True
        model = HGBChangeModel(features, params, mcfg["loss"], mcfg["seed"]).fit(hist)
        pred = model.predict(now)
        pred_naive = NaiveModel().fit(hist).predict(now)
        pred_drift = DriftModel().fit(hist).predict(now)
        r = metrics(now, pred, pred_naive, pred_drift)
        r.update({"month": str(m)[:7], "tuned": tuned})
        rows.append(r)
        diffs.extend(((now["target_balance"] - pred).abs() - (now["target_balance"] - pred_drift).abs()).tolist())
    e = cfg["evaluation"]
    return pd.DataFrame(rows), paired_bootstrap(diffs, e["bootstrap_samples"], e["seed"])


def main(config_path="config.yaml", nested_trials=20):
    cfg = load_config(config_path)
    p, e = cfg["paths"], cfg["evaluation"]
    model, features, meta = load_artifacts(p["models"])        # 1-3. metadata, version check, model
    test_month = pd.Timestamp(meta["months"]["test"][0] + "-01")   # taken from training, not recomputed

    # 4. features with the shared function; cutoff = the test month (its target is now known)
    tx = load_transactions(p["transactions"])
    feat = build_features(tx, last_month=test_month)
    train_months, _, _ = make_split(feat, cfg)
    train = feat[feat["month"].isin(train_months)]
    test = feat[feat["month"] == test_month]
    missing = [c for c in features if c not in test]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")

    # 5-7. predict, score next to naive, drift and the 3-month average, bootstrap intervals
    pred = model.predict(test)
    pred_naive = NaiveModel().fit(train).predict(test)
    pred_drift = DriftModel().fit(train).predict(test)
    pred_3m = ThreeMonthAverageModel().fit(train).predict(test)
    y = test["target_balance"]
    m = metrics(test, pred, pred_naive, pred_drift)
    m["three_month_avg_MAE"] = float((y - pred_3m).abs().mean())
    err = (y - pred).abs()
    m["vs_naive"] = dict(zip(["mae_change", "ci_low", "ci_high"],
                             paired_bootstrap(err - (y - pred_naive).abs(), e["bootstrap_samples"], e["seed"])))
    m["vs_drift"] = dict(zip(["mae_change", "ci_low", "ci_high"],
                             paired_bootstrap(err - (y - pred_drift).abs(), e["bootstrap_samples"], e["seed"])))

    # 8. errors by segment + the 10 worst accounts with their biggest transaction that month
    share = (feat[feat["month"] <= test_month].groupby("account_id")["txn_count_lag1"]
             .apply(lambda s: (s > 0).mean()))
    seg = errors_by_segment(test, pred, pred_drift, active_share=share)
    month_tx = tx[tx["month"] == test_month]
    biggest = (month_tx.assign(size=month_tx["billing_amount"].abs()).sort_values("size")
               .groupby("account_id").tail(1).set_index("account_id")["billing_amount"])
    worst = test[["account_id", "balance_lag1", "target_balance"]].assign(pred=pred, error=err)
    worst = worst.nlargest(10, "error").assign(biggest_transaction=lambda d: d["account_id"].map(biggest))

    # 9. reports
    Path(p["reports"]).mkdir(parents=True, exist_ok=True)
    Path(p["reports"], "test_metrics.json").write_text(json.dumps({"model_version": meta["model_version"],
                                                                   "model_type": meta["model_type"],
                                                                   "test_month": str(test_month)[:7], **m}, indent=2))
    seg.to_csv(Path(p["reports"], "errors_by_segment.csv"), index=False)
    worst.to_csv(Path(p["reports"], "worst_accounts.csv"), index=False)

    # 10-11. database: test rows with the actual filled in, model_runs scores, test-month log
    con = db.init_db(p["db"])
    seen = db.log_test_month(con, str(test_month)[:7], note="first scored by inference.test")
    if seen:
        log.warning("test month %s was already scored on %s: treat it as SEEN and quote the nested "
                    "walk-forward instead", str(test_month)[:7], seen)
    db.upsert_predictions(con, [{
        "account_id": r.account_id, "target_month": str(test_month)[:7], "balance_lag1": r.balance_lag1,
        "predicted_change": pc, "predicted_balance": pb, "drift_balance": pd_, "model_used": meta["model_type"],
        "model_version": meta["model_version"], "split": "test", "actual_balance": r.target_balance,
        "features_json": None}
        for r, pc, pb, pd_ in zip(test.itertuples(), pred - test["balance_lag1"], pred, pred_drift)])
    db.update_model_run(con, meta["model_version"], test_mae=m["MAE"],
                        test_naive_mae=float((y - pred_naive).abs().mean()),
                        test_drift_mae=float((y - pred_drift).abs().mean()), skill=m["skill_vs_naive"],
                        test_scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # 12. nested walk-forward: the honest number to quote when the test month is already seen
    nested, (d, lo, hi) = nested_walk_forward(feat, cfg, features, n_trials=nested_trials)
    nested.to_csv(Path(p["reports"], "nested_walk_forward.csv"), index=False)

    print(f"\n=== test month {str(test_month)[:7]} | model type: {meta['model_type']} ===")
    print(f"MAE {m['MAE']:.1f} | skill vs naive {m['skill_vs_naive']:+.3f} | vs drift {m['skill_vs_drift']:+.3f}")
    print(f"vs drift: {m['vs_drift']['mae_change']:+.2f} (95%: {m['vs_drift']['ci_low']:.2f} "
          f"to {m['vs_drift']['ci_high']:.2f})")
    print("\n=== nested walk-forward (quote this) ===")
    print(nested[["month", "MAE", "skill_vs_naive", "skill_vs_drift", "tuned"]].round(3).to_string(index=False))
    print(f"boosting vs drift, pooled: {d:+.2f} (95%: {lo:.2f} to {hi:.2f})")
    return m, nested


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--nested-trials", type=int, default=20)
    a = ap.parse_args()
    main(a.config, a.nested_trials)
