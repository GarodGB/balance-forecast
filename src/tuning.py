"""Optuna on the walk-forward folds only. Never shuffled K-fold, never the test month."""
import optuna

from src.evaluate import fold_predictions
from src.models import HGBChangeModel


def tune(train, fold_months, features, cfg, n_trials=None):
    t, mcfg = cfg["tuning"], cfg["model"]
    r = t["ranges"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *r["learning_rate"], log=True),
            "max_iter": trial.suggest_int("max_iter", *r["max_iter"]),
            "max_depth": trial.suggest_int("max_depth", *r["max_depth"]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *r["min_samples_leaf"]),
            "l2_regularization": trial.suggest_float("l2_regularization", *r["l2_regularization"], log=True),
        }
        rows = fold_predictions(lambda: HGBChangeModel(features, params, mcfg["loss"], mcfg["seed"]),
                                train, fold_months)
        return float((rows["target_balance"] - rows["pred"]).abs().mean())   # pooled fold MAE

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=t["seed"]))
    study.optimize(objective, n_trials=n_trials or t["n_trials"], timeout=t["timeout_seconds"])
    return study
