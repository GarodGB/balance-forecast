"""Read config.yaml and stop loudly if a key is missing."""
import hashlib
from pathlib import Path

import yaml

REQUIRED = {
    "paths": ["transactions", "models", "reports", "db"],
    "data": ["id_col", "month_col", "target_col", "drop_first_months"],
    "split": ["test_share", "n_folds", "min_train_months"],
    "features": ["use"],
    "model": ["loss", "seed", "default_params"],
    "tuning": ["n_trials", "timeout_seconds", "seed", "ranges", "min_gain_over_default"],
    "evaluation": ["bootstrap_samples", "seed"],
    "promotion": ["min_gain_vs_drift", "require_ci_below_zero", "min_gain_vs_saved"],
}


def load_config(path="config.yaml"):
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"config.yaml: missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"config.yaml: missing key '{section}.{key}'")
    cfg["_hash"] = hashlib.sha256(text.encode()).hexdigest()
    return cfg
