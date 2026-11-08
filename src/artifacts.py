"""Load the files train.py saved. Only load pickles your own pipeline created."""
import json
from pathlib import Path

import joblib
import sklearn


class ModelMissing(Exception):
    pass


def load_artifacts(models_dir):
    d = Path(models_dir)
    for f in ["model.pkl", "feature_columns.json", "metadata.json"]:
        if not (d / f).exists():
            raise ModelMissing(f"{d / f} not found: run `python -m train.train` first")
    meta = json.loads((d / "metadata.json").read_text())
    trained_with = meta["versions"]["scikit-learn"]
    if sklearn.__version__ != trained_with:
        raise RuntimeError(f"scikit-learn {sklearn.__version__} is installed, but the model was trained "
                           f"with {trained_with}. Install the same version (see requirements.txt).")
    model = joblib.load(d / "model.pkl")
    features = json.loads((d / "feature_columns.json").read_text())
    return model, features, meta
