"""Time split by whole months + walk-forward folds. Never random."""


def make_split(feat, cfg):
    """Months are computed from the data, so when a new month arrives the test month moves by itself."""
    s = cfg["split"]
    months = sorted(feat["month"].unique())[cfg["data"]["drop_first_months"]:]   # drop months with no history
    n_test = max(1, round(len(months) * s["test_share"]))
    train_months, test_months = months[:-n_test], months[-n_test:]
    fold_months = [m for m in train_months[-s["n_folds"]:]
                   if sum(t < m for t in train_months) >= s["min_train_months"]]
    assert max(train_months) < min(test_months), "a training month comes after the test month"
    return train_months, test_months, fold_months


def folds(train, fold_months):
    """Walk-forward: train on every month before m, validate on m."""
    for m in fold_months:
        yield m, train[train["month"] < m], train[train["month"] == m]


def useless_columns(train, cols):
    """Constant columns and exact copies, decided on the TRAINING rows only.
    Of two copies, keep the one with fewer missing values."""
    drop = {c: "constant" for c in cols if train[c].nunique(dropna=True) <= 1}
    rest = [c for c in cols if c not in drop]
    corr = train[rest].corr().abs()
    for i, a in enumerate(rest):
        for b in rest[i + 1:]:
            if a in drop or b in drop or not corr.loc[a, b] >= 0.999:
                continue
            worse = b if train[b].isna().sum() >= train[a].isna().sum() else a
            drop[worse] = f"copy of {b if worse == a else a}"
    return drop
