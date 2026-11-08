# Balance Forecast

Predicts each account's balance at the **end of next month**, standing on day 1 of that month.
One model for all accounts. Input: the clean transactions from the Stage 3 pipeline.

## Setup (once)

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the clean transactions in `data/processed/transactions_clean.csv`
(columns: `account_id, txn_date_time, billing_amount`; export of `transactions_cleaned`).
Optional: `data/processed/features_targets.csv` (Stage 4 output) to cross-check the features.

## Run (from the repo root)

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m train.train` | checks the data, compares models on walk-forward folds, tunes with Optuna, promotes a model, saves it |
| 2 | `python -m inference.test` | scores the saved model on the test month once + nested walk-forward |
| 3 | `python -m inference.predict` | forecasts next month for every account into `db/forecasts.db` |
| 4 | `python -m src.db fill_actuals` | when that month closes, copies the real balances in |
| 5 | `pytest` | runs the tests |

Quick run while developing: `python -m train.train --trials 5`.

## Rules

- A transaction counts in the month of its **transaction date**, not its settle date.
- Pending and unmatched transactions **are** counted in the balance (same rule in training and prediction).
- Every feature for month M uses only transactions **before** month M. `MATCHES_STATUS` is never a feature.
- The balance is the running total of net flows from 0 (no opening balance in the data).
- Split by **whole months**, never at random: test = the last 20% of months; walk-forward folds inside the rest.
- **Promotion:** boosting is used only if it beats the drift baseline by >= $2 with a bootstrap 95% interval below 0.
  Otherwise the drift baseline (last balance + median past move) is the forecast.

## Outputs

| Folder | Files |
|---|---|
| `models/` | `model.pkl`, `optuna_study.pkl`, `feature_columns.json`, `metadata.json` |
| `reports/` | `data_checks.json`, `model_comparison.csv`, `feature_ablation.csv`, `test_metrics.json`, `errors_by_segment.csv`, `worst_accounts.csv`, `nested_walk_forward.csv` |
| `db/forecasts.db` | tables `predictions`, `model_runs`, `test_month_log`; view `monthly_scores` |

## Latest results

_Fill in after running on the real data (see `reports/`)._

## Findings

_One page: see the Stage 5 findings._
