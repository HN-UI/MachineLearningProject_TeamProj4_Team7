# Hull Tactical – Market Prediction

This script trains and evaluates a variety of models (ElasticNet/OLS baselines + deep sequence models) on the Hull Tactical S&P 500 dataset, using a **time-series–safe** setup (chronological splits, walk-forward CV) and a **trading-style metric** (modified Sharpe with volatility cap).

## Main Features

- Supports multiple model types:
  - Tabular / baselines: `baseline_ols`, `baseline_enet`, `ridge`, `lasso`, `svr`, `rf`, `gbr`, `lgbm`, `xgb`
  - Sequence models (`model.py`): `lstm`, `gru`, `tcn`, `tx` (Transformer)
- Time-series handling:
  - Feature engineering from `train.csv` (lags, excess returns, etc.)
  - Chronological train/test split and optional walk-forward cross-validation (`--do_cross`)
- Evaluation:
  - Local backtest using a **modified Sharpe** with:
    - Volatility cap: strategy vol ≤ 1.2 × market vol
    - Penalty if strategy underperforms the market
- Outputs:
  - Kaggle-ready submission file (`--out_csv`, default: `out.csv`) with:
    - `date_id`, `prediction` (signal ∈ [0, 2])
  - Test truth vs prediction file:
    - `test_truth_prediction.csv` with columns:
      - `truth` (true `target` = market forward excess return)
      - `prediction` (model’s predicted `target`)
  - Optional backtest plots if enabled (cumulative returns, etc. – plotting hooks already in place)

## Requirements

- Python 3.x
- Core libraries: `numpy`, `pandas`, `polars`, `scikit-learn`, `torch`, `matplotlib`
- Local modules:
  - `model.py` with a `build_model(model_name, n_features, args)` factory

Install typical dependencies (example):

```bash
pip install numpy pandas polars scikit-learn torch matplotlib
```

## How to Run

- Best configurations for each model are in the `best_config.sh` file.