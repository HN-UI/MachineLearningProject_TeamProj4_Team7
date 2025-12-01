# Hull Tactical – Market Prediction
# Project-compliant training & inference script
# Notes:
#  - Adds a clear Baseline (ElasticNet/OLS), plus LSTM/GRU/TCN/Transformer options
#  - Uses strict time-series walk-forward CV (no leakage)
#  - Implements local backtest with modified Sharpe metric and volatility cap (≤ 1.2× benchmark)
#  - Produces plots (cumulative returns, volatility ratio)
#  - Exports a Kaggle-ready submission CSV (`/kaggle/working/submission.csv`)
#  - Keeps your existing deep models and data engineering, but extends features (momentum/volatility/calendar)

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import pandas as pd
import polars as pl
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV

import matplotlib.pyplot as plt

from model import build_model
# Optional: LightGBM / XGBoost (guarded imports)

# =========================
# CLI & Logging
# =========================
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', default='lstm', choices=[
        # tabular/baselines
        'baseline_ols','baseline_enet',
        'ridge','lasso','svr','rf','gbr','lgbm','xgb',
        # sequence models
        'lstm','gru','tcn','tx'
    ])
    parser.add_argument('--drop', choices=[
        # tabular/baselines
        'D','M','E','I','P','V','S','MOM1','MOM5','None'
    ])
    parser.add_argument('--data', default='snp', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--tcn_channels', default=128, type=int)
    parser.add_argument('--tcn_levels', default=4, type=int)
    parser.add_argument('--tcn_kernel', default=5, type=int)
    parser.add_argument('--tx_heads', default=4, type=int)
    parser.add_argument('--tx_ff', default=256, type=int)
    parser.add_argument('--seq_len', default=64, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--val_splits', default=5, type=int, help='walk-forward folds')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--out_csv', default='out.csv')
    parser.add_argument('--do_cross', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--n_est', default=1000, type=int)
    parser.add_argument('--max_depth', default=2, type=int)
    parser.add_argument('--subsample', default=2, type=int)
    parser.add_argument('--colsample', default=2, type=int)
    return parser.parse_args(args)


def set_logger(args):
    log_file = f'./logs/{args.model}/{args.seed}_{args.tx_heads}_{args.tx_ff}.log'
    if args.do_test:
        log_file = f'./logs/{args.data}/{args.model}/test__{args.seed}.log'
    
    if os.path.exists(log_file):
        print('Already exists')
        #exit()
        
    os.makedirs(f'./logs/{args.data}/{args.model}', exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# =========================
# Paths & Constants
# =========================

DATA_PATH: Path = Path('./')
MIN_SIGNAL: float = 0.0
MAX_SIGNAL: float = 2.0
SIGNAL_MULTIPLIER: float = 1

CV: int = 10
L1_RATIO: float = 0.5
ALPHAS: np.ndarray = np.logspace(-4, 2, 100)
MAX_ITER: int = 1_000_000


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Data Loading & Feature Engineering
# =========================

@dataclass
class DatasetOutput:
    X_train : pl.DataFrame
    X_test: pl.DataFrame
    y_train: pl.Series
    y_test: pl.Series
    scaler: StandardScaler

@dataclass
class RetToSignalParameters:
    signal_multiplier: float
    min_signal : float = MIN_SIGNAL
    max_signal : float = MAX_SIGNAL

ret_signal_params = RetToSignalParameters(signal_multiplier=SIGNAL_MULTIPLIER)


def load_trainset(args) -> pl.DataFrame:
    df = pl.read_csv(f"./data/{args.data}/train.csv")

    # 1. Basic return and price series
    df = df.with_columns([
        pl.col("forward_returns").shift(1).alias("ret"),
        (100 * (1 + pl.col("forward_returns").shift(1)).cum_prod()).alias("price"),
    ])

    

    # # 2. Momentum (rolling product of 1+ret)
    # df = df.with_columns([
    #     pl.col("ret").alias("MOM1"),
    #     pl.col("ret").add(1).log().rolling_sum(5).exp().sub(1).alias("MOM5"),
    # ])

    # 3. Excess-return momentum
    df = df.with_columns(
        (pl.col("ret") - pl.col("risk_free_rate").shift(1)).alias("ret_ex")
    )

    # 5. Lagged versions
    df = df.with_columns([
        pl.col("forward_returns").shift(1).alias("lagged_forward_returns"),
        pl.col("risk_free_rate").shift(1).alias("lagged_risk_free_rate"),
        pl.col("market_forward_excess_returns").shift(1).alias("lagged_market_forward_excess_returns"),
    ])

    # 6. Rename target
    df = df.rename({"market_forward_excess_returns": "target"})

    # 7. Cast everything except date_id to float
    df = df.with_columns(
        pl.exclude("date_id").cast(pl.Float64, strict=False)
    )
    
    # which prefixes to drop
    drops = [args.drop]  # e.g., drop all S* sentiment features
    if 'None' not in drops:
        for drop in drops:
            cols_to_drop = [c for c in df.columns if c.startswith(drop)]
            if cols_to_drop:
                df = df.drop(cols_to_drop)

    # 8. Drop last 10 rows (Polars version)
    df = df[:-10]

    return df


def load_testset() -> pl.DataFrame:
    """
    Test:
      - already has lagged_forward_returns, lagged_risk_free_rate, lagged_market_forward_excess_returns
      - we keep them as features and create a dummy 'target' column
        (needed only so that create_example_dataset has the same schema).
    """
    df = pl.read_csv(DATA_PATH / 'test.csv')

    df = df.with_columns(
        pl.exclude('date_id').cast(pl.Float64, strict=False)
    )

    # Dummy target (not used for training, just to keep the same schema)
    if 'target' not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias('target'))

    return df



def _add_momentum_vol_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Minimal, assignment-aligned features (momentum/volatility/calendar) without external data.
    - mom_5/21/63: simple momentum over different horizons on P9 proxy
    - vol_21: rolling std on P9 proxy
    - mdraw_63: rolling drawdown proxy from P9
    - dow, month: calendar
    """
    # Use P9 as a price-like proxy available in competition data
    df = df.with_columns([
        (pl.col('P9') / pl.col('P9').shift(5) - 1).alias('mom_5'),
        (pl.col('P9') / pl.col('P9').shift(21) - 1).alias('mom_21'),
        (pl.col('P9') / pl.col('P9').shift(63) - 1).alias('mom_63'),
        pl.col('P9').log().diff().rolling_std(window_size=21).alias('vol_21'),
    ])
    # drawdown proxy
    roll_max = pl.col('P9').rolling_max(window_size=63)
    df = df.with_columns((pl.col('P9')/roll_max - 1).alias('mdraw_63'))

    # calendar (date_id is int; map to pseudo-day-of-week and month buckets)
    df = df.with_columns([
        (pl.col('date_id') % 5).alias('dow'),
        ((pl.col('date_id') // 21) % 12).alias('month'),
    ])
    return df


def create_example_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Use ALL available feature columns from the CSV
    (D*, M*, E*, I*, P*, V*, S*, lagged_* etc.) + engineered
    features (U1, U2, mom_*, vol_21, mdraw_63, dow, month).

    Only 'date_id' and 'target' are reserved (not used as features).
    """

    

    # 2) Add momentum / volatility / calendar features on P9
    base = df
    #base = _add_momentum_vol_calendar(base)

    # 3) All columns except date_id / target are features
    feature_cols: list[str] = [
        c for c in base.columns
        if c not in ('date_id', 'target')
    ]

    # 4) EWM fill only numeric feature columns
    numeric_cols = [
        name for name, dt in base.schema.items()
        if dt in (pl.Float64, pl.Float32) and name in feature_cols
    ]

    filled = base.with_columns([
        pl.col(c).fill_null(pl.col(c).ewm_mean(com=0.5))
        for c in numeric_cols
    ])

    # 5) Final dataset: date_id, target, and ALL feature columns
    return filled.select(['date_id', 'target'] + feature_cols).drop_nulls()




def join_train_test_dataframes(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    common = [c for c in train.columns if c in test.columns]
    return pl.concat([train.select(common), test.select(common)], how='vertical')


def split_dataset(train: pl.DataFrame, test: pl.DataFrame, features: List[str]) -> DatasetOutput:
    X_train = train.drop(['date_id','target'])
    y_train = train.get_column('target')
    X_test  = test.drop(['date_id','target'])
    y_test  = test.get_column('target')

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train)
    X_test_np  = scaler.transform(X_test)

    X_train = pl.from_numpy(X_train_np, schema=features)
    X_test  = pl.from_numpy(X_test_np, schema=features)

    return DatasetOutput(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, scaler=scaler)


def convert_ret_to_signal(ret_arr: np.ndarray, params: RetToSignalParameters) -> np.ndarray:
    return np.clip(ret_arr * params.signal_multiplier + 1.0, params.min_signal, params.max_signal)

# =========================
# Metric, Backtest, and Volatility Cap
# =========================

MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0

class ParticipantVisibleError(Exception):
    pass


def modified_sharpe_with_penalties(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """Same structure as your earlier `score` for local evaluation."""
    if not pd.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    df = solution.copy()
    df['position'] = submission['prediction']
    if df['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position {df["position"].max()} exceeds {MAX_INVESTMENT}')
    if df['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position {df["position"].min()} below {MIN_INVESTMENT}')

    df['strategy_returns'] = df['risk_free_rate'] * (1 - df['position']) + df['position'] * df['forward_returns']

    strat_excess = df['strategy_returns'] - df['risk_free_rate']
    strat_cum = (1 + strat_excess).prod()
    strat_mean = strat_cum ** (1/len(df)) - 1
    strat_std = df['strategy_returns'].std()
    if strat_std == 0:
        raise ParticipantVisibleError('Zero strategy std')

    trading_days = 252
    sharpe = strat_mean / strat_std * np.sqrt(trading_days)

    mkt_excess = df['forward_returns'] - df['risk_free_rate']
    mkt_cum = (1 + mkt_excess).prod()
    mkt_mean = mkt_cum ** (1/len(df)) - 1
    mkt_std = df['forward_returns'].std()
    if mkt_std == 0:
        raise ParticipantVisibleError('Zero market std')

    strat_vol = float(strat_std * np.sqrt(trading_days) * 100)
    mkt_vol   = float(mkt_std   * np.sqrt(trading_days) * 100)

    excess_vol = max(0, strat_vol / mkt_vol - 1.2) if mkt_vol > 0 else 0
    vol_penalty = 1 + excess_vol

    return_gap = max(0, (mkt_mean - strat_mean) * 100 * trading_days)
    return_penalty = 1 + (return_gap ** 2) / 100

    adj = sharpe / (vol_penalty * return_penalty)
    return float(min(adj, 1_000_000))


def enforce_vol_cap(solution: pd.DataFrame, pred_signal: np.ndarray, max_ratio: float = 1.2) -> np.ndarray:
    # auto-align to avoid length errors
    pos = align_pred_to_solution(solution, pred_signal)
    pos = np.clip(pos, MIN_INVESTMENT, MAX_INVESTMENT)

    tmp = solution.copy()
    tmp['position'] = pos

    strat = tmp['risk_free_rate'] * (1 - tmp['position']) + tmp['position'] * tmp['forward_returns']
    strat_std = strat.std()
    mkt_std = tmp['forward_returns'].std()
    if mkt_std == 0:
        return pos

    ratio = strat_std / mkt_std
    if ratio <= max_ratio:
        return pos

    # rescale around neutral 1.0
    dev = pos - 1.0
    scale = max_ratio / ratio
    pos_cap = 1.0 + dev * scale
    return np.clip(pos_cap, MIN_INVESTMENT, MAX_INVESTMENT)


# =========================
# Sequence Dataset & Models
# =========================

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len-1, len(X)):
        Xs.append(X[i-seq_len+1:i+1])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# =========================
# Walk-forward CV & Training
# =========================

def make_walk_splits(n_rows: int, n_splits: int, holdout_frac: float=0.2) -> List[Tuple[slice, slice]]:
    """Produce rolling (train_idx, val_idx) slices by chronology."""
    idxs = np.arange(n_rows)
    step = (n_rows*(1-holdout_frac)) / (n_splits+1)
    splits = []
    for k in range(1, n_splits+1):
        cut = int(k*step)
        tr = slice(0, cut)
        va = slice(cut, int(cut + n_rows*holdout_frac))
        if va.stop <= n_rows:
            splits.append((tr, va))
    return splits

def train_tabular_and_eval(
    X_np: np.ndarray,
    y_np: np.ndarray,
    dates: np.ndarray,
    raw_train_df: pl.DataFrame,
    args,
    model_name: str,
    val_splits: int = 4
) -> list:
    """
    Walk-forward CV for tabular regressors.
    Returns: (best_metric, fitted_model_on_full_train)
    """
    splits = make_walk_splits(len(X_np), n_splits=val_splits, holdout_frac=0.2)
    val_metrics = []

    # we will refit on full train after CV
    # (ElasticNet: we do a nested CV to pick alpha)
    for i, (tr, va) in enumerate(splits, 1):
        Xtr, ytr = X_np[tr], y_np[tr]
        Xva, yva = X_np[va], y_np[va]
        dates_va = dates[va]

        if model_name == 'baseline_enet':
            enet_cv = ElasticNetCV(l1_ratio=L1_RATIO, alphas=ALPHAS, max_iter=MAX_ITER, cv=CV)
            enet_cv.fit(Xtr, ytr)
            model = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=L1_RATIO, max_iter=MAX_ITER)
            model.fit(Xtr, ytr)
        else:
            model = build_model(model_name, N_FEATURES, args)
            model.fit(Xtr, ytr)

        raw_pred = model.predict(Xva)
        signal = convert_ret_to_signal(raw_pred, ret_signal_params)

        sol = (
            raw_train_df.select(['date_id','forward_returns','risk_free_rate'])
            .filter(pl.col('date_id').is_in(dates_va.tolist()))
            .sort('date_id').to_pandas()
        )
        signal = enforce_vol_cap(sol, signal, max_ratio=1.2)
        metric = modified_sharpe_with_penalties(sol, pd.DataFrame({'prediction': signal}))
        logging.info(f'[{model_name.upper()}][Fold {i}/{len(splits)}] metric={metric:.6f}')
        val_metrics.append(metric)

    # Refit on full train once best CV metric is known
    if model_name == 'baseline_enet':
        enet_cv = ElasticNetCV(l1_ratio=L1_RATIO, alphas=ALPHAS, max_iter=MAX_ITER, cv=CV)
        enet_cv.fit(X_np, y_np)
        fitted = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=L1_RATIO, max_iter=MAX_ITER).fit(X_np, y_np)
    else:
        fitted = build_model(model_name, N_FEATURES, args)
        fitted.fit(X_np, y_np)

    return val_metrics

def train_seq_model_and_eval(X_np: np.ndarray, y_np: np.ndarray, dates: np.ndarray, n_features: int,
                             raw_train_df: pl.DataFrame, args) -> list:
    """Walk-forward CV for deep model; returns best metric and best state dict."""
    splits = make_walk_splits(len(X_np), n_splits=args.val_splits, holdout_frac=0.2)
    val_metrics = []

    for i, (tr, va) in enumerate(splits, 1):
        Xtr_np, ytr_np = X_np[tr], y_np[tr]
        Xva_np, yva_np = X_np[va], y_np[va]
        dates_va = dates[va]

        Xtr_win, ytr_win = make_sequences(Xtr_np, ytr_np, SEQ_LEN)
        Xva_win, yva_win = make_sequences(np.concatenate([Xtr_np[-(SEQ_LEN-1):], Xva_np], axis=0),
                                          np.concatenate([ytr_np[-(SEQ_LEN-1):], yva_np], axis=0), SEQ_LEN)

        dl_tr = DataLoader(SeqDS(Xtr_win, ytr_win), batch_size=BATCH, shuffle=True)
        dl_va = DataLoader(SeqDS(Xva_win, yva_win), batch_size=BATCH, shuffle=False)

        model = build_model(args.model, n_features, args).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        crit = nn.MSELoss()


        for ep in range(1, EPOCHS+1):
            model.train(); tr_loss = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); pred = model(xb)
                loss = crit(pred, yb); loss.backward(); opt.step()
                tr_loss += loss.item()*len(xb)
            tr_loss /= len(dl_tr.dataset)
            if not ep % 50:
                logging.info(f'[FULL] Epoch {ep:02d} MSE={tr_loss:.6e}')

        # validation raw preds -> signals -> cap -> metric
        model.eval()
        with torch.no_grad():
            va_raw = model(torch.from_numpy(Xva_win).to(DEVICE)).reshape(-1).cpu().numpy()
        va_signal = convert_ret_to_signal(va_raw, ret_signal_params)

        sol = (
            raw_train_df.select(['date_id','forward_returns','risk_free_rate'])
            .filter(pl.col('date_id').is_in(dates_va.tolist()))
            .sort('date_id')
            .to_pandas()
        )
        va_signal = enforce_vol_cap(sol, va_signal, max_ratio=1.2)
        metric = modified_sharpe_with_penalties(sol, pd.DataFrame({'prediction': va_signal}))
        logging.info(f'[{args.model.upper()}][Fold {i}/{len(splits)}] metric={metric:.6f}')

        val_metrics.append(metric)

    return val_metrics

# =========================
# Plotting (local report figures)
# =========================

def plot_backtest(solution: pd.DataFrame, pred_signal: np.ndarray, title: str, out_png: str):
    df = solution.copy()
    df['position'] = pred_signal
    df['strategy_returns'] = df['risk_free_rate'] * (1 - df['position']) + df['position'] * df['forward_returns']
    strat_cum = (1 + df['strategy_returns']).cumprod()
    mkt_cum   = (1 + df['forward_returns']).cumprod()

    strat_std = df['strategy_returns'].std(); mkt_std = df['forward_returns'].std()
    if mkt_std > 0:
        vol_ratio = (strat_std / mkt_std)
    else:
        vol_ratio = np.nan

    plt.figure(figsize=(8,4))
    plt.plot(strat_cum.values, label='Strategy')
    plt.plot(mkt_cum.values, label='S&P500 proxy')
    plt.title(f"{title} | Vol ratio≈{vol_ratio:.2f}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150)
    plt.close()

def split_dataset_for_eval(
    df: pl.DataFrame,
    features: List[str],
    train_frac: float = 0.6,
) -> tuple[
    np.ndarray, np.ndarray,      # X_train, X_val, X_test
    np.ndarray, np.ndarray,      # y_train, y_val, y_test
    np.ndarray, np.ndarray,      # dates_train, dates_val, dates_test
    StandardScaler
]:
    """
    Chronological 8:1:1 split on a single (feature-engineered) DataFrame.

    - df: must contain columns ['date_id', 'target'] + FEATURES.
    - features: list of feature column names to use.
    - train_frac: fraction for train
    - val_frac: fraction for validation (test_frac = 1 - train_frac - val_frac).
    """
    global X_all_np
    global dates_all
    global n_train
    global n_test
    n = df.height
    n_train = int(n * train_frac)

    # Chronological slices (no shuffle!)
    df_train = df[:n_train]
    df_test  = df[n_train:]

    # Extract features and targets
    X_train_pl = df_train.select(features)
    X_test_pl  = df_test.select(features)

    y_train_np = df_train['target'].to_numpy()
    y_test_np  = df_test['target'].to_numpy()

    dates_train = df_train['date_id'].to_numpy()
    dates_test  = df_test['date_id'].to_numpy()

    # Standardize using TRAIN ONLY
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_pl.to_numpy())
    X_test_np  = scaler.transform(X_test_pl.to_numpy())
    # Concatenate in strict chronological order
    
    X_all_np = np.concatenate([X_train_np, X_test_np], axis=0)
    dates_all = np.concatenate([dates_train, dates_test], axis=0)

    n_train = len(dates_train)
    n_test  = len(dates_test)
    
    return (
        X_train_np, X_test_np,
        y_train_np, y_test_np,
        dates_train, dates_test,
        scaler
    )

def align_pred_to_solution(sol_df: pd.DataFrame, pred_signal: np.ndarray) -> np.ndarray:
    """
    Make prediction length == len(sol_df).
    If longer -> keep last n.
    If shorter -> left-pad with neutral 1.0 to length n.
    """
    pred = np.asarray(pred_signal, dtype=float)
    n, m = len(sol_df), len(pred)
    if m == n:
        return pred
    if m > n:
        return pred[-n:]
    # m < n
    pad = np.full(n - m, 1.0, dtype=float)
    return np.concatenate([pad, pred])

def main(args):
    set_logger(args)
    set_random_seeds(args.seed)
    global EPOCHS, SEQ_LEN, BATCH, N_FEATURES
    EPOCHS = args.epochs
    SEQ_LEN = args.seq_len
    BATCH   = args.batch
    # 1) Load full train.csv (with lagged_* already added)
    full = load_trainset(args)

    # 2) Feature engineering on the full train set
    full_fe = create_example_dataset(full)

    # 3) Feature list (all except date_id / target)
    FORWARD_PATTERNS = ('forward_', 'future_')
    FEATURES = [
        c for c in full_fe.columns
        if c not in ['date_id', 'target']
        and not any(c.startswith(p) for p in FORWARD_PATTERNS)
    ]


    
    
    # 5) For backtest metrics, we still need the raw train.csv with returns
    raw_train_for_metric = (
        pl.read_csv(f'./data/{args.data}/train.csv')
        .rename({'market_forward_excess_returns':'target'})
        .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
    )

    # =========================
    # Choose & Train model per args.model
    # =========================

    base_model = None  # will hold fitted tabular model for test & online inference
    TABULAR_MODELS = {'baseline_ols','baseline_enet','ridge','lasso','svr','rf','gbr','lgbm','xgb'}
    (
            X_train_np, X_test_np,
            y_train_np, y_test_np,
            dates_train, dates_test,
            scaler_full
        ) = split_dataset_for_eval(full_fe, FEATURES)
    if args.do_cross:
        N_FEATURES = X_train_np.shape[1]
        
        if args.model in TABULAR_MODELS:
            # unified tabular training
            val_metrics = train_tabular_and_eval(
                X_train_np, y_train_np, dates_train, raw_train_for_metric, args, model_name=args.model, val_splits=args.val_splits
            )
            logging.info(f'Best CV ({args.model}) metric: {max(val_metrics):.6f}')
            logging.info(f'Mean CV ({args.model}) metric: {sum(val_metrics) / len(val_metrics):.6f}')

        else:
            # We already did CV inside train:
            val_metrics = train_seq_model_and_eval(
                X_train_np, y_train_np, dates_train, N_FEATURES, raw_train_for_metric, args
            )
            logging.info(f'Best CV ({args.model}) metric: {max(val_metrics):.6f}')
            logging.info(f'Mean CV ({args.model}) metric: {sum(val_metrics) / len(val_metrics):.6f}')

    if args.do_test:
        
        N_FEATURES = len(FEATURES)
        # Build the "entire series" features using the SAME columns as during training
        X_all_pl = full_fe.select(FEATURES)
        X_all_np = scaler_full.transform(X_all_pl.to_numpy())

        dates_all = full_fe['date_id'].to_numpy()
        n_train   = len(X_train_np)          # or len(dates_train)
        n_test    = len(X_test_np)
        
        if args.model in TABULAR_MODELS:
            if args.model == 'baseline_enet':
                enet_cv = ElasticNetCV(
                    l1_ratio=L1_RATIO, alphas=ALPHAS,
                    max_iter=MAX_ITER, cv=CV
                )
                enet_cv.fit(X_train_np, y_train_np)
                model = ElasticNet(
                    alpha=enet_cv.alpha_,
                    l1_ratio=L1_RATIO,
                    max_iter=MAX_ITER
                )
                model.fit(X_train_np, y_train_np)
            else:
                model = build_model(args.model, N_FEATURES, args)
                model.fit(X_train_np, y_train_np)
                # coef_df = pd.DataFrame({
                # "feature": FEATURES,
                # "weight": model.coef_
                # })

                # print(coef_df.sort_values("weight", ascending=False))

            # ---- PREDICT ON ENTIRE SERIES ----
            raw_all = model.predict(X_all_np)
           
        else:
            model = build_model(args.model, N_FEATURES, args).to(DEVICE)
            X_train_full = X_train_np
            y_train_full = y_train_np

            Xtr_win, ytr_win = make_sequences(X_train_full, y_train_full, SEQ_LEN)
            dl_tr_full = DataLoader(SeqDS(Xtr_win, ytr_win), batch_size=BATCH, shuffle=True)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            crit = nn.MSELoss()
            best_loss = np.inf
            best_model = None
            for ep in range(1, EPOCHS+1):
                model.train(); tr_loss = 0.0
                for xb, yb in dl_tr_full:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(); pred = model(xb)
                    loss = crit(pred, yb); loss.backward(); opt.step()
                    tr_loss += loss.item() * len(xb)
                tr_loss /= len(dl_tr_full.dataset)
                if best_loss > tr_loss:
                    best_loss = tr_loss
                    best_model = model
                logging.info(f'[FULL] Epoch {ep:02d} MSE={tr_loss:.6e}')
                
            model = best_model
            model.eval()

            # Now predict over the ENTIRE SERIES X_all_np with sliding window
            raw_all = []
            with torch.no_grad():
                if len(X_all_np) >= SEQ_LEN:
                    for i in range(SEQ_LEN - 1, len(X_all_np)):
                        win = X_all_np[i - SEQ_LEN + 1: i + 1]     # seq_len × n_features
                        x = torch.from_numpy(win[None, ...].astype(np.float32)).to(DEVICE)
                        r = model(x).cpu().numpy().ravel()[0]
                        raw_all.append(float(r))
                    # pad the first SEQ_LEN-1 positions with a neutral value (e.g. 0.0 return)
                    raw_all = [0.0] * (SEQ_LEN - 1) + raw_all
                else:
                    raw_all = [0.0] * len(X_all_np)

            raw_all = np.asarray(raw_all, dtype=float)

        # Split back into train / val / test by index
        raw_train_pred = raw_all[:n_train]
        raw_test_pred  = raw_all[n_train:]

        # predictions now align 1:1 with all test rows
        dates_test_aligned = dates_test

        # =========================
        # Build submission & local figures
        # =========================
        if args.model in TABULAR_MODELS:
            signal_test = convert_ret_to_signal(np.asarray(raw_test_pred), ret_signal_params)
            submission = pl.DataFrame({'date_id': dates_test, 'prediction': signal_test})
        else:
            signal_test = convert_ret_to_signal(np.asarray(raw_test_pred), ret_signal_params)
            submission = pl.DataFrame({'date_id': dates_test_aligned, 'prediction': signal_test})

        # Save Kaggle submission
        submission.write_csv(args.out_csv)
        logging.info(f'Saved submission -> {args.out_csv}')


        # Convert to signals
        sig_train = convert_ret_to_signal(raw_train_pred, ret_signal_params)
        sig_test  = convert_ret_to_signal(raw_test_pred,  ret_signal_params)

        # Build solution slices
        sol_train = (
            raw_train_for_metric
            .select(['date_id', 'forward_returns', 'risk_free_rate'])
            .filter(pl.col('date_id').is_in(dates_train.tolist()))
            .sort('date_id')
            .to_pandas()
        )

        sol_test = (
            raw_train_for_metric
            .select(['date_id', 'forward_returns', 'risk_free_rate'])
            .filter(pl.col('date_id').is_in(dates_test.tolist()))
            .sort('date_id')
            .to_pandas()
        )

        # Align lengths (esp. if any padding/rolling weirdness)
        sig_train = align_pred_to_solution(sol_train, sig_train)
        sig_test  = align_pred_to_solution(sol_test,  sig_test)

        # Apply volatility cap
        sig_train_cap = enforce_vol_cap(sol_train, sig_train, max_ratio=1.2)
        sig_test_cap  = enforce_vol_cap(sol_test,  sig_test,  max_ratio=1.2)

        # Final metrics
        train_metric = modified_sharpe_with_penalties(sol_train, pd.DataFrame({'prediction': sig_train_cap}))
        test_metric  = modified_sharpe_with_penalties(sol_test,  pd.DataFrame({'prediction': sig_test_cap}))

        logging.info(f'[Train] metric (with vol-cap): {train_metric:.6f}')
        logging.info(f'[Test]  metric (with vol-cap): {test_metric:.6f}')

        for name, (dates_seg, sol_seg) in {
            "TEST": (dates_test, sol_test),
        }.items():
            ones = np.ones(len(sol_seg))
            ones_cap = enforce_vol_cap(sol_seg, ones, max_ratio=1.2)
            m = modified_sharpe_with_penalties(sol_seg, pd.DataFrame({'prediction': ones_cap}))
            logging.info(f'[Naive=1.0][{name}] metric: {m:.6f}')


    # if args.model in TABULAR_MODELS:
    #     raw_test = base_model.predict(X_test_np)
    # else:
    #     model.eval()

    #     # concat tail of TRAIN+VAL + full TEST for context
    #     if SEQ_LEN > 1:
    #         before_test = np.concatenate([X_train_np, X_val_np], axis=0)
    #         tail = before_test[-(SEQ_LEN-1):] if before_test.shape[0] >= (SEQ_LEN-1) else before_test
    #         concat = np.concatenate([tail, X_test_np], axis=0)
    #     else:
    #         concat = X_test_np

    #     raw_test_list = []
    #     with torch.no_grad():
    #         for i in range(SEQ_LEN - 1, concat.shape[0]):
    #             win = concat[i-SEQ_LEN+1:i+1]
    #             x = torch.from_numpy(win[None, ...].astype(np.float32)).to(DEVICE)
    #             r = model(x).cpu().numpy().ravel()[0]
    #             raw_test_list.append(float(r))

    #     raw_test = np.asarray(raw_test_list, dtype=float)

    # # then:
    # sig_test = convert_ret_to_signal(raw_test, ret_signal_params)
    # sol_test = (
    #     raw_train_for_metric
    #     .select(['date_id', 'forward_returns', 'risk_free_rate'])
    #     .filter(pl.col('date_id').is_in(dates_test.tolist()))
    #     .sort('date_id')
    #     .to_pandas()
    # )
    # sig_test = align_pred_to_solution(sol_test, sig_test)
    # sig_test_cap = enforce_vol_cap(sol_test, sig_test, max_ratio=1.2)
    # test_metric = modified_sharpe_with_penalties(sol_test, pd.DataFrame({'prediction': sig_test_cap}))
    # logging.info(f'[TEST] metric (with vol-cap): {test_metric:.6f}')


    
if __name__ == '__main__':
    main(parse_args())

# # =========================
# # Online inference gateway (predict function)
# # =========================
# # The following mirrors your earlier gateway design. It avoids heavy work per-call and
# # returns a single scalar signal in [0,2]. We do not hard-cap volatility online since
# # full benchmark volatility is unknown in real-time; the cap is enforced in local eval.

# _model_ready = False
# _HISTORY_LEN = None
# _history = None
# _DEVICE = None
# _FEATURES = None
# model_online = None
# scaler_online: StandardScaler = None


# def _model_device(m: nn.Module):
#     try:
#         return next(m.parameters()).device
#     except StopIteration:
#         return torch.device('cpu')


# def _first_time_setup():
#     global _model_ready, _history, _HISTORY_LEN, _DEVICE, _FEATURES, model_online, scaler_online

#     _FEATURES = FEATURES
#     _HISTORY_LEN = int(SEQ_LEN) - 1
#     scaler_online = dataset.scaler

#     if args.model in TABULAR_MODELS:
#         model_online = base_model
#         _DEVICE = torch.device('cpu')
#     else:
#         model_online = build_model(args.model, X_train_np.shape[1], args)
#         # load trained state as before
#         model_online.load_state_dict(best_state if 'best_state' in globals() and best_state is not None else model.state_dict())
#         _DEVICE = _model_device(model)
#         model_online.to(_DEVICE).eval()

#     # Seed history from the same engineered + scaled *train* logic used offline
#     base_train = pl.read_csv(DATA_PATH / 'train.csv')

#     base_train = base_train.with_columns([
#         pl.col('forward_returns').shift(1).alias('lagged_forward_returns'),
#         pl.col('risk_free_rate').shift(1).alias('lagged_risk_free_rate'),
#         pl.col('market_forward_excess_returns').shift(1).alias('lagged_market_forward_excess_returns'),
#     ]).rename({'market_forward_excess_returns': 'target'}).with_columns(
#         pl.exclude('date_id').cast(pl.Float64, strict=False)
#     )

#     fe_train = create_example_dataset(base_train)
#     X_train_pl = fe_train.select(_FEATURES).to_pandas()
#     X_train_np_seed = scaler_online.transform(X_train_pl)

#     hist_seed = X_train_np_seed[-_HISTORY_LEN:] if _HISTORY_LEN > 0 else X_train_np_seed[:0]
#     _history = deque([row.copy() for row in hist_seed], maxlen=_HISTORY_LEN)

#     _model_ready = True



# @torch.no_grad()
# def _predict_window_np(win_np: np.ndarray) -> float:
#     if args.model in TABULAR_MODELS:
#         # Tabular regressors take the latest row features
#         return float(model_online.predict(win_np[-1][None, :])[0])
#     else:
#         x = torch.from_numpy(win_np.astype(np.float32))[None, ...].to(_DEVICE)
#         y = model_online(x)
#         return float(y.squeeze().detach().cpu().item())


# def _scale_last_with_names(df_pl: pl.DataFrame) -> np.ndarray:
#     df = df_pl.select(_FEATURES).to_pandas()
#     return scaler_online.transform(df)


# # Kaggle gateway entry
# # (Signature kept compatible with the competition’s default inference harness)

# def predict(test: pl.DataFrame) -> float:
#     global _model_ready, _history
#     if not _model_ready:
#         _first_time_setup()

#     test = test.rename({'lagged_forward_returns':'target'})
#     fe = create_example_dataset(test)
#     if fe.height == 0:
#         return float(convert_ret_to_signal(np.array([0.0]), ret_signal_params)[0])

#     X_scaled = _scale_last_with_names(fe)
#     x_last = X_scaled[-1]

#     if _HISTORY_LEN > 0:
#         _history.append(x_last)
#         if len(_history) < _HISTORY_LEN:
#             need = _HISTORY_LEN - len(_history)
#             pad = np.tile(_history[0], (need, 1))
#             window = np.vstack([pad, np.array(_history), x_last[None, :]])
#         else:
#             window = np.vstack([np.array(_history), x_last[None, :]])
#     else:
#         window = x_last[None, :]

#     if window.shape[0] > SEQ_LEN:
#         window = window[-SEQ_LEN:]
#     elif window.shape[0] < SEQ_LEN:
#         pad = np.tile(window[0], (SEQ_LEN - window.shape[0], 1))
#         window = np.vstack([pad, window])

#     raw = _predict_window_np(window)
#     sig = float(convert_ret_to_signal(np.array([raw]), ret_signal_params)[0])
#     return sig
