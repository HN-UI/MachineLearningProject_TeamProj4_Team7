from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
import torch
import numpy as np
import matplotlib.pyplot as plt
# --- new tabular models ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import torch.nn as nn

class ReturnLSTM(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

class ReturnGRU(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self._cut = pad
    def forward(self, x):
        y = super().forward(x)
        return y[:, :, :-self._cut] if self._cut > 0 else y

class TCNBlock(nn.Module):
    def __init__(self, ch, k=5, d=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, k, d), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(ch, ch, k, d), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)

class ReturnTCN(nn.Module):
    def __init__(self, n_features, channels=128, levels=4, kernel_size=5, dropout=0.5):
        super().__init__()
        self.inp = nn.Conv1d(n_features, channels, kernel_size=1)
        blocks = [TCNBlock(channels, k=kernel_size, d=2**i, dropout=dropout) for i in range(levels)]
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1))
    def forward(self, x):
        z = self.inp(x.transpose(1,2))
        z = self.tcn(z)
        y = self.head(z[:, :, -1:]).squeeze(-1)
        return y

class PosEnc(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class ReturnTx(nn.Module):
    def __init__(self, n_features, d_model=256, nhead=8, num_layers=2, dim_ff=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True,
                                         dropout=dropout, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pe = PosEnc(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, x):
        x = self.proj(x); x = self.pe(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.enc(x, mask=mask)
        return self.head(h[:, -1, :])

TABULAR_MODELS = {'baseline_ols','baseline_enet','ridge','lasso','svr','rf','gbr','lgbm','xgb'}

def build_model(name: str, n_features: int, args):
    """
    Returns an unfitted sklearn/lightgbm/xgboost regressor by name.
    Sensible defaults (tune as you like).
    """
    name = name.lower()
    if name == 'baseline_ols':
        return LinearRegression()
    if name == 'rf':
        return RandomForestRegressor(
            n_estimators=args.n_est, max_depth=args.max_depth, n_jobs=-1, random_state=args.seed
        )
    if name == 'lgbm':
        return lgb.LGBMRegressor(
            n_estimators=args.n_est, learning_rate=args.lr, max_depth=args.max_depth,
            subsample=args.subsample, colsample_bytree=args.colsample, random_state=args.seed
        )
    if name == 'xgb':
        return xgb.XGBRegressor(
            n_estimators=args.n_est, learning_rate=args.lr, max_depth=args.max_depth,
            subsample=args.subsample, colsample_bytree=args.colsample, tree_method="hist", random_state=args.seed
        )
    if name == 'lstm':
        return ReturnLSTM(n_features, hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    if name == 'gru':
        return ReturnGRU(n_features, hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    if name == 'tcn':
        return ReturnTCN(n_features, channels=args.tcn_channels, levels=args.tcn_levels, kernel_size=args.tcn_kernel, dropout=args.dropout)
    if name in ('tx','transformer'):
        return ReturnTx(n_features, d_model=args.hidden, nhead=args.tx_heads, num_layers=args.layers, dim_ff=args.tx_ff, dropout=args.dropout)
    raise ValueError(f'Unknown model {name}')
