import pandas as pd
import numpy as np
import json, pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, accuracy_score
import lightgbm as lgb

def build_features(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain/loss.replace(0, 1e-10)))
    delta7 = df['close'].diff()
    g7 = delta7.clip(lower=0).rolling(7).mean()
    l7 = (-delta7.clip(upper=0)).rolling(7).mean()
    df['rsi7'] = 100 - (100 / (1 + g7/l7.replace(0, 1e-10)))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_pctb'] = (df['close'] - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    df['bb_bandwidth'] = (4*std20) / (sma20 + 1e-10)
    
    # Williams %R
    highest14 = df['high'].rolling(14).max()
    lowest14 = df['low'].rolling(14).min()
    df['williams_r'] = (highest14 - df['close']) / (highest14 - lowest14 + 1e-10) * -100
    
    # ATR normalized
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr14_norm'] = tr.rolling(14).mean() / df['close']
    
    # EMAs and crossovers
    df['ema5'] = df['close'].ewm(span=5).mean()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema5_20_cross'] = df['ema5'] - df['ema20']
    df['ema20_50_cross'] = df['ema20'] - df['ema50']
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['ema50']
    
    # Momentum
    for n in [1, 3, 5, 10]:
        df[f'mom{n}'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    
    # OBV normalized
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_zscore'] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)
    
    # Volume ratio
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Heikin Ashi
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = (df['open'].shift() + df['close'].shift()) / 2
    df['ha_trend'] = (ha_close > ha_open).astype(int)
    df['ha_body'] = (ha_close - ha_open) / df['close']
    
    # VWAP rolling
    df['vwap20'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_dev'] = (df['close'] - df['vwap20']) / df['vwap20']
    
    # ADX regime
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    atr_s = tr.rolling(14).mean()
    df['adx'] = (plus_dm.rolling(14).mean() / atr_s - minus_dm.rolling(14).mean() / atr_s).abs().rolling(14).mean() * 100
    
    # Volatility regime
    df['vol_regime'] = df['close'].pct_change().rolling(20).std()
    
    # Target: 1 if next close > current close
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

# Load data
print('Loading data...')
df5 = pd.read_csv('data/eth_5m.csv', parse_dates=['timestamp'])
df15 = pd.read_csv('data/eth_15m.csv', parse_dates=['timestamp'])

feature_cols = ['rsi14', 'rsi7', 'macd', 'macd_signal', 'macd_hist',
                'bb_pctb', 'bb_bandwidth', 'williams_r', 'atr14_norm',
                'ema5_20_cross', 'ema20_50_cross', 'price_vs_ema50',
                'mom1', 'mom3', 'mom5', 'mom10',
                'obv_zscore', 'vol_ratio', 'ha_trend', 'ha_body',
                'vwap_dev', 'adx', 'vol_regime']

results = {}

for label, df in [('5m', df5), ('15m', df15)]:
    print(f'\n=== Training {label} model ===')
    df = build_features(df)
    print(f'Dataset: {len(df)} rows, {df["timestamp"].min()} → {df["timestamp"].max()}')
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Walk-forward: 5 folds
    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=63,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            n_estimators=500,
            verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        
        probs = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        best_wr, best_thresh, best_n = 0, 0.5, 0
        for t in np.arange(0.50, 0.75, 0.01):
            mask = (probs >= t) | (probs <= (1-t))
            if mask.sum() < 20: continue
            wr = accuracy_score(y_test[mask], probs[mask] >= 0.5)
            if wr > best_wr:
                best_wr, best_thresh, best_n = wr, t, mask.sum()
        
        brier = brier_score_loss(y_test, probs)
        print(f'  Fold {fold+1}: WR={best_wr:.1%} @ thresh={best_thresh:.2f} ({best_n} trades), Brier={brier:.4f}')
        fold_results.append({'wr': best_wr, 'threshold': best_thresh, 'n_trades': best_n, 'brier': brier})
    
    # Train final model on all data
    final_model = lgb.LGBMClassifier(
        objective='binary', num_leaves=63, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=20, n_estimators=500, verbose=-1
    )
    final_model.fit(X, y)
    
    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    
    avg_wr = np.mean([f['wr'] for f in fold_results])
    avg_thresh = np.mean([f['threshold'] for f in fold_results])
    avg_trades = np.mean([f['n_trades'] for f in fold_results])
    
    print(f'\n{label} Summary:')
    print(f'  Avg WR: {avg_wr:.1%}')
    print(f'  Avg threshold: {avg_thresh:.2f}')
    print(f'  Avg trades/fold: {avg_trades:.0f}')
    print(f'  Top 5 features: {list(importance.keys())[:5]}')
    
    # Save model
    with open(f'model_{label}.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(f'feature_importance_{label}.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    results[label] = {
        'avg_wr': avg_wr,
        'avg_threshold': avg_thresh,
        'avg_trades_per_fold': avg_trades,
        'folds': fold_results,
        'top_features': list(importance.keys())[:10],
        'dataset_rows': len(df)
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print('\n\n=== FINAL RESULTS ===')
for label, r in results.items():
    print(f'{label}: WR={r["avg_wr"]:.1%}, threshold={r["avg_threshold"]:.2f}, top feature: {r["top_features"][0]}')
print('\nAll models saved.')
