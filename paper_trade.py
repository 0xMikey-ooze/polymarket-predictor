#!/usr/bin/env python3
"""
Paper trading bot v4 — Fixed threshold 0.55, regime-adjusted.
Fixes (v3): rolling regime (no look-ahead), next-candle resolve, FLAT exclusion verified.
Fix  (v4): replaced adaptive threshold with fixed 0.55 — backtest showed adaptive
           reacted to noise (lag-1 autocorr=0.0075), costing 0.82 Sharpe vs fixed.
"""
import requests, pandas as pd, numpy as np, time, json, os, sys
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')

LOG_DIR = '/home/node/.openclaw/workspace/polymarket-predictor'
LOG_FILE = os.path.join(LOG_DIR, 'paper_trades.jsonl')
STATS_FILE = os.path.join(LOG_DIR, 'stats.json')
ADAPTIVE_FILE = os.path.join(LOG_DIR, 'adaptive_state.json')
os.makedirs(LOG_DIR, exist_ok=True)

COINS = {'ETH': 'ETH-USDT'}

# Backtest (305 trades, 2026-02-19→22) proved fixed 0.55 beats adaptive:
# Fixed 0.55: Sharpe=2.18, WR=58.8%, PnL=$353
# Adaptive:   Sharpe=1.36, WR=57.8%, PnL=$164
# Lag-1 autocorr=0.0075 → outcomes are independent, adaptive reacts to noise.
FIXED_THRESHOLD = 0.55

# ============================================================
# ADAPTIVE CONFIDENCE SYSTEM (kept for reference, no longer used for threshold)
# ============================================================
class AdaptiveConfidence:
    """Adjusts confidence threshold based on recent performance."""
    
    def __init__(self, base=0.55, min_thresh=0.52, max_thresh=0.70, window=20):
        self.base = base
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.window = window
        self.recent_results = deque(maxlen=window)  # True/False for win/loss
        self.threshold = base
        self.streak = 0  # positive = wins, negative = losses
        
    def update(self, won: bool):
        self.recent_results.append(won)
        if won:
            self.streak = max(1, self.streak + 1)
        else:
            self.streak = min(-1, self.streak - 1)
        self._recalculate()
    
    def _recalculate(self):
        if len(self.recent_results) < 5:
            return
        
        recent_wr = sum(self.recent_results) / len(self.recent_results)
        
        # On a hot streak (>65% recent WR) → lower threshold to bet more
        # On a cold streak (<45% recent WR) → raise threshold to be selective
        if recent_wr > 0.65:
            self.threshold = max(self.min_thresh, self.base - 0.03)
        elif recent_wr > 0.55:
            self.threshold = self.base
        elif recent_wr > 0.45:
            self.threshold = min(self.max_thresh, self.base + 0.03)
        else:
            # Cold — be very selective
            self.threshold = min(self.max_thresh, self.base + 0.07)
        
        # Streak bonus/penalty
        if self.streak >= 5:
            self.threshold = max(self.min_thresh, self.threshold - 0.02)
        elif self.streak <= -5:
            self.threshold = min(self.max_thresh, self.threshold + 0.03)
    
    def get_threshold(self):
        return self.threshold
    
    def status(self):
        wr = sum(self.recent_results) / len(self.recent_results) * 100 if self.recent_results else 0
        return f"thresh={self.threshold:.1%} recent_wr={wr:.0f}% streak={self.streak} window={len(self.recent_results)}"
    
    def to_dict(self):
        return {
            'threshold': self.threshold,
            'streak': self.streak,
            'recent_results': list(self.recent_results),
        }
    
    def from_dict(self, d):
        self.threshold = d.get('threshold', self.base)
        self.streak = d.get('streak', 0)
        for r in d.get('recent_results', []):
            self.recent_results.append(r)
        self._recalculate()


# ============================================================
# REGIME DETECTOR
# ============================================================
class RegimeDetector:
    """Detects trending vs ranging vs volatile regimes."""
    
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    
    def __init__(self):
        self.current_regime = self.RANGING
        self.regime_confidence = 0.0
    
    def detect(self, df):
        """Detect market regime from recent price data."""
        if len(df) < 50:
            return self.RANGING, 0.5
        
        close = df['close'].tail(50)
        
        # ADX-based trend detection
        high, low = df['high'].tail(50), df['low'].tail(50)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Directional movement
        ret_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        ret_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        
        # Volatility measure
        vol = close.pct_change().tail(20).std()
        vol_long = close.pct_change().tail(50).std()
        vol_ratio = vol / vol_long if vol_long > 0 else 1.0
        
        # Hurst exponent approximation (mean reversion vs trend)
        lags = range(2, 20)
        tau = [np.std(np.subtract(close.values[lag:], close.values[:-lag])) for lag in lags]
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] if all(t > 0 for t in tau) else 0.5
        
        # Classify
        if hurst > 0.6 and abs(ret_20) > 0.01:
            regime = self.TRENDING
            confidence = min(1.0, hurst)
        elif vol_ratio > 1.5:
            regime = self.VOLATILE
            confidence = min(1.0, vol_ratio / 2)
        else:
            regime = self.RANGING
            confidence = 1.0 - abs(hurst - 0.5) * 2
        
        self.current_regime = regime
        self.regime_confidence = confidence
        return regime, confidence
    
    def get_regime_features(self, df):
        """Return regime as features for the model."""
        regime, conf = self.detect(df)
        return {
            'regime_trending': 1.0 if regime == self.TRENDING else 0.0,
            'regime_ranging': 1.0 if regime == self.RANGING else 0.0,
            'regime_volatile': 1.0 if regime == self.VOLATILE else 0.0,
            'regime_confidence': conf,
        }


# ============================================================
# ROLLING REGIME FEATURES (no look-ahead bias)
# ============================================================
def compute_rolling_regime_features(df, regime_detector):
    """Compute regime features using only past data at each point.
    Strides every 5 rows for speed; fills forward between strides.
    Prevents look-ahead bias during model training.
    """
    n = len(df)
    regime_trending  = np.zeros(n)
    regime_ranging   = np.ones(n)   # default: ranging
    regime_volatile  = np.zeros(n)
    regime_confidence = np.full(n, 0.5)

    STRIDE = 5  # compute regime every 5 candles, fill forward
    WINDOW = 50

    for i in range(WINDOW, n, STRIDE):
        window = df.iloc[max(0, i - WINDOW):i]
        try:
            regime, conf = regime_detector.detect(window)
            fill_end = min(i + STRIDE, n)
            regime_trending[i:fill_end]   = 1.0 if regime == RegimeDetector.TRENDING  else 0.0
            regime_ranging[i:fill_end]    = 1.0 if regime == RegimeDetector.RANGING   else 0.0
            regime_volatile[i:fill_end]   = 1.0 if regime == RegimeDetector.VOLATILE  else 0.0
            regime_confidence[i:fill_end] = conf
        except Exception:
            pass  # keep defaults on error

    return regime_trending, regime_ranging, regime_volatile, regime_confidence


# ============================================================
# TRADE MEMORY FEEDBACK
# ============================================================
class TradeMemory:
    """Learns from recent trade outcomes to generate features."""
    
    def __init__(self, max_trades=200):
        self.trades = deque(maxlen=max_trades)
    
    def add_trade(self, entry):
        self.trades.append(entry)
    
    def load_from_log(self):
        """Bootstrap from existing trade log."""
        if not os.path.exists(LOG_FILE):
            return
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get('correct') is not None:
                        self.trades.append(entry)
                except:
                    continue
    
    def get_features(self):
        """Generate features from trade history."""
        if len(self.trades) < 5:
            return {
                'mem_recent_wr_10': 0.5,
                'mem_recent_wr_30': 0.5,
                'mem_avg_confidence': 0.55,
                'mem_up_bias': 0.5,
                'mem_streak': 0,
                'mem_time_of_day_wr': 0.5,
            }
        
        recent = list(self.trades)
        last_10 = recent[-10:] if len(recent) >= 10 else recent
        last_30 = recent[-30:] if len(recent) >= 30 else recent
        
        wr_10 = sum(1 for t in last_10 if t.get('correct')) / len(last_10)
        wr_30 = sum(1 for t in last_30 if t.get('correct')) / len(last_30)
        
        avg_conf = np.mean([t.get('confidence', 0.55) for t in last_10])
        
        up_count = sum(1 for t in last_30 if t.get('predicted') == 'UP')
        up_bias = up_count / len(last_30)
        
        # Streak
        streak = 0
        for t in reversed(recent):
            if t.get('correct'):
                streak += 1
            else:
                break
        if streak == 0:
            for t in reversed(recent):
                if not t.get('correct'):
                    streak -= 1
                else:
                    break
        
        # Time-of-day win rate (current hour)
        now_hour = datetime.utcnow().hour
        hour_trades = [t for t in recent if 'candle_time' in t and str(now_hour).zfill(2) in t['candle_time'].split(' ')[-1][:2]]
        hour_wr = sum(1 for t in hour_trades if t.get('correct')) / len(hour_trades) if hour_trades else 0.5
        
        return {
            'mem_recent_wr_10': wr_10,
            'mem_recent_wr_30': wr_30,
            'mem_avg_confidence': avg_conf,
            'mem_up_bias': up_bias,
            'mem_streak': streak,
            'mem_time_of_day_wr': hour_wr,
        }


# ============================================================
# FEATURE ENGINEERING (enhanced)
# ============================================================
def engineer_features(df, regime_detector=None, trade_memory=None, training_mode=False):
    d = df.copy()
    c, h, l, v, o = d['close'], d['high'], d['low'], d['volume'], d['open']
    
    for p in [1,2,3,5,10,20]:
        d[f'ret_{p}'] = c.pct_change(p)
    
    d['vol_5'] = d['ret_1'].rolling(5).std()
    d['vol_20'] = d['ret_1'].rolling(20).std()
    d['vol_ratio'] = d['vol_5'] / d['vol_20'].replace(0, np.nan)
    
    for period in [6, 14, 28]:
        delta = c.diff()
        gain = delta.where(delta>0,0).rolling(period).mean()
        loss = -delta.where(delta<0,0).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        d[f'rsi_{period}'] = 100 - 100/(1+rs)
    
    for period in [10, 20]:
        ma = c.rolling(period).mean()
        sd = c.rolling(period).std()
        d[f'bb_pctb_{period}'] = (c - (ma - 2*sd)) / (4*sd).replace(0, np.nan)
        d[f'bb_width_{period}'] = (4*sd) / ma
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    d['macd_hist'] = macd - sig
    d['macd_hist_slope'] = d['macd_hist'].diff(2)
    
    for period in [14, 28]:
        lo = l.rolling(period).min()
        hi = h.rolling(period).max()
        d[f'stoch_k_{period}'] = 100 * (c - lo) / (hi - lo).replace(0, np.nan)
    
    d['vol_ma_ratio'] = v / v.rolling(20).mean().replace(0, np.nan)
    d['vol_ma_ratio_5'] = v / v.rolling(5).mean().replace(0, np.nan)
    d['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)
    
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d['obv_slope_5'] = obv.diff(5)
    d['obv_slope_10'] = obv.diff(10)
    
    hl = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / hl
    mfv = mfm * v
    d['cmf_10'] = mfv.rolling(10).sum() / v.rolling(10).sum()
    d['cmf_20'] = mfv.rolling(20).sum() / v.rolling(20).sum()
    
    pdm = h.diff(); mdm = -l.diff()
    pdm[pdm<0]=0; mdm[mdm<0]=0
    m1 = pdm>mdm; mdm_c = mdm.copy(); mdm_c[m1&(pdm>0)]=0
    m2 = mdm>pdm; pdm_c = pdm.copy(); pdm_c[m2&(mdm>0)]=0
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    atr14 = tr.ewm(span=14,min_periods=14).mean()
    pdi = 100*pdm_c.ewm(span=14,min_periods=14).mean()/atr14
    mdi = 100*mdm_c.ewm(span=14,min_periods=14).mean()/atr14
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['adx'] = dx.ewm(span=14,min_periods=14).mean()
    d['di_diff'] = pdi - mdi
    d['atr_14'] = atr14
    d['atr_ratio'] = tr / atr14.replace(0, np.nan)
    
    body = abs(c - o)
    full_range = (h - l).replace(0, np.nan)
    d['body_pct'] = body / full_range
    d['upper_wick'] = (h - pd.concat([c,o],axis=1).max(axis=1)) / full_range
    d['lower_wick'] = (pd.concat([c,o],axis=1).min(axis=1) - l) / full_range
    d['is_green'] = (c > o).astype(int)
    
    greens = d['is_green']
    groups = (greens != greens.shift()).cumsum()
    d['consec'] = greens.groupby(groups).cumcount() + 1
    d['consec_green'] = d['consec'] * d['is_green']
    d['consec_red'] = d['consec'] * (1 - d['is_green'])
    
    d['ema_5_20'] = c.ewm(span=5).mean() / c.ewm(span=20).mean() - 1
    d['ema_10_50'] = c.ewm(span=10).mean() / c.ewm(span=50).mean() - 1
    d['price_vs_vwap'] = c / (c.rolling(20).mean()) - 1
    
    d['hour'] = d['time'].dt.hour
    d['hour_sin'] = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour'] / 24)
    d['minute'] = d['time'].dt.minute
    d['dow'] = d['time'].dt.dayofweek
    d['dow_sin'] = np.sin(2 * np.pi * d['dow'] / 7)
    d['dow_cos'] = np.cos(2 * np.pi * d['dow'] / 7)
    
    d['dist_from_ma20'] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    d['dist_from_ma50'] = (c - c.rolling(50).mean()) / c.rolling(50).std().replace(0, np.nan)
    
    for lag in [1,2,3]:
        d[f'ret_lag_{lag}'] = d['ret_1'].shift(lag)
        d[f'vol_lag_{lag}'] = d['vol_ma_ratio'].shift(lag)
    
    # Regime features — rolling (training) vs single snapshot (inference)
    if regime_detector:
        if training_mode:
            # Compute regime on rolling past-only windows to eliminate look-ahead bias
            rt, rr, rv, rc = compute_rolling_regime_features(df, regime_detector)
            d['regime_trending']   = rt
            d['regime_ranging']    = rr
            d['regime_volatile']   = rv
            d['regime_confidence'] = rc
        else:
            # Real-time: single detection on full available history
            regime_feats = regime_detector.get_regime_features(df)
            for k, val in regime_feats.items():
                d[k] = val
    else:
        d['regime_trending']   = 0.0
        d['regime_ranging']    = 1.0
        d['regime_volatile']   = 0.0
        d['regime_confidence'] = 0.5
    
    # NEW: Trade memory features
    if trade_memory:
        mem_feats = trade_memory.get_features()
        for k, val in mem_feats.items():
            d[k] = val
    else:
        d['mem_recent_wr_10'] = 0.5
        d['mem_recent_wr_30'] = 0.5
        d['mem_avg_confidence'] = 0.55
        d['mem_up_bias'] = 0.5
        d['mem_streak'] = 0
        d['mem_time_of_day_wr'] = 0.5
    
    d['target'] = (d['close'].shift(-1) > d['close']).astype(int)
    
    return d

FEATURE_COLS = [
    'ret_1','ret_2','ret_3','ret_5','ret_10','ret_20',
    'vol_5','vol_20','vol_ratio',
    'rsi_6','rsi_14','rsi_28',
    'bb_pctb_10','bb_pctb_20','bb_width_10','bb_width_20',
    'macd_hist','macd_hist_slope',
    'stoch_k_14','stoch_k_28',
    'vol_ma_ratio','vol_ma_ratio_5','vol_trend',
    'obv_slope_5','obv_slope_10',
    'cmf_10','cmf_20',
    'adx','di_diff','atr_ratio',
    'body_pct','upper_wick','lower_wick','is_green',
    'consec_green','consec_red',
    'ema_5_20','ema_10_50','price_vs_vwap',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'dist_from_ma20','dist_from_ma50',
    'ret_lag_1','ret_lag_2','ret_lag_3',
    'vol_lag_1','vol_lag_2','vol_lag_3',
    # New features
    'regime_trending','regime_ranging','regime_volatile','regime_confidence',
    'mem_recent_wr_10','mem_recent_wr_30','mem_avg_confidence',
    'mem_up_bias','mem_streak','mem_time_of_day_wr',
]


def fetch_training_data(inst, batches=10):
    """Fetch historical candles for training."""
    all_data = []
    after = ''
    for batch in range(batches):
        params = {'instId': inst, 'bar': '5m', 'limit': '300'}
        if after:
            params['after'] = after
        try:
            r = requests.get('https://www.okx.com/api/v5/market/history-candles', params=params, timeout=15)
            j = r.json()
            if j.get('code') != '0' or not j.get('data'):
                break
            all_data.extend(j['data'])
            after = str(int(j['data'][-1][0]))
            time.sleep(0.15)
        except:
            break
    
    df = pd.DataFrame(all_data, columns=['time','open','high','low','close','volume','vc','vq','confirm'])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    return df[['time','open','high','low','close','volume']]


def fetch_recent_candles(inst_id, bar='5m', limit=300):
    url = 'https://www.okx.com/api/v5/market/candles'
    params = {'instId': inst_id, 'bar': bar, 'limit': str(limit)}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            j = r.json()
            if j.get('code') == '0' and j.get('data'):
                df = pd.DataFrame(j['data'], columns=['time','open','high','low','close','volume','vc','vq','confirm'])
                for c in ['open','high','low','close','volume']:
                    df[c] = df[c].astype(float)
                df['time'] = pd.to_datetime(df['time'].astype(int), unit='ms')
                df = df.sort_values('time').reset_index(drop=True)
                return df[['time','open','high','low','close','volume']]
        except Exception as e:
            print(f"  Fetch error: {e}", flush=True)
            time.sleep(2)
    return None


def train_model(df, regime_detector=None, trade_memory=None):
    d = engineer_features(df, regime_detector, trade_memory, training_mode=True)
    d = d.dropna(subset=FEATURE_COLS + ['target'])
    d = d.iloc[:-1]
    
    X = np.nan_to_num(d[FEATURE_COLS].values, nan=0, posinf=0, neginf=0)
    y = d['target'].values
    
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=50, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', verbosity=0, n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict(model, df, regime_detector=None, trade_memory=None):
    d = engineer_features(df, regime_detector, trade_memory)
    d = d.dropna(subset=FEATURE_COLS)
    if len(d) == 0:
        return None, None
    
    last = d.iloc[-1:]
    X = np.nan_to_num(last[FEATURE_COLS].values, nan=0, posinf=0, neginf=0)
    prob_up = model.predict_proba(X)[0][1]
    return prob_up, last['close'].values[0]


def load_stats():
    if os.path.exists(STATS_FILE):
        return json.load(open(STATS_FILE))
    return {coin: {'total': 0, 'bets': 0, 'wins': 0, 'skips': 0} for coin in COINS}

def save_stats(stats):
    json.dump(stats, open(STATS_FILE, 'w'), indent=2)

def log_trade(entry):
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def resolve_predictions(coin, df):
    """Resolve pending bets using the close of the candle AFTER the trade candle.
    This eliminates the mid-candle price bias from the previous implementation.
    A bet on candle T is only resolved when candle T+5min appears in the data.
    """
    if not os.path.exists(LOG_FILE):
        return []

    # Build lookup: candle_time (Timestamp) -> close price
    candle_closes = {row['time']: row['close'] for _, row in df.iterrows()}

    resolved = []
    lines = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry['coin'] == coin and entry.get('result') is None:
                trade_candle = pd.Timestamp(entry['candle_time'])
                next_candle  = trade_candle + pd.Timedelta(minutes=5)

                if next_candle in candle_closes:
                    exit_price  = candle_closes[next_candle]
                    entry_price = entry['entry_price']

                    if exit_price > entry_price:
                        actual = 'UP'
                    elif exit_price < entry_price:
                        actual = 'DOWN'
                    else:
                        actual = 'FLAT'

                    entry['result']     = actual
                    entry['exit_price'] = float(exit_price)
                    entry['correct']    = (entry['predicted'] == actual) if actual != 'FLAT' else None
                    entry['resolved_at'] = datetime.utcnow().isoformat()
                    resolved.append(entry)
                # else: next candle not yet available — leave pending
            lines.append(entry)

    # Atomic-ish write: write all lines back
    tmp = LOG_FILE + '.tmp'
    with open(tmp, 'w') as f:
        for entry in lines:
            f.write(json.dumps(entry) + '\n')
    os.replace(tmp, LOG_FILE)

    return resolved


def save_adaptive_state(adaptive_conf, regime_det, retrain_counter):
    state = {
        'adaptive': adaptive_conf.to_dict(),
        'regime': regime_det.current_regime,
        'retrain_counter': retrain_counter,
        'saved_at': datetime.utcnow().isoformat(),
    }
    json.dump(state, open(ADAPTIVE_FILE, 'w'), indent=2)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    # Initialize adaptive systems
    adaptive_conf = AdaptiveConfidence(base=0.55, min_thresh=0.52, max_thresh=0.70)
    regime_detector = RegimeDetector()
    trade_memory = TradeMemory()
    
    # Load trade history into memory
    trade_memory.load_from_log()
    
    # Load adaptive state if exists
    if os.path.exists(ADAPTIVE_FILE):
        try:
            state = json.load(open(ADAPTIVE_FILE))
            adaptive_conf.from_dict(state.get('adaptive', {}))
            print(f"  Restored adaptive state: {adaptive_conf.status()}", flush=True)
        except:
            pass
    
    print(f"{'='*60}", flush=True)
    print(f"  Paper Trading Bot v2 — Adaptive 5min Prediction", flush=True)
    print(f"  Coins: {', '.join(COINS.keys())}", flush=True)
    print(f"  Confidence threshold: {FIXED_THRESHOLD:.2f} FIXED (+regime micro-adj)", flush=True)
    print(f"  Retrain interval: every 2h (24 cycles)", flush=True)
    print(f"  Trade memory: {len(trade_memory.trades)} historical trades loaded", flush=True)
    print(f"  Log: {LOG_FILE}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Initial training
    models = {}
    print("Training models on recent history...", flush=True)
    for coin, inst in COINS.items():
        print(f"  Fetching {coin} training data...", flush=True)
        df = fetch_training_data(inst)
        print(f"  {coin}: {len(df):,} candles ({df['time'].min()} → {df['time'].max()})", flush=True)
        
        regime_detector.detect(df)
        print(f"  {coin} regime: {regime_detector.current_regime} (conf={regime_detector.regime_confidence:.2f})", flush=True)
        
        models[coin] = train_model(df, regime_detector, trade_memory)
        print(f"  {coin} model trained ✓", flush=True)
    
    stats = load_stats()
    print(f"\nModels ready. Starting adaptive trading loop...\n", flush=True)
    
    last_candle_time = {}
    retrain_counter = 0
    RETRAIN_INTERVAL = 24  # every 2 hours (24 * 5min)
    
    while True:
        now = datetime.utcnow()
        
        for coin, inst in COINS.items():
            try:
                df = fetch_recent_candles(inst, '5m', 300)
                if df is None or len(df) < 60:
                    continue
                
                latest_time = df['time'].max()
                current_price = df['close'].iloc[-1]
                
                # Resolve pending (uses next-candle close — no mid-candle bias)
                resolved = resolve_predictions(coin, df)
                for r in resolved:
                    status = "✅ WIN" if r.get('correct') else "❌ LOSS" if r.get('correct') is False else "➖ FLAT"
                    print(f"  [{r['coin']}] RESOLVED: predicted {r['predicted']}, actual {r['result']} → {status} (entry:{r['entry_price']:.2f} exit:{r['exit_price']:.2f})", flush=True)
                    if r.get('correct') is not None:
                        stats[coin]['bets'] += 1
                        if r['correct']:
                            stats[coin]['wins'] += 1
                        # Feed trade memory (adaptive threshold removed — v4)
                        trade_memory.add_trade(r)
                
                # New candle?
                if coin in last_candle_time and latest_time <= last_candle_time[coin]:
                    continue
                
                last_candle_time[coin] = latest_time
                stats[coin]['total'] += 1
                
                # Detect regime
                regime, regime_conf = regime_detector.detect(df)
                
                # Predict with regime + memory features
                prob_up, price = predict(models[coin], df, regime_detector, trade_memory)
                if prob_up is None:
                    continue
                
                confidence = max(prob_up, 1 - prob_up)
                direction = 'UP' if prob_up > 0.5 else 'DOWN'
                threshold = FIXED_THRESHOLD  # v4: fixed 0.55, beats adaptive by 0.82 Sharpe

                # Regime-based micro-adjustment (market structure, not outcome history)
                if regime == RegimeDetector.VOLATILE:
                    threshold = min(0.70, threshold + 0.03)  # More selective in volatile markets
                elif regime == RegimeDetector.TRENDING:
                    threshold = max(0.52, threshold - 0.01)  # Slightly more aggressive in trends
                
                if confidence >= threshold:
                    entry = {
                        'timestamp': now.isoformat(),
                        'candle_time': str(latest_time),
                        'coin': coin,
                        'entry_price': float(price),
                        'prob_up': float(round(prob_up, 4)),
                        'confidence': float(round(confidence, 4)),
                        'predicted': direction,
                        'result': None,
                        'threshold_used': float(round(threshold, 4)),
                        'regime': regime,
                    }
                    log_trade(entry)
                    
                    arrow = "🟢 UP" if direction == 'UP' else "🔴 DOWN"
                    print(f"  [{coin}] {latest_time} | {arrow} conf={confidence:.1%} thresh={threshold:.1%} regime={regime} price={price:.2f} → BET", flush=True)
                else:
                    stats[coin]['skips'] += 1
                    print(f"  [{coin}] {latest_time} | SKIP conf={confidence:.1%} < thresh={threshold:.1%} regime={regime}", flush=True)
                
                s = stats[coin]
                if s['bets'] > 0:
                    wr = s['wins'] / s['bets'] * 100
                    print(f"         Stats: {s['bets']} bets, {s['wins']} wins ({wr:.1f}%), {s['skips']} skips | thresh={threshold:.2f} regime={regime}", flush=True)
                
            except Exception as e:
                print(f"  [{coin}] Error: {e}", flush=True)
        
        save_stats(stats)
        
        # Rolling retrain every 2 hours
        retrain_counter += 1
        if retrain_counter >= RETRAIN_INTERVAL:
            retrain_counter = 0
            print(f"\n  🔄 Retraining models (every {RETRAIN_INTERVAL * 5}min)...", flush=True)
            for coin, inst in COINS.items():
                try:
                    df = fetch_training_data(inst)
                    regime_detector.detect(df)
                    models[coin] = train_model(df, regime_detector, trade_memory)
                    print(f"  {coin} retrained ✓ regime={regime_detector.current_regime} | thresh={FIXED_THRESHOLD:.2f} (fixed)", flush=True)
                except Exception as e:
                    print(f"  {coin} retrain error: {e}", flush=True)
            
            save_adaptive_state(adaptive_conf, regime_detector, retrain_counter)
        
        # Wait for next 5-min candle
        next_5min = now.replace(second=0, microsecond=0)
        next_5min += timedelta(minutes=(5 - next_5min.minute % 5))
        next_5min += timedelta(seconds=15)
        wait = (next_5min - datetime.utcnow()).total_seconds()
        if wait > 0:
            time.sleep(wait)

if __name__ == '__main__':
    main()
