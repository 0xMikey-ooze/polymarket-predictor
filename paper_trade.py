#!/usr/bin/env python3
"""
Paper trading bot for 5-min crypto prediction.
Runs continuously, logs predictions vs actuals.
"""
import requests, pandas as pd, numpy as np, time, json, os, sys
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

LOG_DIR = '/home/node/.openclaw/workspace/polymarket-predictor'
LOG_FILE = os.path.join(LOG_DIR, 'paper_trades.jsonl')
STATS_FILE = os.path.join(LOG_DIR, 'stats.json')
os.makedirs(LOG_DIR, exist_ok=True)

COINS = {
    'BTC': 'BTC-USDT',
    'ETH': 'ETH-USDT',
}

CONFIDENCE_THRESHOLD = 0.55

def fetch_recent_candles(inst_id, bar='5m', limit=300):
    """Fetch recent candles from OKX"""
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

def engineer_features(df):
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
]

def train_model(df):
    """Train XGBoost on all available data"""
    d = engineer_features(df)
    d = d.dropna(subset=FEATURE_COLS + ['target'])
    # Use all but last row (no target for last)
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

def predict(model, df):
    """Get prediction for the latest candle"""
    d = engineer_features(df)
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

def get_pending_predictions():
    """Load predictions that haven't been resolved yet"""
    pending = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get('result') is None:
                    pending.append(entry)
    return pending

def resolve_predictions(coin, current_price):
    """Check if any pending predictions can be resolved"""
    if not os.path.exists(LOG_FILE):
        return []
    
    resolved = []
    lines = []
    with open(LOG_FILE) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry['coin'] == coin and entry.get('result') is None:
                # Check if we have the result now
                predicted_dir = entry['predicted']
                entry_price = entry['entry_price']
                
                if current_price > entry_price:
                    actual = 'UP'
                elif current_price < entry_price:
                    actual = 'DOWN'
                else:
                    actual = 'FLAT'
                
                entry['result'] = actual
                entry['exit_price'] = current_price
                entry['correct'] = (entry['predicted'] == actual) if actual != 'FLAT' else None
                entry['resolved_at'] = datetime.utcnow().isoformat()
                resolved.append(entry)
            lines.append(entry)
    
    # Rewrite file with resolved entries
    with open(LOG_FILE, 'w') as f:
        for entry in lines:
            f.write(json.dumps(entry) + '\n')
    
    return resolved

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print(f"{'='*60}", flush=True)
    print(f"  Paper Trading Bot — 5min Prediction", flush=True)
    print(f"  Coins: {', '.join(COINS.keys())}", flush=True)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"  Log: {LOG_FILE}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Initial model training using recent history
    models = {}
    print("Training models on recent history...", flush=True)
    for coin, inst in COINS.items():
        print(f"  Fetching {coin} training data...", flush=True)
        # Fetch max available from /candles endpoint (recent data)
        all_data = []
        after = ''
        for batch in range(10):  # ~3000 candles = ~10 days
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
        df = df[['time','open','high','low','close','volume']]
        
        print(f"  {coin}: {len(df):,} candles ({df['time'].min()} → {df['time'].max()})", flush=True)
        models[coin] = train_model(df)
        print(f"  {coin} model trained ✓", flush=True)
    
    stats = load_stats()
    print(f"\nModels ready. Starting paper trading loop...\n", flush=True)
    
    last_candle_time = {}
    retrain_counter = 0
    
    while True:
        now = datetime.utcnow()
        
        for coin, inst in COINS.items():
            try:
                # Fetch latest candles
                df = fetch_recent_candles(inst, '5m', 300)
                if df is None or len(df) < 60:
                    continue
                
                latest_time = df['time'].max()
                current_price = df['close'].iloc[-1]
                
                # Resolve any pending predictions
                resolved = resolve_predictions(coin, current_price)
                for r in resolved:
                    status = "✅ WIN" if r.get('correct') else "❌ LOSS" if r.get('correct') is False else "➖ FLAT"
                    print(f"  [{r['coin']}] RESOLVED: predicted {r['predicted']}, actual {r['result']} → {status} (entry:{r['entry_price']:.2f} exit:{r['exit_price']:.2f})", flush=True)
                    if r.get('correct') is not None:
                        stats[coin]['bets'] += 1  # only count resolved bets
                        if r['correct']:
                            stats[coin]['wins'] += 1
                
                # Check if new candle
                if coin in last_candle_time and latest_time <= last_candle_time[coin]:
                    continue
                
                last_candle_time[coin] = latest_time
                stats[coin]['total'] += 1
                
                # Predict
                prob_up, price = predict(models[coin], df)
                if prob_up is None:
                    continue
                
                # Determine signal
                confidence = max(prob_up, 1 - prob_up)
                direction = 'UP' if prob_up > 0.5 else 'DOWN'
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # LOG THE BET
                    entry = {
                        'timestamp': now.isoformat(),
                        'candle_time': str(latest_time),
                        'coin': coin,
                        'entry_price': float(price),
                        'prob_up': float(round(prob_up, 4)),
                        'confidence': float(round(confidence, 4)),
                        'predicted': direction,
                        'result': None,
                    }
                    log_trade(entry)
                    
                    arrow = "🟢 UP" if direction == 'UP' else "🔴 DOWN"
                    print(f"  [{coin}] {latest_time} | {arrow} conf={confidence:.1%} prob_up={prob_up:.3f} price={price:.2f} → BET PLACED", flush=True)
                else:
                    stats[coin]['skips'] += 1
                    print(f"  [{coin}] {latest_time} | SKIP conf={confidence:.1%} (below {CONFIDENCE_THRESHOLD:.0%})", flush=True)
                
                # Print running stats
                s = stats[coin]
                if s['bets'] > 0:
                    wr = s['wins'] / s['bets'] * 100
                    print(f"         Stats: {s['bets']} bets, {s['wins']} wins ({wr:.1f}%), {s['skips']} skips", flush=True)
                
            except Exception as e:
                print(f"  [{coin}] Error: {e}", flush=True)
        
        save_stats(stats)
        
        # Retrain every 288 cycles (~24h)
        retrain_counter += 1
        if retrain_counter >= 288:
            retrain_counter = 0
            print(f"\n  Retraining models...", flush=True)
            for coin, inst in COINS.items():
                try:
                    all_data = []
                    after = ''
                    for batch in range(10):
                        params = {'instId': inst, 'bar': '5m', 'limit': '300'}
                        if after: params['after'] = after
                        r = requests.get('https://www.okx.com/api/v5/market/history-candles', params=params, timeout=15)
                        j = r.json()
                        if j.get('code') != '0' or not j.get('data'): break
                        all_data.extend(j['data'])
                        after = str(int(j['data'][-1][0]))
                        time.sleep(0.15)
                    
                    df = pd.DataFrame(all_data, columns=['time','open','high','low','close','volume','vc','vq','confirm'])
                    for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
                    df['time'] = pd.to_datetime(df['time'].astype(int), unit='ms')
                    df = df.sort_values('time').reset_index(drop=True)[['time','open','high','low','close','volume']]
                    models[coin] = train_model(df)
                    print(f"  {coin} retrained ✓", flush=True)
                except Exception as e:
                    print(f"  {coin} retrain error: {e}", flush=True)
        
        # Wait for next 5-min candle
        # Align to 5-min boundaries + 15s buffer for candle close
        next_5min = now.replace(second=0, microsecond=0)
        next_5min += timedelta(minutes=(5 - next_5min.minute % 5))
        next_5min += timedelta(seconds=15)
        wait = (next_5min - datetime.utcnow()).total_seconds()
        if wait > 0:
            time.sleep(wait)

if __name__ == '__main__':
    main()
