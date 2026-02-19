#!/usr/bin/env python3
"""Polymarket BTC Up/Down Paper Trading Bot with XGBoost ML Model."""

import json
import time
import math
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
DOME_API_KEY = "807f46adf40424114c15ed7b2429b6e9a095e370"
KUCOIN_URL = "https://api.kucoin.com/api/v1/market/candles"
DOME_URL = "https://api.domeapi.io/v1/polymarket/markets"

# ── API helpers ──────────────────────────────────────────────────────────────

def fetch_json(url, params=None, headers=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [retry {i+1}/{retries}] {e}")
            if i < retries - 1:
                time.sleep(2 ** i)
    return None

def fetch_kucoin_candles(end_at=None, limit=1500):
    """Fetch 5-min candles from KuCoin. Returns list of [time,open,close,high,low,volume,turnover] oldest-first."""
    params = {"type": "5min", "symbol": "BTC-USDT"}
    if end_at:
        params["endAt"] = int(end_at)
        params["startAt"] = int(end_at) - limit * 300
    data = fetch_json(KUCOIN_URL, params=params)
    if not data or data.get("code") != "200000" or not data.get("data"):
        return []
    # KuCoin returns newest first; reverse to oldest first
    rows = data["data"][::-1]
    # Each row: [time, open, close, high, low, volume, turnover] all strings
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])] for r in rows]

def fetch_all_candles(n=10000):
    """Fetch ~n candles by paginating backwards."""
    print(f"Fetching {n} candles from KuCoin...")
    all_candles = []
    end_at = int(time.time())
    while len(all_candles) < n:
        batch = fetch_kucoin_candles(end_at=end_at, limit=1500)
        if not batch:
            print(f"  Got empty batch, have {len(all_candles)} candles so far")
            break
        all_candles = batch + all_candles  # prepend older
        end_at = batch[0][0] - 1  # go further back
        print(f"  Fetched batch, total: {len(all_candles)} candles")
        time.sleep(0.3)
    # Deduplicate by timestamp
    seen = set()
    deduped = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            deduped.append(c)
    deduped.sort(key=lambda x: x[0])
    print(f"Total unique candles: {len(deduped)}")
    return deduped[-n:] if len(deduped) > n else deduped

def fetch_polymarket_btc():
    """Try to find BTC up/down 5-min markets from Dome API."""
    try:
        data = fetch_json(DOME_URL, params={"tag": "crypto", "limit": 50},
                          headers={"Authorization": f"Bearer {DOME_API_KEY}"})
        if data and isinstance(data, dict) and data.get("data"):
            markets = data["data"]
        elif data and isinstance(data, list):
            markets = data
        else:
            return None
        for m in markets:
            slug = m.get("slug", "") or m.get("question", "")
            if "btc" in slug.lower() and ("up" in slug.lower() or "down" in slug.lower()):
                return m
    except:
        pass
    return None

# ── Feature engineering ──────────────────────────────────────────────────────

def ema(values, period):
    """Compute EMA array."""
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result

def compute_features(candles):
    """Compute 33 features for each candle. Returns (features_array, valid_mask)."""
    n = len(candles)
    ts = np.array([c[0] for c in candles])
    op = np.array([c[1] for c in candles])
    cl = np.array([c[2] for c in candles])
    hi = np.array([c[3] for c in candles])
    lo = np.array([c[4] for c in candles])
    vol = np.array([c[5] for c in candles])

    ret = np.zeros(n)
    ret[1:] = (cl[1:] - cl[:-1]) / cl[:-1]

    # EMAs
    ema5 = ema(cl, 5)
    ema20 = ema(cl, 20)
    ema50 = ema(cl, 50)
    vol_ema20 = ema(vol, 20)

    # RSI(14)
    rsi = np.full(n, np.nan)
    gains = np.maximum(ret, 0)
    losses = np.maximum(-ret, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > 14:
        avg_gain[14] = np.mean(gains[1:15])
        avg_loss[14] = np.mean(losses[1:15])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gains[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + losses[i]) / 14
        for i in range(14, n):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger Bands (20,2)
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    for i in range(19, n):
        window = cl[i-19:i+1]
        bb_mid[i] = np.mean(window)
        bb_std[i] = np.std(window, ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pctb = np.where(bb_std > 0, (cl - bb_lower) / (bb_upper - bb_lower), 0.5)
    bb_width = np.where(bb_mid > 0, 4 * bb_std / bb_mid, 0)

    # MACD
    ema12 = ema(cl, 12)
    ema26 = ema(cl, 26)
    macd_line = ema12 - ema26
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # ATR(14)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr = np.full(n, np.nan)
    if n > 14:
        atr[14] = np.mean(tr[1:15])
        for i in range(15, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14

    # Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if ret[i] > 0:
            streak[i] = max(streak[i-1], 0) + 1
        elif ret[i] < 0:
            streak[i] = min(streak[i-1], 0) - 1

    # Build feature matrix
    features = np.full((n, 33), np.nan)
    for i in range(n):
        hour = datetime.fromtimestamp(ts[i], tz=timezone.utc).hour
        features[i, 0] = ret[i]
        features[i, 1] = ret[i-1] if i >= 1 else 0
        features[i, 2] = ret[i-2] if i >= 2 else 0
        features[i, 3] = np.sum(ret[max(0,i-4):i+1])
        features[i, 4] = np.sum(ret[max(0,i-11):i+1])
        features[i, 5] = (cl[i] - op[i]) / op[i] if op[i] else 0
        features[i, 6] = (hi[i] - max(op[i], cl[i])) / op[i] if op[i] else 0
        features[i, 7] = (min(op[i], cl[i]) - lo[i]) / op[i] if op[i] else 0
        features[i, 8] = (hi[i] - lo[i]) / op[i] if op[i] else 0
        features[i, 9] = rsi[i]
        features[i, 10] = rsi[i] - rsi[i-1] if i >= 1 and not np.isnan(rsi[i]) and not np.isnan(rsi[i-1]) else 0
        features[i, 11] = bb_pctb[i]
        features[i, 12] = bb_width[i]
        features[i, 13] = macd_hist[i]
        features[i, 14] = macd_hist[i] - macd_hist[i-1] if i >= 1 and not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i-1]) else 0
        features[i, 15] = macd_line[i]
        features[i, 16] = (cl[i] / ema5[i] - 1) if not np.isnan(ema5[i]) and ema5[i] else 0
        features[i, 17] = (cl[i] / ema20[i] - 1) if not np.isnan(ema20[i]) and ema20[i] else 0
        features[i, 18] = (cl[i] / ema50[i] - 1) if not np.isnan(ema50[i]) and ema50[i] else 0
        features[i, 19] = (ema5[i] / ema20[i] - 1) if not np.isnan(ema5[i]) and not np.isnan(ema20[i]) and ema20[i] else 0
        features[i, 20] = (ema20[i] / ema50[i] - 1) if not np.isnan(ema20[i]) and not np.isnan(ema50[i]) and ema50[i] else 0
        features[i, 21] = vol[i] / vol_ema20[i] if not np.isnan(vol_ema20[i]) and vol_ema20[i] else 1
        features[i, 22] = (vol[i] / vol[i-1] - 1) if i >= 1 and vol[i-1] else 0
        features[i, 23] = streak[i]
        features[i, 24] = abs(streak[i])
        features[i, 25] = atr[i] / cl[i] if not np.isnan(atr[i]) and cl[i] else 0
        features[i, 26] = (cl[i] - lo[i]) / (hi[i] - lo[i]) if (hi[i] - lo[i]) > 0 else 0.5
        # 20-bar realized vol
        if i >= 20:
            features[i, 27] = np.std(ret[i-19:i+1])
        # 5-bar realized vol
        if i >= 5:
            features[i, 28] = np.std(ret[i-4:i+1])
        # 5-bar avg return
        if i >= 5:
            features[i, 29] = np.mean(ret[i-4:i+1])
        # 20-bar avg return
        if i >= 20:
            features[i, 30] = np.mean(ret[i-19:i+1])
        features[i, 31] = ret[i] * features[i, 21]
        features[i, 32] = hour

    return features

# ── Model ────────────────────────────────────────────────────────────────────

class BTCPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.candles = []
        self.features = None
        self.train_count = 0
        self.predictions_since_train = 0

    def load_data(self):
        self.candles = fetch_all_candles(10000)
        if len(self.candles) < 100:
            raise RuntimeError(f"Only got {len(self.candles)} candles, need at least 100")

    def train(self):
        print("Training XGBoost model...")
        features = compute_features(self.candles)
        n = len(self.candles)
        # Target: next candle up
        target = np.zeros(n)
        for i in range(n - 1):
            target[i] = 1 if self.candles[i+1][2] > self.candles[i][2] else 0

        # Use rows 50..n-2 (need lookback and next candle for target)
        # Replace any remaining NaN with 0
        features_clean = np.nan_to_num(features, nan=0.0)
        valid = np.ones(n, dtype=bool)
        valid[-1] = False  # no target for last
        valid[:50] = False  # need lookback

        X = features_clean[valid]
        y = target[valid]
        print(f"  Training samples: {len(X)}, up ratio: {y.mean():.3f}")

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        # Replace any inf/nan from scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, use_label_encoder=False
        )
        self.model.fit(X_scaled, y)
        self.features = features
        self.train_count += 1
        self.predictions_since_train = 0
        print(f"  Model trained (#{self.train_count})")

    def predict_current(self):
        """Predict direction for next candle. Returns (prediction, probability)."""
        features = compute_features(self.candles)
        current = features[-1:]
        if np.any(np.isnan(current)):
            # Fill NaN with 0
            current = np.nan_to_num(current, 0)
        X = self.scaler.transform(current)
        prob = self.model.predict_proba(X)[0][1]  # prob of UP
        return prob

    def add_candle(self, candle):
        self.candles.append(candle)
        # Keep last 10500
        if len(self.candles) > 10500:
            self.candles = self.candles[-10000:]

# ── Main loop ────────────────────────────────────────────────────────────────

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

def main():
    print("=" * 60)
    print("Polymarket BTC Up/Down Paper Trading Bot")
    print("=" * 60)

    predictor = BTCPredictor()
    predictor.load_data()
    predictor.train()

    results = {
        "started": datetime.now(timezone.utc).isoformat(),
        "total_predictions": 0,
        "correct": 0,
        "wrong": 0,
        "skipped": 0,
        "win_rate": 0,
        "confident_predictions": 0,
        "confident_correct": 0,
        "confident_win_rate": 0,
        "pnl_if_real": 0.0,
        "trades": []
    }
    save_results(results)

    last_candle_time = predictor.candles[-1][0]
    pending_prediction = None  # (prediction, confidence, btc_price_start, market_info, time)

    print(f"\nBot started. Last candle: {datetime.fromtimestamp(last_candle_time, tz=timezone.utc)}")
    print("Waiting for next 5-min candle...\n")

    while True:
        try:
            time.sleep(30)

            # Fetch latest candles
            batch = fetch_kucoin_candles(end_at=int(time.time()), limit=10)
            if not batch:
                continue

            latest_time = batch[-1][0]
            if latest_time <= last_candle_time:
                continue

            # New candle(s) arrived
            new_candles = [c for c in batch if c[0] > last_candle_time]

            # If we have a pending prediction, verify it
            if pending_prediction and len(new_candles) >= 1:
                pred, conf, price_start, mkt_info, pred_time = pending_prediction
                # The verification candle is the first new one
                verify_candle = new_candles[0]
                price_end = verify_candle[2]
                actual = "UP" if price_end > price_start else "DOWN"
                correct = (pred == actual)
                result_str = "WIN" if correct else "LOSS"

                # PnL: if we bet $1 at 0.50 odds, win pays $1, lose costs $1
                pnl_change = 1.0 if correct else -1.0

                trade = {
                    "time": pred_time,
                    "market_slug": mkt_info,
                    "prediction": pred,
                    "confidence": round(conf, 4),
                    "actual": actual,
                    "result": result_str,
                    "btc_price_start": round(price_start, 2),
                    "btc_price_end": round(price_end, 2)
                }

                results["total_predictions"] += 1
                if correct:
                    results["correct"] += 1
                else:
                    results["wrong"] += 1
                results["pnl_if_real"] = round(results["pnl_if_real"] + pnl_change, 2)
                results["win_rate"] = round(results["correct"] / results["total_predictions"], 4)

                if conf >= 0.60 or conf <= 0.40:
                    results["confident_predictions"] += 1
                    if correct:
                        results["confident_correct"] += 1
                    results["confident_win_rate"] = round(
                        results["confident_correct"] / results["confident_predictions"], 4
                    ) if results["confident_predictions"] else 0

                results["trades"].append(trade)
                # Keep last 500 trades in file
                if len(results["trades"]) > 500:
                    results["trades"] = results["trades"][-500:]

                save_results(results)

                print(f"  ✓ RESULT: {result_str} | Pred: {pred} | Actual: {actual} | "
                      f"BTC: {price_start:.0f}→{price_end:.0f} | Conf: {conf:.3f}")
                print(f"  📊 Record: {results['correct']}W-{results['wrong']}L "
                      f"({results['win_rate']:.1%}) | PnL: ${results['pnl_if_real']}")
                pending_prediction = None

            # Add new candles to predictor
            for c in new_candles:
                predictor.add_candle(c)
            last_candle_time = predictor.candles[-1][0]

            # Retrain every 100 predictions
            predictor.predictions_since_train += 1
            if predictor.predictions_since_train >= 100:
                predictor.train()

            # Make prediction for next candle
            prob = predictor.predict_current()
            current_price = predictor.candles[-1][2]
            now_str = datetime.now(timezone.utc).isoformat()

            # Try to find Polymarket market
            poly_market = fetch_polymarket_btc()
            market_slug = "N/A"
            if poly_market:
                market_slug = poly_market.get("slug", poly_market.get("question", "found"))[:60]

            ts_label = datetime.fromtimestamp(last_candle_time, tz=timezone.utc).strftime("%H:%M")
            if prob > 0.60:
                prediction = "UP"
                confidence = prob
                pending_prediction = (prediction, confidence, current_price, market_slug, now_str)
                print(f"\n[{ts_label}] 🟢 PREDICT UP  | conf={prob:.3f} | BTC=${current_price:,.0f} | market={market_slug}")
            elif prob < 0.40:
                prediction = "DOWN"
                confidence = 1 - prob
                pending_prediction = (prediction, confidence, current_price, market_slug, now_str)
                print(f"\n[{ts_label}] 🔴 PREDICT DOWN | conf={1-prob:.3f} | BTC=${current_price:,.0f} | market={market_slug}")
            else:
                results["skipped"] += 1
                save_results(results)
                print(f"\n[{ts_label}] ⚪ SKIP (prob={prob:.3f}) | BTC=${current_price:,.0f}")

        except KeyboardInterrupt:
            print("\nShutting down...")
            save_results(results)
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
