"""Diagnosis: CNN-only vs meta-label signal quality comparison."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from config.loader import load_config
config = load_config()

from backtest.runner import _load_bars, _load_predictions
bars_by_ticker = _load_bars(config)
preds_by_ticker = _load_predictions(config, bars_by_ticker)

from data.meta_features import build_meta_features

TP = 0.005
SL = 0.003
MAX_BARS = 48


def forward_test(closes, highs, lows, probs, threshold, direction):
    n = len(closes)
    wins = losses = timeouts = 0
    pnls = []
    for i in range(n - MAX_BARS):
        if np.isnan(probs[i]) or probs[i] <= threshold:
            continue
        entry = closes[i]
        hit_tp = hit_sl = False
        for j in range(1, min(MAX_BARS + 1, n - i)):
            if direction == "SHORT":
                if lows[i + j] <= entry * (1 - TP):
                    hit_tp = True; break
                if highs[i + j] >= entry * (1 + SL):
                    hit_sl = True; break
            else:
                if highs[i + j] >= entry * (1 + TP):
                    hit_tp = True; break
                if lows[i + j] <= entry * (1 - SL):
                    hit_sl = True; break
        if hit_tp:
            wins += 1; pnls.append(TP)
        elif hit_sl:
            losses += 1; pnls.append(-SL)
        else:
            timeouts += 1
            ep = closes[min(i + MAX_BARS, n - 1)]
            pnls.append((entry - ep) / entry if direction == "SHORT" else (ep - entry) / entry)
    return {"total": wins + losses + timeouts, "wins": wins, "losses": losses,
            "timeouts": timeouts, "pnls": pnls}


def report(label, r):
    t = r["total"]
    if t == 0:
        print(f"  {label}: 0 signals")
        return
    pnls = r["pnls"]
    wr = r["wins"] / t * 100
    avg = np.mean(pnls) * 100
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gw / gl if gl > 0 else float('inf')
    print(f"  {label}: signals={t}, wins={r['wins']}, losses={r['losses']}, "
          f"timeout={r['timeouts']}, WR={wr:.1f}%, avg_pnl={avg:+.3f}%, PF={pf:.2f}")


# ── Build CNN probabilities per ticker ──
print("Building CNN probabilities for each ticker...\n")

cnn_bottom_by_ticker = {}
cnn_top_by_ticker = {}

for ticker in sorted(bars_by_ticker.keys()):
    for task, store in [("bottom", cnn_bottom_by_ticker), ("top", cnn_top_by_ticker)]:
        try:
            df = build_meta_features(ticker, config, cnn_task=task)
            if df is not None and len(df) > 0:
                cnn_col = df.iloc[:, -1]  # cnn_prob is the last column
                non_nan = cnn_col.dropna()
                if len(non_nan) > 0:
                    store[ticker] = cnn_col
                    print(f"  {ticker} cnn_{task}: {len(non_nan)} values, "
                          f"min={non_nan.min():.4f}, max={non_nan.max():.4f}, "
                          f"mean={non_nan.mean():.4f}")
        except Exception as e:
            print(f"  {ticker} {task} failed: {e}")

# ── CNN probability distributions ──
print("\n" + "=" * 60)
print("CNN PROBABILITY DISTRIBUTIONS (all tickers)")
print("=" * 60)

for label, store in [("cnn_bottom_prob", cnn_bottom_by_ticker),
                     ("cnn_top_prob", cnn_top_by_ticker)]:
    all_vals = [store[t].dropna() for t in sorted(store.keys())]
    if not all_vals:
        print(f"\n{label}: NO DATA")
        continue
    s = pd.concat(all_vals)
    print(f"\n{label}: n={len(s)}")
    print(f"  min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        n_above = (s > thresh).sum()
        print(f"  > {thresh}: {n_above} ({n_above/len(s)*100:.2f}%)")

# ── CNN-only signal quality ──
print("\n" + "=" * 60)
print("CNN-ONLY SIGNAL QUALITY (TP=0.5%, SL=0.3%, max_bars=48)")
print("=" * 60)

for thresh in [0.50, 0.55, 0.60]:
    print(f"\n--- CNN threshold = {thresh} ---")
    for direction, pred_label, store in [
        ("BUY", "cnn_bottom", cnn_bottom_by_ticker),
        ("SHORT", "cnn_top", cnn_top_by_ticker),
    ]:
        agg = {"total": 0, "wins": 0, "losses": 0, "timeouts": 0, "pnls": []}
        for ticker in sorted(bars_by_ticker.keys()):
            if ticker not in store:
                continue
            bar_df = bars_by_ticker[ticker]
            aligned = store[ticker].reindex(bar_df.index, method='ffill')
            r = forward_test(bar_df['Close'].values, bar_df['High'].values,
                             bar_df['Low'].values, aligned.values, thresh, direction)
            for k in ["total", "wins", "losses", "timeouts"]:
                agg[k] += r[k]
            agg["pnls"].extend(r["pnls"])
        report(f"{direction:5s} ({pred_label})", agg)

# ── Meta-label comparison ──
print("\n" + "=" * 60)
print("META-LABEL SIGNAL QUALITY (same TP/SL/max_bars for comparison)")
print("=" * 60)

for thresh in [0.50, 0.55, 0.60]:
    print(f"\n--- Meta threshold = {thresh} ---")
    for direction, col in [("BUY", "tp_bottom"), ("SHORT", "tp_top")]:
        agg = {"total": 0, "wins": 0, "losses": 0, "timeouts": 0, "pnls": []}
        for ticker in sorted(bars_by_ticker.keys()):
            pred_df = preds_by_ticker.get(ticker)
            if pred_df is None or col not in pred_df.columns:
                continue
            bar_df = bars_by_ticker[ticker]
            aligned = pred_df[col].reindex(bar_df.index, method='ffill')
            r = forward_test(bar_df['Close'].values, bar_df['High'].values,
                             bar_df['Low'].values, aligned.values, thresh, direction)
            for k in ["total", "wins", "losses", "timeouts"]:
                agg[k] += r[k]
            agg["pnls"].extend(r["pnls"])
        report(f"{direction:5s} ({col})", agg)
