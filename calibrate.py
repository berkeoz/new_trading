"""
Calibrate the checklist's decision rules (patterns.decide) on 2021-2025 and test them on 2026.

1. Replay every trading day from 2021 for a set of liquid ETFs with only the bars up to that
   day (and SPY up to that day for the market regime), and keep each day's checklist features.
   This is the slow part; it is cached in data/calib_features.json.
2. Try every combination in GRID on the training years only. Strategy: in the market while
   the verdict is Buy / Buy on dip / Hold, in cash otherwise, switching at the next open.
   Objective: the average Sharpe ratio across the tickers, with the market exposure kept
   at 35% or more (so "always in cash" cannot win).
3. Test the winner and the original rules on 2026, which the search never saw, and write
   verdict_params.json (used by patterns.decide) and verdicts/calibration.json (shown on the site).

Usage:  python calibrate.py
"""

import os, sys, json, itertools, time
from datetime import datetime
import numpy as np
import pandas as pd

import patterns as P

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "calib_features.json")
UNIVERSE = ["QQQ", "SPY", "SOXX", "IWM", "DIA", "SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
TRAIN = ("2021-01-04", "2025-12-31")
TEST = ("2026-01-02", "2099-12-31")
LONG = {"Buy", "Buy on dip", "Hold"}
GRID = {"trend": ["stack", "sma200"], "down": ["stack", "sma200"], "stretch_rsi": [75, 80, 101],
        "stretch_ext": [3.5, 5.0], "tp_rsi": [78, 101], "warn": ["ema21_macd", "sma50", "none"],
        "mixed_hold": [False, True], "alerts_warn": [3, 5, 99]}


def features():
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))
    spy = P.load_prices("SPY", "max")["Close"]
    for sym in UNIVERSE:
        full = P.load_prices(sym, "max")
        rows = {r[0]: r for r in cache.get(sym, [])}
        t0, n0 = time.time(), len(rows)
        for d in full.index[full.index >= pd.Timestamp(TRAIN[0])]:
            ds = f"{d:%Y-%m-%d}"
            if ds in rows:
                continue
            sl = full[full.index <= d].iloc[-900:]
            r = P.analyze(sym, "2y", full=sl, regime=P.market_regime(spy[spy.index <= d].iloc[-400:]))
            a = r.get("assessment") or {}
            if "features" in a:
                rows[ds] = [ds, a["features"]]
        cache[sym] = sorted(rows.values(), key=lambda x: x[0])
        print(f"{sym}: {len(rows)} days ({len(rows) - n0} new, {time.time() - t0:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    return cache


def strat(dates, verdicts, o, c):
    """Daily returns of following the verdicts (decide at the close, trade at the next open)."""
    out, prev = [], False
    for i in range(1, len(dates)):
        want = verdicts[i - 1] in LONG
        if want and prev:
            r = c[i] / c[i - 1] - 1
        elif want:
            r = c[i] / o[i] - 1
        elif prev:
            r = o[i] / c[i - 1] - 1
        else:
            r = 0.0
        out.append(r)
        prev = want
    return np.array(out)


def stats(r, inm=None):
    if len(r) == 0:
        return {}
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sd = r.std()
    return {"return_pct": round(float(eq[-1] - 1) * 100, 1), "sharpe": round(float(r.mean() / sd * np.sqrt(252)), 2) if sd > 0 else 0.0,
            "max_dd_pct": round(dd * 100, 1), **({"in_market_pct": round(inm * 100)} if inm is not None else {})}


def prepare(cache, period):
    data = {}
    for sym in UNIVERSE:
        full = P.load_prices(sym, "max")
        rows = [r for r in cache.get(sym, []) if period[0] <= r[0] <= period[1]]
        if len(rows) < 30:
            continue
        idx = pd.DatetimeIndex([r[0] for r in rows])
        px = full.reindex(idx)
        data[sym] = {"dates": [r[0] for r in rows], "f": [r[1] for r in rows],
                     "o": px["Open"].values, "c": px["Close"].values}
    return data


def evaluate(data, params, detail=False):
    res, sharpes, inms = {}, [], []
    fwd = {}
    for sym, d in data.items():
        v = [P.decide(f, params)[0] for f in d["f"]]
        r = strat(d["dates"], v, d["o"], d["c"])
        inm = sum(x in LONG for x in v[:-1]) / max(len(v) - 1, 1)
        s = stats(r, inm)
        sharpes.append(s["sharpe"]); inms.append(inm)
        if detail:
            bh = d["c"][1:] / d["c"][:-1] - 1
            res[sym] = {"strategy": s, "buy_hold": stats(bh)}
            c = d["c"]
            for i, x in enumerate(v):
                if i + 21 < len(c):
                    fwd.setdefault(x, []).append(c[i + 21] / c[i] - 1)
    out = {"avg_sharpe": round(float(np.mean(sharpes)), 3), "avg_in_market_pct": round(float(np.mean(inms)) * 100)}
    if detail:
        out["tickers"] = res
        out["avg_return_pct"] = round(float(np.mean([x["strategy"]["return_pct"] for x in res.values()])), 1)
        out["avg_buy_hold_pct"] = round(float(np.mean([x["buy_hold"]["return_pct"] for x in res.values()])), 1)
        out["avg_buy_hold_sharpe"] = round(float(np.mean([x["buy_hold"]["sharpe"] for x in res.values()])), 2)
        out["avg_max_dd_pct"] = round(float(np.mean([x["strategy"]["max_dd_pct"] for x in res.values()])), 1)
        out["avg_buy_hold_dd_pct"] = round(float(np.mean([x["buy_hold"]["max_dd_pct"] for x in res.values()])), 1)
        out["next_month_by_verdict"] = {k: {"n": len(x), "avg_pct": round(float(np.mean(x)) * 100, 2),
                                             "up_pct": round(100 * float(np.mean([y > 0 for y in x])))} for k, x in fwd.items()}
    return out


def main():
    cache = features()
    train, test = prepare(cache, TRAIN), prepare(cache, TEST)
    keys = list(GRID)
    results = []
    t0 = time.time()
    for combo in itertools.product(*GRID.values()):
        p = {**P.DEFAULT_PARAMS, **dict(zip(keys, combo))}
        e = evaluate(train, p)
        if e["avg_in_market_pct"] >= 35:
            results.append((e["avg_sharpe"], e["avg_in_market_pct"], p))
    results.sort(key=lambda x: -x[0])
    print(f"grid: {len(results)} valid combinations in {time.time() - t0:.0f}s")
    best = results[0][2]
    rep = {"trained_on": f"{TRAIN[0]} to {TRAIN[1]}", "tested_on": f"{TEST[0]} to latest", "universe": UNIVERSE,
           "objective": "average Sharpe of following the verdicts (in the market on Buy / Buy on dip / Hold), exposure >= 35%",
           "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "params": best, "default_params": P.DEFAULT_PARAMS,
           "top5": [{"sharpe": s, "in_market_pct": m, "params": {k: p[k] for k in keys}} for s, m, p in results[:5]]}
    for name, p in (("default", P.DEFAULT_PARAMS), ("tuned", best)):
        rep[name + "_train"] = evaluate(train, p, True)
        rep[name + "_test"] = evaluate(test, p, True)
    json.dump({"params": best, "trained_on": rep["trained_on"], "updated": rep["updated"]},
              open(os.path.join(ROOT, "verdict_params.json"), "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.join(ROOT, "verdicts"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "verdicts", "calibration.json"), "w", encoding="utf-8"), indent=1)
    print("best params:", {k: best[k] for k in keys})
    for name in ("default", "tuned"):
        for per in ("train", "test"):
            e = rep[f"{name}_{per}"]
            print(f"{name:<8}{per:<6} Sharpe {e['avg_sharpe']:.2f} (buy & hold {e['avg_buy_hold_sharpe']:.2f}) · return {e['avg_return_pct']:+.1f}% "
                  f"(buy & hold {e['avg_buy_hold_pct']:+.1f}%) · max DD {e['avg_max_dd_pct']}% (b&h {e['avg_buy_hold_dd_pct']}%) · in market {e['avg_in_market_pct']}%")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
