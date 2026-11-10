"""
Rotation watch — "if SPY moves higher into year end, retail will buy the previous cycle leaders as
they consolidate, missing the rotation into the next group of winners."

For every sector / theme ETF of the rotation map (rotation.py), using price returns vs SPY:
  * previous leader  = top third of the universe by 12-month excess return
  * consolidating    = within 15% of its 52-week high, 30-day range at most 4x its daily ATR... and its
                       1-month excess return <= 0 or the rotation map has it Weakening / Lagging
  * next group       = NOT a previous leader, rotation map Improving or Leading with rising relative
                       momentum, 1-month excess return > 0 and price above its 50-day average
  * everything else  = previous leader still leading / no signal
Each ETF also gets the buy & sell checklist (patterns.assess): verdict, buy zone, stop, reward/risk —
"is it at a good point to invest now?".

Test (since 2012, monthly): the 3-month return vs SPY that followed each class, so the page shows
whether the "next group" idea has actually worked.

    python rotwatch.py        # writes rotation/watch.json (the daily GitHub Action runs it)
"""

import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

import patterns as P
from rotation import ETFS, _quadrant

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "rotation", "watch.json")
UNIVERSE = [e for e in ETFS if e[2] in ("sector", "theme")]
LONG = {"Buy", "Buy on dip", "Hold"}


def _prices():
    syms = [e[0] for e in UNIVERSE] + ["SPY"]
    full = yf.download(syms, start="2010-01-01", auto_adjust=True, progress=False)
    px, hi, lo = full["Close"], full["High"], full["Low"]
    px.index = px.index.tz_localize(None) if px.index.tz is not None else px.index
    hi.index, lo.index = px.index, px.index
    keep = px["SPY"].notna()
    return px[keep].ffill(limit=5), hi[keep].ffill(limit=5), lo[keep].ffill(limit=5)


def features(px, hi, lo):
    """Per-day feature frames (rows = dates, columns = ETFs)."""
    spy = px["SPY"]
    ex = lambda n: (px / px.shift(n) - 1).sub(spy / spy.shift(n) - 1, axis=0) * 100
    f = {"ex12": ex(252), "ex6": ex(126), "ex3": ex(63), "ex1": ex(21), "ex1w": ex(5)}
    f["rs"] = 0.5 * f["ex1"] + 0.3 * f["ex3"] + 0.2 * f["ex1w"]
    f["rm"] = f["ex1"] - f["ex1"].shift(21)
    f["from_high"] = (px / px.rolling(252, min_periods=120).max() - 1) * 100
    tr = pd.concat({c: pd.concat([hi[c] - lo[c], (hi[c] - px[c].shift()).abs(), (lo[c] - px[c].shift()).abs()], axis=1).max(axis=1)
                    for c in px.columns}, axis=1)
    atr = tr.rolling(14).mean()
    f["range30_atr"] = (px.rolling(30).max() - px.rolling(30).min()) / atr
    f["above50"] = px > px.rolling(50).mean()
    return f


def classify(f, i, syms):
    """Class per ETF on row i."""
    ex12 = f["ex12"].iloc[i][syms]
    if ex12.notna().sum() < 6:
        return {}
    top = ex12.rank(pct=True) >= 2 / 3
    out = {}
    for s in syms:
        if pd.isna(ex12[s]) or pd.isna(f["rs"].iloc[i][s]) or pd.isna(f["rm"].iloc[i][s]):
            continue
        q = _quadrant(f["rs"].iloc[i][s], f["rm"].iloc[i][s])
        if top[s]:
            consol = (f["from_high"].iloc[i][s] >= -15 and f["range30_atr"].iloc[i][s] <= 4.5
                      and (f["ex1"].iloc[i][s] <= 0 or q in ("Weakening", "Lagging")))
            out[s] = "Previous leader, consolidating" if consol else ("Previous leader, still leading" if q == "Leading" else "Previous leader, fading")
        elif q in ("Improving", "Leading") and f["rm"].iloc[i][s] > 0 and f["ex1"].iloc[i][s] > 0 and bool(f["above50"].iloc[i][s]):
            out[s] = "Next group candidate"
        else:
            out[s] = "No signal"
    return out


def backtest(px, f, syms, start="2012-01-01", h=63, only=None):
    """Monthly: 3-month excess return vs SPY after each class (only: a filter on the month's row index)."""
    spy = px["SPY"]
    fwd = (px.shift(-h) / px - 1).sub(spy.shift(-h) / spy - 1, axis=0) * 100
    idx = [i for i, d in enumerate(px.index) if d >= pd.Timestamp(start) and i + h < len(px)]
    months = sorted({(px.index[i].year, px.index[i].month): i for i in idx}.values())      # last trading day per month
    res = {}
    for i in months:
        if only and not only(i):
            continue
        for s, c in classify(f, i, syms).items():
            v = fwd.iloc[i][s]
            if not pd.isna(v):
                res.setdefault(c, []).append(float(v))
    return {c: {"n": len(v), "avg_vs_spy": round(float(np.mean(v)), 2), "beat_spy_pct": round(100 * float(np.mean([x > 0 for x in v]))),
                "median_vs_spy": round(float(np.median(v)), 2)} for c, v in sorted(res.items())}


def run():
    px, hi, lo = _prices()
    syms = [e[0] for e in UNIVERSE if e[0] in px.columns]
    f = features(px, hi, lo)
    i = len(px) - 1
    cls = classify(f, i, syms)
    spy = px["SPY"]
    regime = P.market_regime(spy)
    rows = []
    for sym, label, group in UNIVERSE:
        if sym not in cls:
            continue
        r = {"symbol": sym, "label": label, "group": group, "class": cls[sym],
             "quadrant": _quadrant(f["rs"].iloc[i][sym], f["rm"].iloc[i][sym]),
             **{k: round(float(f[k].iloc[i][sym]), 2) for k in ("ex12", "ex6", "ex3", "ex1", "rs", "rm", "from_high", "range30_atr")},
             "price": round(float(px[sym].iloc[-1]), 2)}
        try:
            a = P.analyze(sym, "2y", full=P.load_prices(sym, "5y", fill_today=True), regime=regime).get("assessment") or {}
            if "verdict" in a:
                lv, ft = a["levels"], a.get("features") or {}
                atr = abs((lv["price"] - lv["ema21"]) / ft["ext"]) if ft.get("ext") else lv["price"] * 0.015
                r["check"] = {"verdict": a["verdict"], "why": a["why"], "ema21": lv["ema21"], "sma50": lv["sma50"], "sma200": lv["sma200"],
                              "buy_zone": [round(max(lv["ema21"] - 1.5 * atr, lv["support"] or 0), 2), round(lv["ema21"] + 0.5 * atr, 2)],
                              "stop": lv["stop"], "target": lv["target"], "rr": lv["rr"], "sell_alerts": a["sell_alerts"]}
        except Exception as e:
            r["check"] = {"error": str(e)}
        c = r.get("check", {})
        good = r["class"] == "Next group candidate" and c.get("verdict") in ("Buy", "Buy on dip")
        r["entry"] = ("Good point" if c.get("verdict") == "Buy" else "Wait for the buy zone") if good else (
            "Not the rotation target" if r["class"].startswith("Previous leader") else "No setup yet")
        rows.append(r)
    out = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "asof": f"{px.index[-1]:%Y-%m-%d}",
           "spy_ytd_pct": round(float(spy.iloc[-1] / spy[spy.index.year == px.index[-1].year].iloc[0] - 1) * 100, 1),
           "regime": regime, "rows": rows, "test": backtest(px, f, syms),
           # the quote's case: SPY rising into year end (Oct-Dec month-ends with SPY above its 50-day average)
           "test_year_end": backtest(px, f, syms, only=lambda k: px.index[k].month >= 10 and spy.iloc[k] > spy.iloc[max(0, k - 49):k + 1].mean())}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    return out


def summary(d):
    by = {}
    for r in d["rows"]:
        by.setdefault(r["class"], []).append(r)
    L = [f"ROTATION WATCH (as of {d['asof']}; SPY YTD {d['spy_ytd_pct']:+}%)"]
    for c in ("Previous leader, consolidating", "Next group candidate", "Previous leader, still leading", "Previous leader, fading"):
        if by.get(c):
            L.append(f"  {c}: " + "; ".join(f"{r['symbol']} ({r['label']}) 12m {r['ex12']:+.0f} / 1m {r['ex1']:+.1f} vs SPY, {r['quadrant']}, "
                                             f"checklist {r.get('check', {}).get('verdict', '?')} -> {r['entry']}" for r in by[c]))
    t = d["test"]
    L.append("  test since 2012 (3 months later vs SPY): " + "; ".join(f"{c} {v['avg_vs_spy']:+}pp avg, beat SPY {v['beat_spy_pct']}% (n={v['n']})" for c, v in t.items()))
    t2 = d.get("test_year_end") or {}
    if t2:
        L.append("  same test, only Oct-Dec with SPY above its 50-day (the quote's case): "
                 + "; ".join(f"{c} {v['avg_vs_spy']:+}pp, beat SPY {v['beat_spy_pct']}% (n={v['n']})" for c, v in t2.items()))
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(summary(run()))
