"""
Verdict track record — stores the Buy / Hold / Sell ... decisions and tests them.

Two sources:
  * Backtest of the technical checklist (patterns.assess) since START: every trading day is
    replayed with only the bars up to that day (and SPY up to that day for the market regime),
    so there is no look-ahead. Past days never change, so each run only adds the new days.
  * The market brief's verdicts (checklist + news / theme / events): briefing.py --index appends
    briefs/verdicts.json to briefs/verdict_log.jsonl once per brief, with the price at that time.
    These start on 2026-10-23 (no history exists before that).

Scoring:
  * per verdict label: the return 5 and 21 trading days later, and the same vs SPY;
  * "follow the verdicts": long while the verdict is Buy / Buy on dip / Hold, in cash otherwise,
    switching at the next day's open; compared with buying and holding the ticker.

Usage:
    python verdicts.py              # update verdicts/backtest.json (run by the daily GitHub Action)
    python verdicts.py SYM ...      # also backtest other tickers
"""

import os, sys, json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

import patterns as P

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "verdicts", "backtest.json")
LOG = os.path.join(ROOT, "briefs", "verdict_log.jsonl")
START = "2026-01-02"
LONG = {"Buy", "Buy on dip", "Hold"}
# also tracked (the calibration ETFs), so the rarer verdicts (Sell, Wait, Take profit) have a record
EXTRA = ["IWM", "DIA", "SMH", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]
LABELS = P.VERDICTS
HORIZONS = {"1 week": 5, "1 month": 21}


# ── recording the brief's verdicts (called by briefing.py --index) ─────────────
def record_brief():
    """Append the latest briefs/verdicts.json to the log once per brief id."""
    f = os.path.join(ROOT, "briefs", "verdicts.json")
    try:
        v = json.load(open(f, encoding="utf-8"))
    except Exception:
        return 0
    bid = v.get("brief_id")
    seen = set()
    if os.path.exists(LOG):
        seen = {json.loads(l).get("brief_id") for l in open(LOG, encoding="utf-8") if l.strip()}
    if not bid or bid in seen:
        return 0
    n = 0
    with open(LOG, "a", encoding="utf-8") as fh:
        for sym, x in (v.get("tickers") or {}).items():
            price = None
            try:
                price = float(P.load_prices(sym, "1y", fill_today=True)["Close"].iloc[-1])
            except Exception:
                pass
            fh.write(json.dumps({"brief_id": bid, "time": v.get("updated"), "symbol": sym, "verdict": x.get("verdict"),
                                 "technical": x.get("technical"), "confidence": x.get("confidence"),
                                 "price": round(price, 2) if price else None}) + "\n")
            n += 1
    return n


# ── backtest ───────────────────────────────────────────────────────────────────
def _load():
    try:
        return json.load(open(OUT, encoding="utf-8"))
    except Exception:
        return {"start": START, "tickers": {}}


def replay(sym, prev_days, spy):
    full = P.load_prices(sym, "5y")
    have = {d[0] for d in prev_days}
    days = list(prev_days)
    now = datetime.now(ZoneInfo("America/New_York"))
    for d in full.index[full.index >= pd.Timestamp(START)]:
        ds = f"{d:%Y-%m-%d}"
        if ds in have or (ds == f"{now:%Y-%m-%d}" and (now.hour, now.minute) < (16, 30)):
            continue                                    # a day is stored only once its bar is final
        sl = full[full.index <= d]
        if len(sl) < 260:
            continue
        r = P.analyze(sym, "2y", full=sl, regime=P.market_regime(spy[spy.index <= d]))
        a = r.get("assessment") or {}
        if "verdict" not in a:
            continue
        days.append([ds, round(float(sl["Close"].iloc[-1]), 2), a["verdict"], a["good"], a["bad"], a["sell_alerts"], a.get("features")])
    days.sort(key=lambda x: x[0])
    # re-apply the current decision rules (verdict_params.json) to the stored features
    days = [d[:2] + [P.decide(d[6])[0] if len(d) > 6 and d[6] else d[2]] + d[3:] for d in days]
    return days, full


def score(days, full, spy):
    idx = {f"{d:%Y-%m-%d}": i for i, d in enumerate(full.index)}
    c, o = full["Close"].values, full["Open"].values
    spyc = spy.reindex(full.index).ffill().values
    by = {}
    for ds, _, v, *_rest in days:
        i = idx.get(ds)
        if i is None:
            continue
        for name, h in HORIZONS.items():
            if i + h < len(c):
                r = c[i + h] / c[i] - 1
                ex = r - (spyc[i + h] / spyc[i] - 1)
                b = by.setdefault(v, {}).setdefault(name, [])
                b.append((r, ex))
    stats = {}
    for v, hs in by.items():
        # "right": a Buy / Buy on dip / Hold that beat SPY afterwards, or a Wait / Warning / Take profit / Sell
        # that lagged SPY afterwards (stepping aside was the better choice)
        stats[v] = {name: {"n": len(x), "avg": round(float(np.mean([a for a, _ in x])) * 100, 2),
                           "vs_spy": round(float(np.mean([e for _, e in x])) * 100, 2),
                           "up_pct": round(100 * sum(a > 0 for a, _ in x) / len(x)),
                           "right_pct": round(100 * sum((e > 0) if v in LONG else (e < 0) for _, e in x) / len(x))} for name, x in hs.items()}
    # follow the verdicts: decide at the close, trade at the next open
    pos = {d[0]: (d[2] in LONG) for d in days}
    eq, bh, curve, trades, inpos = 1.0, 1.0, [], 0, False
    first = idx.get(days[0][0]) if days else None
    if first is not None:
        prev_in = False
        for i in range(first + 1, len(c)):
            want = pos.get(f"{full.index[i - 1]:%Y-%m-%d}", prev_in)        # decision made at yesterday's close
            if want and prev_in:
                eq *= c[i] / c[i - 1]
            elif want and not prev_in:
                eq *= c[i] / o[i]; trades += 1                              # buy at today's open
            elif prev_in and not want:
                eq *= o[i] / c[i - 1]                                        # sell at today's open
            bh *= c[i] / c[i - 1]
            prev_in = want
            curve.append([f"{full.index[i]:%Y-%m-%d}", round((eq - 1) * 100, 2), round((bh - 1) * 100, 2), int(want)])
        inpos = prev_in
    peak = dd = 0.0
    for _, e, *_ in curve:
        peak = max(peak, e); dd = min(dd, (100 + e) / (100 + peak) - 1)
    strat = {"return_pct": curve[-1][1] if curve else 0, "buy_hold_pct": curve[-1][2] if curve else 0,
             "entries": trades, "in_market_pct": round(100 * sum(x[3] for x in curve) / len(curve)) if curve else 0,
             "max_drawdown_pct": round(dd * 100, 1), "in_position_now": inpos, "curve": curve}
    return stats, strat


def score_log(spy):
    """Forward returns of the brief's recorded verdicts (price at the brief vs later closes)."""
    if not os.path.exists(LOG):
        return {"rows": [], "by_verdict": {}}
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
    cache, out, agg = {}, [], {}
    for r in rows:
        if not r.get("price"):
            continue
        try:
            full = cache.setdefault(r["symbol"], P.load_prices(r["symbol"], "1y"))
        except Exception:
            continue
        day = pd.Timestamp((r.get("time") or r["brief_id"])[:10])
        after = full[full.index > day]["Close"]
        res = dict(r)
        res["since_pct"] = round(float(full["Close"].iloc[-1] / r["price"] - 1) * 100, 2)
        spy_then = spy[spy.index <= day]
        if len(spy_then):
            res["since_vs_spy"] = round(res["since_pct"] - float(spy.iloc[-1] / spy_then.iloc[-1] - 1) * 100, 2)
        for name, h in HORIZONS.items():
            if len(after) >= h:
                res[name] = round(float(after.iloc[h - 1] / r["price"] - 1) * 100, 2)
                agg.setdefault(r["verdict"], {}).setdefault(name, []).append(res[name])
        out.append(res)
    by = {v: {k: {"n": len(x), "avg": round(float(np.mean(x)), 2)} for k, x in hs.items()} for v, hs in agg.items()}
    return {"rows": out[-60:], "by_verdict": by, "first": rows[0].get("time") if rows else None}


def run(extra=()):
    st = _load()
    syms = list(dict.fromkeys(list(P.DEFAULT_SYMBOLS) + EXTRA + [s.upper() for s in extra]))
    spy = P.load_prices("SPY", "5y")["Close"]
    allv = {}
    for sym in syms:
        try:
            prev = [d for d in st["tickers"].get(sym, {}).get("days", []) if len(d) > 6]   # older rows without features are replayed
            days, full = replay(sym, prev, spy)
            stats, strat = score(days, full, spy)
        except Exception as e:
            print(f"{sym}: ERROR {e}")
            continue
        st["tickers"][sym] = {"days": days, "stats": stats, "strategy": strat}
        for v, hs in stats.items():
            for name, s_ in hs.items():
                a = allv.setdefault(v, {}).setdefault(name, {"n": 0, "sum": 0.0, "sum_ex": 0.0, "up": 0.0, "right": 0.0})
                a["n"] += s_["n"]; a["sum"] += s_["avg"] * s_["n"]; a["sum_ex"] += s_["vs_spy"] * s_["n"]; a["up"] += s_["up_pct"] * s_["n"]
                a["right"] += s_["right_pct"] * s_["n"]
        print(f"{sym}: {len(days)} days · follow verdicts {strat['return_pct']:+.1f}% vs buy & hold {strat['buy_hold_pct']:+.1f}% "
              f"({strat['entries']} entries, in the market {strat['in_market_pct']}% of days, max drawdown {strat['max_drawdown_pct']}%)")
    st["by_verdict"] = {v: {name: {"n": a["n"], "avg": round(a["sum"] / a["n"], 2), "vs_spy": round(a["sum_ex"] / a["n"], 2),
                                   "up_pct": round(a["up"] / a["n"]), "right_pct": round(a["right"] / a["n"])} for name, a in hs.items()}
                        for v, hs in allv.items()}
    st["live"] = score_log(spy)
    st["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st["labels"] = LABELS
    st["watchlist"] = list(P.DEFAULT_SYMBOLS)
    st["params"] = P.PARAMS
    for t in st["tickers"].values():                      # keep the published file small: no features
        t["days"] = [d[:6] for d in t["days"]]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(st, open(OUT, "w", encoding="utf-8"), separators=(",", ":"))
    for v in LABELS:
        if v in st["by_verdict"]:
            m = st["by_verdict"][v]["1 month"] if "1 month" in st["by_verdict"][v] else st["by_verdict"][v]["1 week"]
            print(f"  {v:<12} next month avg {m['avg']:+.2f}% (vs SPY {m['vs_spy']:+.2f}pp, up {m['up_pct']}% of {m['n']})")
    return st


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(sys.argv[1:])
