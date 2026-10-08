"""
Market brief data collector — the numbers behind the 9:15 / 15:45 ET briefs.

For each ticker in watchlist.txt (also shown on the formations page) it computes the daily
and 4H trend state (EMA9/EMA21/SMA50/SMA200 score, RSI14, MACD 12/26/9), the
nearest supports / resistances (pivot levels, active trendlines and channels,
key moving averages, anchored VWAPs) and open chart patterns. It also snapshots
the cross-asset backdrop: index futures, VIX, Treasury yields, dollar, oil,
gold and crypto.

The scheduled routine runs this, adds news / calendar research, and writes the
narrative brief next to the JSON snapshot in briefs/.

Usage:
    python briefing.py                       # picks the slot from the clock: 9:15 pre,
                                             # 9:45 open, 15:45 preclose, 16:15 close,
                                             # anything else = on-demand ("now")
    python briefing.py --session close       # force a specific kind of brief
    python briefing.py --index               # rebuild briefs/index.json
"""

import os, sys, json, glob, argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import patterns as P

ET = ZoneInfo("America/New_York")
ROOT = os.path.dirname(os.path.abspath(__file__))
BRIEF_DIR = os.path.join(ROOT, "briefs")
TICKERS = P.DEFAULT_SYMBOLS
# Scheduled slots (ET). One routine fires at 9:15, 9:45, 15:45 and 16:15; a single
# cron line also fires at 15:15 and 16:45, which are skipped. Any other time is an
# on-demand run ("now").
SESSIONS = {
    "pre":      ((9, 15),  "Pre-market brief"),
    "open":     ((9, 45),  "Opening brief"),
    "preclose": ((15, 45), "Pre-close brief"),
    "close":    ((16, 15), "Closing brief"),
    "now":      (None,     "Market brief"),
}
SKIP_SLOTS = [(15, 15), (16, 45)]
WINDOW = timedelta(minutes=12)

CONTEXT = [  # symbol, label, group, kind ("yield" values are in %, changes in basis points)
    ("ES=F",     "S&P 500 futures",        "Futures",    "price"),
    ("NQ=F",     "Nasdaq 100 futures",     "Futures",    "price"),
    ("^VIX",     "VIX",                    "Volatility", "level"),
    ("^VIX9D",   "VIX 9-day",              "Volatility", "level"),
    ("^IRX",     "US 13-week T-bill",      "Rates",      "yield"),
    ("^FVX",     "US 5Y yield",            "Rates",      "yield"),
    ("^TNX",     "US 10Y yield",           "Rates",      "yield"),
    ("^TYX",     "US 30Y yield",           "Rates",      "yield"),
    ("TLT",      "20+Y Treasury ETF",      "Rates",      "price"),
    ("HYG",      "High-yield bond ETF",    "Credit",     "price"),
    ("DX-Y.NYB", "US dollar index",        "FX",         "price"),
    ("CL=F",     "WTI crude oil",          "Commodities", "price"),
    ("BZ=F",     "Brent crude oil",        "Commodities", "price"),
    ("GC=F",     "Gold",                   "Commodities", "price"),
    ("BTC-USD",  "Bitcoin",                "Crypto",     "price"),
    ("ETH-USD",  "Ether",                  "Crypto",     "price"),
]
MAS = [("EMA", 9), ("EMA", 21), ("SMA", 50), ("SMA", 200)]


# ── Indicators (same rules as the formations page) ─────────────────────────────
def _ma(c, kind, n):
    return c.ewm(span=n, adjust=False).mean() if kind == "EMA" else c.rolling(n).mean()


def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)


def _wma(c, n):
    w = np.arange(1, n + 1)
    return c.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hma(c, n):
    return _wma(2 * _wma(c, int(round(n / 2 + 1e-9))) - _wma(c, n), int(round(n ** 0.5)))


def _verdict(sc, prev):
    d = 0 if prev is None else sc - prev
    if sc >= 0.5:  return "Bullish, weakening" if d <= -0.4 else "Bullish"
    if sc <= -0.5: return "Bearish, improving" if d >= 0.4 else "Bearish"
    if d >= 0.3:   return "Turning bullish"
    if d <= -0.3:  return "Turning bearish"
    return "Neutral"


def trend_state(close, label=str):
    """MA score / verdict, MA table, RSI and MACD for the last bar of `close`
    (integer-indexed; `label(i)` gives the bar's time for position i)."""
    mas = {f"{k}{n}": _ma(close, k, n) for k, n in MAS}
    names = list(mas)

    def score(i):
        pts = tot = 0
        for k, name in enumerate(names):
            v = mas[name].iloc[i]
            if pd.isna(v): continue
            pts += 1 if close.iloc[i] > v else -1; tot += 1
            pv = mas[name].iloc[i - 5]
            if not pd.isna(pv): pts += 1 if v > pv else -1; tot += 1
            if k + 1 < len(names):
                nv = mas[names[k + 1]].iloc[i]
                if not pd.isna(nv): pts += 1 if v > nv else -1; tot += 1
        return pts / tot if tot else None

    sc, prev = score(-1), score(-6)
    # WMA20 / HMA55: shown, not scored (HMA is read by its turns)
    extra = {}
    for name, ser in (("WMA20", _wma(close, 20)), ("HMA55", _hma(close, 55))):
        if pd.isna(ser.iloc[-1]) or pd.isna(ser.iloc[-6]):
            continue
        extra[name] = {"value": round(float(ser.iloc[-1]), 2),
                       "price_vs_pct": round((float(close.iloc[-1]) / float(ser.iloc[-1]) - 1) * 100, 2),
                       "slope_5_bars_pct": round((float(ser.iloc[-1]) / float(ser.iloc[-6]) - 1) * 100, 2)}
        if name == "HMA55":
            d = np.sign(ser.diff()).dropna()
            ch = d[d.diff().fillna(0) != 0]
            if not ch.empty:
                extra[name]["last_turn"] = {"direction": "up" if ch.iloc[-1] > 0 else "down",
                                            "when": label(int(ch.index[-1])),
                                            "bars_ago": int(len(close) - 1 - int(ch.index[-1]))}
    c = float(close.iloc[-1])
    rsi = _rsi(close)
    m = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    hist = m - sig
    cross = np.sign(m - sig).diff().fillna(0)
    last_x = cross[cross != 0]
    return {
        "score": None if sc is None else round(sc, 2),
        "score_5_bars_ago": None if prev is None else round(prev, 2),
        "verdict": None if sc is None else _verdict(sc, prev),
        "mas": {n: {"value": round(float(mas[n].iloc[-1]), 2),
                    "price_vs_pct": round((c / float(mas[n].iloc[-1]) - 1) * 100, 2),
                    "slope_5_bars_pct": round((float(mas[n].iloc[-1]) / float(mas[n].iloc[-6]) - 1) * 100, 2)}
                for n in names if not pd.isna(mas[n].iloc[-1])},
        "not_scored": extra,
        "rsi14": round(float(rsi.iloc[-1]), 1),
        "rsi14_5_bars_ago": round(float(rsi.iloc[-6]), 1),
        "macd": {"line": round(float(m.iloc[-1]), 3), "signal": round(float(sig.iloc[-1]), 3),
                 "hist": round(float(hist.iloc[-1]), 3), "hist_prev": round(float(hist.iloc[-2]), 3),
                 "last_cross": None if last_x.empty else {
                     "when": label(int(last_x.index[-1])), "type": "bullish" if last_x.iloc[-1] > 0 else "bearish",
                     "bars_ago": int(len(close) - 1 - int(last_x.index[-1]))}},
    }


def _avwap(df, i0):
    a = df.iloc[i0:]
    tp = (a["High"] + a["Low"] + a["Close"]) / 3
    v = a["Volume"].astype(float)
    return float((tp * v).sum() / v.sum()) if v.sum() > 0 else None


def _frame(r):
    """Rebuild a DataFrame (with warm-up closes) from a patterns.analyze() result."""
    o = r["ohlc"]
    idx = pd.to_datetime(o["x"])
    df = pd.DataFrame({"Open": o["open"], "High": o["high"], "Low": o["low"],
                       "Close": o["close"], "Volume": o["volume"]}, index=idx)
    close = pd.concat([pd.Series(r["warm"]), df["Close"].reset_index(drop=True)], ignore_index=True)
    nw = len(r["warm"])
    label = lambda i: o["x"][i - nw] if i >= nw else "before chart window"
    return df, close, label


def key_levels(r, df, state, price):
    """Nearest supports below and resistances above the current price."""
    lv = []
    for l in r["levels"]:
        lv.append((l["price"], f"horizontal level ({l['touches']} touches)"))
    for p in r["patterns"]:
        if not p["open"]:
            continue
        if p["cat"] == "trendline" and p["status"] == "active":
            lv.append((p["lines"][0][3], p["type"].lower()))
        elif p["cat"] == "range" and p["lines"]:
            lv.append((p["lines"][0][3], "consolidation range top"))
            lv.append((p["lines"][1][3], "consolidation range bottom"))
        elif p["cat"] in ("channel", "triangle") and p["lines"]:
            for L, side in zip(p["lines"], ("lower", "upper") if len(p["lines"]) == 2 else ("line",)):
                lv.append((L[3], f"{p['type'].lower()} {side} line"))
        elif p["cat"] == "reversal":
            lv.append((p["lines"][0][3], f"{p['type'].lower()} neckline"))
    for n, m in state["mas"].items():
        if n in ("EMA21", "SMA50", "SMA200"):
            lv.append((m["value"], n))
    xs = r["ohlc"]["x"]
    for kind, what in (("L", "last swing low"), ("H", "last swing high")):
        pv = [q for q in r["pivots"] if q["kind"] == kind and q["confirmed"]]
        if pv and pv[-1]["date"] in xs:
            i0 = xs.index(pv[-1]["date"])
            v = _avwap(df, i0) if len(df) - i0 >= 3 else None
            if v: lv.append((v, f"anchored VWAP from {what} ({pv[-1]['date']})"))
    look = df.tail(126)   # ~6 months of daily bars
    for i0, what in ((df.index.get_loc(look["Low"].idxmin()), "anchored VWAP from 6-month low"),
                     (df.index.get_loc(look["High"].idxmax()), "anchored VWAP from 6-month high")):
        if len(df) - i0 < 10:
            continue              # anchor too recent to mean anything
        v = _avwap(df, i0)
        if v: lv.append((v, f"{what} ({P.ts(df.index[i0])})"))
    out = {"supports": [], "resistances": []}
    for v, what in sorted(lv, key=lambda x: abs(x[0] - price)):
        d = (v / price - 1) * 100
        if abs(d) > 12: continue
        out["supports" if v < price else "resistances"].append(
            {"price": round(v, 2), "what": what, "distance_pct": round(d, 2)})
    out["supports"] = out["supports"][:6]
    out["resistances"] = out["resistances"][:6]
    return out


def _live_price(sym):
    """Latest trade including pre/post-market (5-minute bars)."""
    try:
        h = yf.Ticker(sym).history(period="2d", interval="5m", prepost=True)
        if h.empty: return None
        t = h.index[-1].tz_convert(ET)
        return {"price": round(float(h["Close"].iloc[-1]), 2), "time_et": t.strftime("%Y-%m-%d %H:%M")}
    except Exception:
        return None


def _today_5m(sym):
    try:
        h = yf.Ticker(sym).history(period="1d", interval="5m")
        h.index = h.index.tz_convert(ET).tz_localize(None)
        return h
    except Exception:
        return pd.DataFrame()


def _earnings(sym):
    """Next earnings date for stocks (ETFs have none)."""
    try:
        cal = yf.Ticker(sym).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            return str(dates[0])
    except Exception:
        pass
    return None


def ticker_snapshot(sym, spy=None):
    d = P.analyze(sym, "2y", fill_today=True)
    df, close, label = _frame(d)
    st = trend_state(close, label)
    price = d["price"]
    out = {
        "symbol": sym, "last_close": price, "last_close_date": d["asof"],
        "prev_close": round(float(df["Close"].iloc[-2]), 2),
        "change_1d_pct": round((price / float(df["Close"].iloc[-2]) - 1) * 100, 2),
        "change_5d_pct": round((price / float(df["Close"].iloc[-6]) - 1) * 100, 2),
        "change_1m_pct": round((price / float(df["Close"].iloc[-22]) - 1) * 100, 2),
        "high_52w": round(float(df["High"].tail(252).max()), 2),
        "low_52w": round(float(df["Low"].tail(252).min()), 2),
        "live": _live_price(sym),
        "daily": st,
        "levels": key_levels(d, df, st, price),
        "open_patterns": [{k: p[k] for k in ("type", "bias", "start", "end", "status", "target", "note")}
                          for p in d["patterns"] if p["open"]][:8],
        "recent_breaks": [{k: p[k] for k in ("type", "bias", "status", "target")}
                          for p in d["patterns"] if not p["open"] and
                          (pd.Timestamp(d["asof"]) - pd.Timestamp(p["status"].split()[-1][:10])).days <= 15
                          if p["status"].split()[-1][:4].isdigit()][:6],
    }
    if spy is not None and sym != "SPY":   # relative strength vs the S&P 500
        out["vs_spy_pct"] = {k: round(out[f"change_{k}_pct"] - spy[f"change_{k}_pct"], 2) for k in ("1d", "5d", "1m")}
    out["next_earnings"] = _earnings(sym)
    t5 = _today_5m(sym)
    if not t5.empty:
        day_open = float(t5["Open"].iloc[0])
        first = t5[t5.index < t5.index[0] + pd.Timedelta(minutes=15)]
        ref_prev = float(df["Close"].iloc[-2]) if P.ts(df.index[-1])[:10] == P.ts(t5.index[0])[:10] else price
        out["today"] = {"date": P.ts(t5.index[0])[:10], "open": round(day_open, 2),
                        "gap_pct": round((day_open / ref_prev - 1) * 100, 2),
                        "opening_range_15m": [round(float(first["Low"].min()), 2), round(float(first["High"].max()), 2)],
                        "high": round(float(t5["High"].max()), 2), "low": round(float(t5["Low"].min()), 2),
                        "last": round(float(t5["Close"].iloc[-1]), 2), "last_bar_et": P.ts(t5.index[-1])}
    try:   # 4H view and today's session VWAP
        h4 = P.analyze(sym, "6mo", interval="4h")
        _, c4, l4 = _frame(h4)
        out["h4"] = trend_state(c4, l4)
        out["h4"]["asof"] = h4["asof"]
        h1 = P.load_prices(sym, None, False, "1h")
        today = h1[h1.index.normalize() == h1.index[-1].normalize()]
        vw = _avwap(today, 0)
        out["session"] = {"date": P.ts(today.index[0])[:10], "open": round(float(today["Open"].iloc[0]), 2),
                          "high": round(float(today["High"].max()), 2), "low": round(float(today["Low"].min()), 2),
                          "last": round(float(today["Close"].iloc[-1]), 2),
                          "vwap": None if vw is None else round(vw, 2)}
    except Exception as e:
        out["h4_error"] = str(e)
    return out


def context_snapshot():
    out = []
    for sym, label, group, kind in CONTEXT:
        try:
            h = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True)["Close"].dropna()
            if len(h) < 23: raise ValueError("not enough data")
            last = float(h.iloc[-1])
            chg = (lambda n: round((last - float(h.iloc[-1 - n])) * 100, 1)) if kind == "yield" else \
                  (lambda n: round((last / float(h.iloc[-1 - n]) - 1) * 100, 2))
            out.append({"symbol": sym, "label": label, "group": group, "kind": kind,
                        "last": round(last, 3 if kind == "yield" else 2), "date": str(h.index[-1].date()),
                        "chg_1d": chg(1), "chg_5d": chg(5), "chg_1m": chg(21),
                        "unit": "bp" if kind == "yield" else "%",
                        "range_3m": [round(float(h.min()), 2), round(float(h.max()), 2)]})
        except Exception as e:
            out.append({"symbol": sym, "label": label, "group": group, "error": str(e)})
    return out


# ── Output ─────────────────────────────────────────────────────────────────────
def rebuild_index():
    items = []
    for f in sorted(glob.glob(os.path.join(BRIEF_DIR, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name == "index" or not os.path.exists(f[:-5] + ".md"):
            continue
        try:
            meta = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        items.append({"id": name, "session": meta.get("session"), "title": meta.get("title"),
                      "tickers": [t.get("symbol") for t in meta.get("tickers", [])],
                      "generated_et": meta.get("generated_et")})
    items.sort(key=lambda x: (x.get("generated_et") or "", x["id"]), reverse=True)
    with open(os.path.join(BRIEF_DIR, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"briefs": items}, fh, indent=1)
    print(f"index: {len(items)} briefs")


def summary_text(snap):
    """Compact plain-text view for the agent writing the brief."""
    L = [f"{snap['title']} — data as of {snap['generated_et']} ET"]
    L.append("\nCROSS-ASSET")
    for c in snap["context"]:
        if "error" in c:
            L.append(f"  {c['label']}: n/a ({c['error']})"); continue
        L.append(f"  {c['label']:<22} {c['last']:>10}  1d {c['chg_1d']:+}{c['unit']}  5d {c['chg_5d']:+}{c['unit']}"
                 f"  1m {c['chg_1m']:+}{c['unit']}  (3m range {c['range_3m'][0]}–{c['range_3m'][1]}, {c['date']})")
    for t in snap["tickers"]:
        if "error" in t:
            L.append(f"\n{t['symbol']}: ERROR {t['error']}"); continue
        d, h4 = t["daily"], t.get("h4", {})
        live = f", live {t['live']['price']} at {t['live']['time_et']} ET" if t.get("live") else ""
        L.append(f"\n{t['symbol']}  close {t['last_close']} ({t['last_close_date']}){live}"
                 f"  1d {t['change_1d_pct']:+}%  5d {t['change_5d_pct']:+}%  1m {t['change_1m_pct']:+}%"
                 f"  52w {t['low_52w']}–{t['high_52w']}")
        L.append(f"  daily: {d['verdict']} (score {d['score']}, 5 bars ago {d['score_5_bars_ago']}), "
                 f"RSI {d['rsi14']} (was {d['rsi14_5_bars_ago']}), MACD hist {d['macd']['hist']} (prev {d['macd']['hist_prev']}), "
                 f"last MACD cross {d['macd']['last_cross']}")
        L.append("  daily MAs: " + ", ".join(f"{n} {m['value']} (price {m['price_vs_pct']:+}%, slope {m['slope_5_bars_pct']:+}%)"
                                             for n, m in d["mas"].items()))
        if d.get("not_scored"):
            L.append("  not scored: " + ", ".join(
                f"{n} {m['value']} (price {m['price_vs_pct']:+}%, slope {m['slope_5_bars_pct']:+}%"
                + (f", turned {m['last_turn']['direction']} {m['last_turn']['when']}, {m['last_turn']['bars_ago']} bars ago" if m.get("last_turn") else "") + ")"
                for n, m in d["not_scored"].items()))
        if h4:
            L.append(f"  4H: {h4.get('verdict')} (score {h4.get('score')}), RSI {h4.get('rsi14')}, "
                     f"MACD hist {h4.get('macd', {}).get('hist')} (prev {h4.get('macd', {}).get('hist_prev')})")
        if t.get("vs_spy_pct"):
            v = t["vs_spy_pct"]
            L.append(f"  vs SPY (relative strength): 1d {v['1d']:+}pp  5d {v['5d']:+}pp  1m {v['1m']:+}pp")
        if t.get("next_earnings"):
            L.append(f"  next earnings: {t['next_earnings']}")
        if t.get("today"):
            td = t["today"]
            L.append(f"  today {td['date']}: open {td['open']} (gap {td['gap_pct']:+}%), first-15-min range "
                     f"{td['opening_range_15m'][0]}–{td['opening_range_15m'][1]}, high {td['high']} low {td['low']} "
                     f"last {td['last']} ({td['last_bar_et']} ET)")
        if t.get("session"):
            s = t["session"]
            L.append(f"  session {s['date']}: open {s['open']} high {s['high']} low {s['low']} last {s['last']} VWAP {s['vwap']}")
        L.append("  supports:    " + "; ".join(f"{x['price']} {x['what']} ({x['distance_pct']:+}%)" for x in t["levels"]["supports"]))
        L.append("  resistances: " + "; ".join(f"{x['price']} {x['what']} ({x['distance_pct']:+}%)" for x in t["levels"]["resistances"]))
        for p in t["open_patterns"]:
            L.append(f"  open pattern: {p['type']} ({p['bias']}) {p['start']}→{p['end']}, {p['status']}"
                     + (f", target {p['target']}" if p["target"] else "") + f" — {p['note']}")
        for p in t["recent_breaks"]:
            L.append(f"  recent: {p['type']} ({p['bias']}) {p['status']}" + (f", target {p['target']}" if p["target"] else ""))
    return "\n".join(L)


def pick_session(now, force=False):
    """Slot for an ET time: a scheduled slot within ±12 min on weekdays, None for the
    unused cron fires (15:15, 16:45) unless forced, otherwise an on-demand run."""
    at = lambda hm: now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    if now.weekday() >= 5:
        return "now"
    if not force and any(abs(now - at(hm)) <= WINDOW for hm in SKIP_SLOTS):
        return None
    return next((k for k, (hm, _) in SESSIONS.items() if hm and abs(now - at(hm)) <= WINDOW), "now")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="auto", choices=["auto"] + list(SESSIONS),
                    help="auto (default): pick the slot from the clock; off-slot times are on-demand runs")
    ap.add_argument("--force", action="store_true", help="never skip (manual runs at 15:15 / 16:45)")
    ap.add_argument("--index", action="store_true", help="rebuild briefs/index.json and exit")
    args = ap.parse_args()
    os.makedirs(BRIEF_DIR, exist_ok=True)
    if args.index:
        rebuild_index(); return
    now = datetime.now(ET)
    session = args.session if args.session != "auto" else pick_session(now, args.force)
    if session is None:
        print(f"SKIP: {now:%H:%M} ET is an unused slot of the schedule. Nothing to do.")
        return
    title = SESSIONS[session][1]

    snap = {"session": session, "title": f"{title} — {now:%a %b %d, %Y} {now:%H:%M} ET",
            "generated_et": now.strftime("%Y-%m-%d %H:%M"), "context": context_snapshot(), "tickers": []}
    spy = None
    if "SPY" in TICKERS:   # SPY first, so the others can be compared with it
        try:
            spy = ticker_snapshot("SPY")
        except Exception:
            spy = None
    for sym in TICKERS:
        try:
            snap["tickers"].append(spy if sym == "SPY" and spy else ticker_snapshot(sym, spy))
        except Exception as e:
            snap["tickers"].append({"symbol": sym, "error": str(e)})
    bid = f"{now:%Y-%m-%d-%H%M}-{session}"
    path = os.path.join(BRIEF_DIR, bid + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, default=float)
    print(f"SESSION={session}")
    print(f"BRIEF_ID={bid}")
    print(f"DATA={os.path.relpath(path, ROOT)}")
    print(f"WRITE_BRIEF_TO=briefs/{bid}.md")
    prev = sorted(glob.glob(os.path.join(BRIEF_DIR, "*.md")))
    if prev:
        print(f"PREVIOUS_BRIEF={os.path.relpath(prev[-1], ROOT)}")
    today = [os.path.relpath(f, ROOT) for f in prev if os.path.basename(f).startswith(f"{now:%Y-%m-%d}")]
    if today:
        print("TODAYS_BRIEFS=" + ", ".join(today))
    print()
    print(summary_text(snap))


if __name__ == "__main__":
    main()
