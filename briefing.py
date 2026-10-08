"""
Market brief data collector — the numbers behind the 9:15 / 15:45 ET briefs.

For each ticker on the formations page (QQQ, SPY, SOXX) it computes the daily
and 4H trend state (EMA9/EMA21/SMA50/SMA200 score, RSI14, MACD 12/26/9), the
nearest supports / resistances (pivot levels, active trendlines and channels,
key moving averages, anchored VWAPs) and open chart patterns. It also snapshots
the cross-asset backdrop: index futures, VIX, Treasury yields, dollar, oil,
gold and crypto.

The scheduled routine runs this, adds news / calendar research, and writes the
narrative brief next to the JSON snapshot in briefs/.

Usage:
    python briefing.py --session am          # exits with "SKIP" unless it is ~9:15 ET
    python briefing.py --session pm          # ~15:45 ET
    python briefing.py --session am --force  # ignore the clock (manual runs)
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
SESSIONS = {"am": (9, 15, "Pre-market brief"), "pm": (15, 45, "Pre-close brief")}
# window around the target time in which a scheduled fire counts (the routine is
# scheduled at both possible UTC times so it is right in summer and winter time)
WINDOW_BEFORE, WINDOW_AFTER = timedelta(minutes=25), timedelta(minutes=40)

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
        elif p["cat"] in ("channel", "triangle") and p["lines"]:
            for L, side in zip(p["lines"], ("lower", "upper") if len(p["lines"]) == 2 else ("line",)):
                lv.append((L[3], f"{p['type'].lower()} {side} line"))
        elif p["cat"] == "reversal":
            lv.append((p["lines"][0][3], f"{p['type'].lower()} neckline"))
    for n, m in state["mas"].items():
        if n in ("EMA21", "SMA50", "SMA200"):
            lv.append((m["value"], n))
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


def ticker_snapshot(sym):
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
                      "generated_et": meta.get("generated_et")})
    items.sort(key=lambda x: x["id"], reverse=True)
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
        if h4:
            L.append(f"  4H: {h4.get('verdict')} (score {h4.get('score')}), RSI {h4.get('rsi14')}, "
                     f"MACD hist {h4.get('macd', {}).get('hist')} (prev {h4.get('macd', {}).get('hist_prev')})")
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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", choices=SESSIONS)
    ap.add_argument("--force", action="store_true", help="run even outside the scheduled time window")
    ap.add_argument("--index", action="store_true", help="rebuild briefs/index.json and exit")
    args = ap.parse_args()
    os.makedirs(BRIEF_DIR, exist_ok=True)
    if args.index:
        rebuild_index(); return
    if not args.session:
        ap.error("--session am|pm is required")

    now = datetime.now(ET)
    hh, mm, title = SESSIONS[args.session]
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if not args.force:
        if now.weekday() >= 5 or not (target - WINDOW_BEFORE <= now <= target + WINDOW_AFTER):
            print(f"SKIP: it is {now:%a %H:%M} ET, outside the {hh}:{mm:02d} ET window. Nothing to do.")
            return

    snap = {"session": args.session, "title": f"{title} — {now:%a %b %d, %Y}",
            "generated_et": now.strftime("%Y-%m-%d %H:%M"), "context": context_snapshot(), "tickers": []}
    for sym in TICKERS:
        try:
            snap["tickers"].append(ticker_snapshot(sym))
        except Exception as e:
            snap["tickers"].append({"symbol": sym, "error": str(e)})
    bid = f"{now:%Y-%m-%d}-{args.session}"
    path = os.path.join(BRIEF_DIR, bid + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, default=float)
    print(f"BRIEF_ID={bid}")
    print(f"DATA={os.path.relpath(path, ROOT)}")
    print(f"WRITE_BRIEF_TO=briefs/{bid}.md")
    prev = sorted(glob.glob(os.path.join(BRIEF_DIR, "*.md")))
    if prev:
        print(f"PREVIOUS_BRIEF={os.path.relpath(prev[-1], ROOT)}")
    print()
    print(summary_text(snap))


if __name__ == "__main__":
    main()
