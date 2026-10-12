"""
Multi-timeframe turn ladder — has a turn (bottom or top) been confirmed on the
5-minute, 15-minute, 1-hour, 4-hour and daily charts, and what level would
confirm it on the next one?

On each timeframe:
    * swing highs / lows come from the zigzag in patterns.py (swing size = 2 x ATR%);
    * a turn UP is confirmed when, after a swing low, a bar CLOSES above the swing
      high that came before that low (break of structure); it is a "reversal"
      when the two prior swing highs were falling (a downtrend), otherwise a
      "continuation". A turn DOWN is the mirror image;
    * if the latest swing is a low that has not been broken out of yet, the
      "pending" level is the swing high to close above (and vice versa);
    * trend context: EMA9 vs EMA21, price vs EMA21, RSI14, MACD histogram.

A typical bottom shows up first on 5m, then 15m, then 1h ... The ladder lists
when each timeframe confirmed and the level the slower ones still need.

Usage:
    python mtf.py QQQ SOXX
"""

import sys
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

import patterns as P

ET = ZoneInfo("America/New_York")
# label, Yahoo interval, period, minimum swing %
TIMEFRAMES = [("5m", "5m", "5d", 0.15), ("15m", "15m", "1mo", 0.25), ("1h", "60m", "3mo", 0.5),
              ("4h", "60m", "1y", 1.0), ("1D", "1d", "2y", 2.0)]


def _load(sym, interval, period, label):
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"no {label} data")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(ET).tz_localize(None) if interval != "1d" else df.index.tz_localize(None)
    if label == "4h":
        df = P._session_bars(df, 4)
    return df


def _ema(c, n):
    return c.ewm(span=n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)


def analyze_tf(df, label, min_swing):
    c = df["Close"]
    t = lambda i: P.ts(df.index[i])
    swing = max(min_swing / 100, round(2 * P.atr_pct(df), 4))
    piv = P.zigzag(df, swing)
    cv, n = c.values, len(c)

    # structure breaks: after a swing low, the first close above the swing high before it
    events = []
    for j in range(1, len(piv)):
        a, b = piv[j - 1], piv[j]                        # a = opposite swing before b
        up = b["kind"] == "L"
        # look for the break until the next swing of b's kind (a new low/high replaces b)
        stop = piv[j + 2]["i"] if j + 2 < len(piv) else n
        for k in range(b["i"] + 1, stop):
            if (cv[k] > a["price"]) if up else (cv[k] < a["price"]):
                prev_same = [p for p in piv[:j - 1] if p["kind"] == a["kind"]][-1:]  # swing before a of a's kind
                trend_against = bool(prev_same) and ((prev_same[0]["price"] > a["price"]) if up else (prev_same[0]["price"] < a["price"]))
                events.append({"dir": "up" if up else "down", "kind": "reversal" if trend_against else "continuation",
                               "when": t(k), "i": k, "level": round(float(a["price"]), 2),
                               "swing": {"price": round(float(b["price"]), 2), "when": t(b["i"])}})
                break
    events.sort(key=lambda e: e["i"])
    last = events[-1] if events else None

    # pending confirmation: latest swing not yet broken out of
    pending = None
    if len(piv) >= 2:
        b, a = piv[-1], piv[-2]
        up = b["kind"] == "L"
        broken = any((cv[k] > a["price"]) if up else (cv[k] < a["price"]) for k in range(b["i"] + 1, n))
        if not broken:
            same = last is not None and last["dir"] == ("up" if up else "down")
            pending = {"dir": "up" if up else "down", "kind": "continuation" if same else "turn",
                       "level": round(float(a["price"]), 2),
                       "from_swing": {"price": round(float(b["price"]), 2), "when": t(b["i"]), "confirmed": b["confirmed"]},
                       "distance_pct": round((a["price"] / cv[-1] - 1) * 100, 2)}

    e9, e21 = _ema(c, 9), _ema(c, 21)
    rsi = _rsi(c)
    m = _ema(c, 12) - _ema(c, 26)
    hist = m - _ema(m, 9)
    xs = np.sign(e9 - e21).diff().fillna(0)
    xs = xs[xs != 0]
    trend = ("up" if e9.iloc[-1] > e21.iloc[-1] and cv[-1] > e21.iloc[-1] else
             "down" if e9.iloc[-1] < e21.iloc[-1] and cv[-1] < e21.iloc[-1] else "mixed")
    return {
        "tf": label, "bars": n, "last_bar": t(n - 1), "close": round(float(cv[-1]), 2), "swing_pct": round(swing * 100, 2),
        "trend": trend,
        "ema_cross": None if xs.empty else {"dir": "up" if xs.iloc[-1] > 0 else "down", "when": P.ts(xs.index[-1])},
        "rsi": round(float(rsi.iloc[-1]), 1), "rsi_prev5": round(float(rsi.iloc[-6]), 1),
        "macd_hist": round(float(hist.iloc[-1]), 3), "macd_rising": bool(hist.iloc[-1] > hist.iloc[-2]),
        "last_turn": None if last is None else {k: last[k] for k in ("dir", "kind", "when", "level", "swing")},
        "last_reversal": next(({k: e[k] for k in ("dir", "when", "level", "swing")} for e in reversed(events) if e["kind"] == "reversal"), None),
        "pending": pending,
        "swing_high": next((round(float(q["price"]), 2) for q in reversed(piv) if q["kind"] == "H"), None),
        "swing_low": next((round(float(q["price"]), 2) for q in reversed(piv) if q["kind"] == "L"), None),
    }


def ladder(sym):
    sym = sym.upper()
    rows, errors = [], {}
    cache = {}
    for label, interval, period, mn in TIMEFRAMES:
        try:
            key = (interval, period) if label != "4h" else ("60m", period, "4h")
            if key not in cache:
                cache[key] = _load(sym, interval, period, label)
            rows.append(analyze_tf(cache[key], label, mn))
        except Exception as e:
            errors[label] = str(e)
    # cascade: has the fastest timeframe's latest turn spread to the slower ones?
    summary = None
    if rows and rows[0].get("last_turn"):
        base = rows[0]["last_turn"]
        d, since = base["dir"], base["swing"]["when"]
        steps = []
        for r in rows:
            lt, pn = r.get("last_turn"), r.get("pending")
            if lt and lt["dir"] == d and lt["when"] >= since:
                st = {"tf": r["tf"], "status": "confirmed", "when": lt["when"], "level": lt["level"]}
            elif lt and lt["dir"] == d:
                st = {"tf": r["tf"], "status": "already trending " + d, "when": lt["when"], "level": lt["level"]}
            elif pn and pn["dir"] == d:
                st = {"tf": r["tf"], "status": "pending", "level": pn["level"], "distance_pct": pn["distance_pct"]}
            else:   # still pointing the other way: the latest swing high (low) flips it
                lv = r.get("swing_high") if d == "up" else r.get("swing_low")
                st = {"tf": r["tf"], "status": "pending", "level": lv,
                      "distance_pct": None if lv is None else round((lv / r["close"] - 1) * 100, 2)}
            steps.append(st)
        nxt = next((x for x in steps if x["status"] == "pending"), None)
        summary = {"dir": d, "from_swing": base["swing"], "steps": steps,
                   "next": nxt}
    return {"symbol": sym, "timeframes": rows, "errors": errors, "summary": summary}



# ── Indicator-to-price: which close puts RSI / MACD at a given value ─────────────
# Wilder's RSI and the MACD EMAs are recursive, so the effect of ONE more close is exact
# algebra. inverse() returns each timeframe's state before the bar being solved; the page
# turns any RSI / MACD level into the close that bar would need.
INV_FRAMES = [("1H", 1), ("2H", 2), ("4H", 4), ("8H = 1D", None)]   # 8h of a 6.5h session = the daily bar


def _forming(df, hours, now):
    """Is the last bar still forming (market open and the bar not finished)?"""
    t = df.index[-1]
    close_t = t.normalize() + pd.Timedelta(hours=16)
    end = close_t if hours is None else min(t + pd.Timedelta(hours=hours), close_t)
    start = t.normalize() + pd.Timedelta(hours=9, minutes=30) if hours is None else t
    return start <= now < end


def _state(c):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    e12, e26 = _ema(c, 12), _ema(c, 26)
    m = e12 - e26
    sig = _ema(m, 9)
    rsi = 100 - 100 / (1 + g / l)
    return {"rsi_prev13": float(rsi.iloc[-13:].mean()), "rsi_ma": float(rsi.iloc[-14:].mean()),   # RSI signal = SMA14 of RSI
            "prev_close": float(c.iloc[-1]), "avg_gain": float(g.iloc[-1]), "avg_loss": float(l.iloc[-1]),
            "ema12": float(e12.iloc[-1]), "ema26": float(e26.iloc[-1]), "signal": float(sig.iloc[-1]),
            "rsi": float(100 - 100 / (1 + g.iloc[-1] / l.iloc[-1])) if l.iloc[-1] > 0 else 100.0,
            "macd": float(m.iloc[-1])}


def inverse(sym):
    from datetime import datetime
    now = datetime.now(ET).replace(tzinfo=None)
    h1 = _load(sym, "60m", "1y", "1h")
    d1 = _load(sym, "1d", "2y", "1D")
    d1 = P._append_today(d1, sym)
    price = float(h1["Close"].iloc[-1])
    out = []
    for label, hours in INV_FRAMES:
        df = d1 if hours is None else (h1 if hours == 1 else P._session_bars(h1, hours))
        forming = _forming(df, hours, now)
        base = df["Close"].iloc[:-1] if forming else df["Close"]
        st = _state(base)
        now_st = _state(df["Close"])
        t = df.index[-1]
        out.append({"tf": label, "forming": forming,
                    "bar": (f"current bar from {t:%H:%M}" if hours else "today's bar") if forming else "next bar",
                    "rsi_now": round(now_st["rsi"], 1), "rsi_ma_now": round(now_st["rsi_ma"], 1), "macd_now": round(now_st["macd"], 3), "signal_now": round(now_st["signal"], 3),
                    **{k: round(v, 6) for k, v in st.items()}})
    return {"symbol": sym, "price": round(price, 2), "asof": f"{h1.index[-1]:%Y-%m-%d %H:%M}", "timeframes": out}


def text(l):
    L = [f"{l['symbol']} turn ladder"]
    for r in l["timeframes"]:
        lt, pd_ = r.get("last_turn"), r.get("pending")
        s = f"  {r['tf']:<4} trend {r['trend']:<5} RSI {r['rsi']:>5} MACD {'+' if r['macd_hist'] > 0 else '-'}{'↑' if r['macd_rising'] else '↓'}"
        if lt:
            up = lt["dir"] == "up"
            s += (f" | turned {lt['dir']} ({lt['kind']}) {lt['when']}: closed {'above' if up else 'below'} the prior swing "
                  f"{'high' if up else 'low'} {lt['level']} after the {'low' if up else 'high'} {lt['swing']['price']} ({lt['swing']['when']})")
        if pd_:
            s += (f" | next {pd_['kind']} {pd_['dir']}: close {'above' if pd_['dir'] == 'up' else 'below'} {pd_['level']} "
                  f"({pd_['distance_pct']:+.2f}%)")
        L.append(s)
    for k, e in l["errors"].items():
        L.append(f"  {k}: ERROR {e}")
    s = l.get("summary")
    if s:
        word = "above" if s["dir"] == "up" else "below"
        parts = []
        for x in s["steps"]:
            if x["status"] == "confirmed":
                parts.append(f"{x['tf']} confirmed {x['when']}")
            elif x["status"] == "pending":
                parts.append(f"{x['tf']} needs a close {word} {x['level']}" + (f" ({x['distance_pct']:+.2f}%)" if x.get("distance_pct") is not None else ""))
            else:
                parts.append(f"{x['tf']} {x['status']}")
        L.append(f"  cascade (turn {s['dir']} from the {s['from_swing']['price']} swing at {s['from_swing']['when']}): " + "; ".join(parts))
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for s in sys.argv[1:] or ["QQQ"]:
        print(text(ladder(s)))
