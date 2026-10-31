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
# label, bar size: hours for intraday, "D" / "W" / "M" built from daily bars
INV_FRAMES = [("1H", 1), ("2H", 2), ("4H", 4), ("8H = 1D", "D"), ("1W", "W"), ("1M", "M")]   # 8h of a 6.5h session = the daily bar
BAR_WORDS = {"D": ("today's bar", "next day"), "W": ("this week's bar", "next week"), "M": ("this month's bar", "next month")}


def _period_bars(d, rule):
    """Weekly / monthly bars from daily bars, stamped with the period's first trading day."""
    g = d.assign(_t=d.index).groupby(d.index.to_period(rule))
    out = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                Close=("Close", "last"), Volume=("Volume", "sum"), _t=("_t", "first"))
    return out.set_index("_t").rename_axis(None)


def _forming(df, size, now):
    """Is the last bar still forming (its session has started and the bar is not finished)?"""
    t = df.index[-1]
    day0 = t.normalize()
    if size == "W":
        end = day0 + pd.Timedelta(days=4 - day0.weekday(), hours=16)               # Friday 16:00
    elif size == "M":
        end = (day0 + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(hours=16)  # last day of the month 16:00
    elif size == "D":
        end = day0 + pd.Timedelta(hours=16)
    else:
        end = min(t + pd.Timedelta(hours=size), day0 + pd.Timedelta(hours=16))
    start = t if isinstance(size, int) else day0 + pd.Timedelta(hours=9, minutes=30)
    return start <= now < end


def _num(v, nd=6):
    v = float(v)
    return None if np.isnan(v) or np.isinf(v) else round(v, nd)


def _state(df):
    """Indicator state after the last bar of df: what the page needs to solve one more close.
    RSI(14) Wilder · MACD(12/26/9) · Stoch RSI(14,14,3,3) · SMI(10,3, signal EMA 10) — TradingView defaults."""
    c, h, lo = df["Close"], df["High"], df["Low"]
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    e12, e26 = _ema(c, 12), _ema(c, 26)
    m = e12 - e26
    sig = _ema(m, 9)
    rsi = 100 - 100 / (1 + g / l)
    # Stochastic RSI: raw = where RSI sits in its 14-bar range; %K = SMA3(raw); %D = SMA3(%K)
    rmin, rmax = rsi.rolling(14).min(), rsi.rolling(14).max()
    raw = (rsi - rmin) / (rmax - rmin) * 100
    k = raw.rolling(3).mean()
    dd = k.rolling(3).mean()
    # SMI (Blau): close vs the middle of the 10-bar range, double-smoothed (EMA3 of EMA3), over half the range
    hh, ll = h.rolling(10).max(), lo.rolling(10).min()
    rel, rng = c - (hh + ll) / 2, hh - ll
    rel1, rng1 = _ema(rel, 3), _ema(rng, 3)
    rel2, rng2 = _ema(rel1, 3), _ema(rng1, 3)
    smi = 200 * rel2 / rng2
    smi_sig = _ema(smi, 10)
    extra = {"rsi_lo13": rsi.iloc[-13:].min(), "rsi_hi13": rsi.iloc[-13:].max(),
             "raw1": raw.iloc[-1], "raw2": raw.iloc[-2], "k1": k.iloc[-1], "k2": k.iloc[-2], "k_now": k.iloc[-1], "d_now": dd.iloc[-1],
             "hi9": h.iloc[-9:].max(), "lo9": lo.iloc[-9:].min(), "rel_e1": rel1.iloc[-1], "rel_e2": rel2.iloc[-1],
             "rng_e1": rng1.iloc[-1], "rng_e2": rng2.iloc[-1], "smi_now": smi.iloc[-1], "smi_sig": smi_sig.iloc[-1]}
    return {**{k_: _num(v) for k_, v in extra.items()},
            "rsi_prev13": float(rsi.iloc[-13:].mean()), "rsi_ma": float(rsi.iloc[-14:].mean()),   # RSI signal = SMA14 of RSI
            "prev_close": float(c.iloc[-1]), "avg_gain": float(g.iloc[-1]), "avg_loss": float(l.iloc[-1]),
            "ema12": float(e12.iloc[-1]), "ema26": float(e26.iloc[-1]), "signal": float(sig.iloc[-1]),
            "rsi": float(100 - 100 / (1 + g.iloc[-1] / l.iloc[-1])) if l.iloc[-1] > 0 else 100.0,
            "macd": float(m.iloc[-1])}


def inverse(sym):
    from datetime import datetime
    now = datetime.now(ET).replace(tzinfo=None)
    h1 = _load(sym, "60m", "1y", "1h")
    d1 = _load(sym, "1d", "max", "1D")            # long history so monthly RSI / EMAs are settled
    d1 = P._append_today(d1, sym)
    price = float(h1["Close"].iloc[-1])
    out = []
    for label, size in INV_FRAMES:
        df = (d1 if size == "D" else _period_bars(d1, "W-FRI") if size == "W" else _period_bars(d1, "M") if size == "M"
              else h1 if size == 1 else P._session_bars(h1, size))
        forming = _forming(df, size, now)
        base = df.iloc[:-1] if forming else df
        st = _state(base)
        now_st = _state(df)
        t = df.index[-1]
        out.append({"tf": label, "forming": forming,
                    "bar": (f"current bar from {t:%H:%M}" if forming else "next bar") if isinstance(size, int) else BAR_WORDS[size][0 if forming else 1],
                    "rsi_now": round(now_st["rsi"], 1), "rsi_ma_now": round(now_st["rsi_ma"], 1), "macd_now": round(now_st["macd"], 3), "signal_now": round(now_st["signal"], 3),
                    "stoch_k_now": _num(now_st["k_now"], 1), "stoch_d_now": _num(now_st["d_now"], 1),
                    "smi_value_now": _num(now_st["smi_now"], 1), "smi_signal_now": _num(now_st["smi_sig"], 1),
                    # the forming bar's high / low so far: a close beyond them also moves the SMI range
                    "bar_hi": float(df["High"].iloc[-1]) if forming else None, "bar_lo": float(df["Low"].iloc[-1]) if forming else None,
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}})
    return {"symbol": sym, "price": round(price, 2), "asof": f"{h1.index[-1]:%Y-%m-%d %H:%M}", "timeframes": out}



# ── Crosses: when did each indicator last cross its average / middle line, per timeframe ───────
def supertrend(df, n=10, mult=3.0):
    """TradingView's Supertrend: ATR (Wilder, n) bands around hl2 that only tighten; +1 up / -1 down."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    trend = np.ones(len(c), dtype=int)
    line = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        if trend[i - 1] == -1 and c[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and c[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
        line[i] = up[i] if trend[i] == 1 else dn[i]
    return pd.Series(line, index=df.index), pd.Series(trend, index=df.index)


def _cross_series(df):
    """(label, a, b, note-function) pairs: a crossing b is the event."""
    c = df["Close"]
    rsi = _rsi(c)
    m = _ema(c, 12) - _ema(c, 26)
    sig = _ema(m, 9)
    ap = (df["High"] + df["Low"] + c) / 3
    esa = ap.ewm(span=10, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
    wt1 = ((ap - esa) / (0.015 * d)).ewm(span=21, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    rmin, rmax = rsi.rolling(14).min(), rsi.rolling(14).max()
    k = ((rsi - rmin) / (rmax - rmin) * 100).rolling(3).mean()
    kd = k.rolling(3).mean()
    hh, ll = df["High"].rolling(10).max(), df["Low"].rolling(10).min()
    smi = 200 * _ema(_ema(c - (hh + ll) / 2, 3), 3) / _ema(_ema(hh - ll, 3), 3)
    smis = _ema(smi, 10)
    st_line, st_dir = supertrend(df)
    zero = pd.Series(0.0, index=df.index)
    wt_zone = lambda i: "in the overbought zone" if wt2.iloc[i] >= 53 else "in the oversold zone" if wt2.iloc[i] <= -53 else ""
    return [
        ("Price vs 21-day EMA", c, _ema(c, 21), None),
        ("Price vs 50 SMA", c, c.rolling(50).mean(), None),
        ("Price vs 200 SMA", c, c.rolling(200).mean(), None),
        ("RSI vs 50 (middle)", rsi, pd.Series(50.0, index=df.index), None),
        ("RSI vs its average (SMA14)", rsi, rsi.rolling(14).mean(), None),
        ("MACD vs signal", m, sig, None),
        ("MACD vs 0", m, zero, None),
        ("WaveTrend WT1 vs WT2", wt1, wt2, wt_zone),
        ("WaveTrend vs 0 (middle)", wt1, zero, None),
        ("Stoch RSI %K vs %D", k, kd, None),
        ("Stoch RSI vs 50 (middle)", k, pd.Series(50.0, index=df.index), None),
        ("SMI vs signal", smi, smis, None),
        ("Supertrend (10, 3)", st_dir.astype(float), zero, None),
    ], st_line, st_dir


def _last_cross(a, b, idx, note):
    x = (a - b).values
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return None
    side = "above" if x[-1] > 0 else "below"
    for i in range(len(x) - 1, 0, -1):
        if not (ok[i] and ok[i - 1]):
            continue
        if (x[i - 1] <= 0 < x[i]) or (x[i - 1] >= 0 > x[i]):
            t = idx[i]
            return {"side": side, "dir": "up" if x[i] > 0 else "down", "bars_ago": int(len(x) - 1 - i),
                    "time": f"{t:%Y-%m-%d}" if (t.hour, t.minute) == (0, 0) else f"{t:%Y-%m-%d %H:%M}",
                    "note": note(i) if note else ""}
    return {"side": side, "dir": None, "bars_ago": None, "time": None, "note": ""}


def crosses(sym):
    """Last cross of each indicator on 1H, 2H, 4H, 1D and 1W bars."""
    h1 = _load(sym, "60m", "1y", "1h")
    d1 = P._append_today(_load(sym, "1d", "5y", "1D"), sym)
    frames = [("1H", h1), ("2H", P._session_bars(h1, 2)), ("4H", P._session_bars(h1, 4)), ("1D", d1), ("1W", _period_bars(d1, "W-FRI"))]
    out, labels = {}, None
    for tf, df in frames:
        series, st_line, st_dir = _cross_series(df)
        labels = [s[0] for s in series]
        out[tf] = {lab: _last_cross(a, b, df.index, note) for lab, a, b, note in series}
        out[tf]["Supertrend (10, 3)"]["level"] = None if np.isnan(st_line.iloc[-1]) else round(float(st_line.iloc[-1]), 2)
        out[tf]["Supertrend (10, 3)"]["side"] = "up" if st_dir.iloc[-1] == 1 else "down"
    return {"symbol": sym, "price": round(float(h1["Close"].iloc[-1]), 2), "asof": f"{h1.index[-1]:%Y-%m-%d %H:%M}",
            "timeframes": [f[0] for f in frames], "indicators": labels, "crosses": out}


def _tf_label(n_up, n):
    return "bullish" if n_up >= 0.65 * n else "bearish" if n_up <= 0.35 * n else "mixed"


def trend_check(syms):
    """Short term vs longer term per ticker: how many of the crosses-table indicators are above their
    average / middle line on each timeframe, Supertrend per timeframe, and the key levels."""
    out = []
    for sym in syms:
        try:
            r = crosses(sym)
        except Exception as e:
            out.append({"symbol": sym, "error": str(e)})
            continue
        c, n = r["crosses"], len(r["indicators"])
        tfs = {}
        for tf in r["timeframes"]:
            up = sum(1 for k in r["indicators"] if c[tf][k] and c[tf][k]["side"] in ("above", "up"))
            st = c[tf]["Supertrend (10, 3)"]
            tfs[tf] = {"up": up, "of": n, "label": _tf_label(up, n), "st_dir": st["side"], "st_level": st.get("level"),
                       "st_flip": st.get("time")}
        short = [tfs[t]["label"] for t in ("1H", "2H", "4H", "1D")]
        s_lab = "bearish" if short.count("bearish") >= 3 else "bullish" if short.count("bullish") >= 3 else "mixed"
        d1 = P._append_today(_load(sym, "1d", "2y", "1D"), sym)
        cl = d1["Close"]
        hi52 = float(cl.iloc[-252:].max())
        sma200 = float(cl.rolling(200).mean().iloc[-1])
        out.append({"symbol": sym, "price": r["price"], "asof": r["asof"], "timeframes": tfs, "short_term": s_lab,
                    "weekly": tfs["1W"]["label"],
                    "levels": {"daily_supertrend": tfs["1D"]["st_level"], "weekly_supertrend": tfs["1W"]["st_level"],
                               "sma200": round(sma200, 2), "high_52w": round(hi52, 2), "minus10": round(hi52 * 0.9, 2),
                               "minus20": round(hi52 * 0.8, 2), "from_high_pct": round((float(cl.iloc[-1]) / hi52 - 1) * 100, 1)}})
    return out


def trend_text(rows, bear=None):
    L = ["SHORT-TERM TREND vs BEAR MARKET (crosses table: indicators above their average / middle line, of 13)"]
    for t in rows:
        if "error" in t:
            L.append(f"  {t['symbol']}: ERROR {t['error']}")
            continue
        tf = t["timeframes"]
        lv = t["levels"]
        L.append(f"  {t['symbol']} {t['price']}: short term {t['short_term'].upper()}, weekly {t['weekly']} · "
                 + " · ".join(f"{k} {v['up']}/{v['of']} ST {v['st_dir']} {v['st_level']}" for k, v in tf.items())
                 + f" · daily Supertrend {lv['daily_supertrend']}, weekly Supertrend {lv['weekly_supertrend']}, 200-day {lv['sma200']}, "
                   f"52w high {lv['high_52w']} ({lv['from_high_pct']}%), -10% {lv['minus10']}, -20% {lv['minus20']}")
    if bear:
        L.append(f"  bear market watch: {bear.get('state')} ({bear.get('signs_on')} of {bear.get('signs_total')} signs)")
    return "\n".join(L)


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
