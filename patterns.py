"""
Chart Formation Detector — finds classic price patterns on daily charts.

Detects (from zigzag swing pivots):
    Reversals:   double top / bottom, head & shoulders / inverse
    Triangles:   ascending, descending, symmetrical; rising / falling wedges;
                 short channels / rectangles; broadening formations
    Channels:    parallel trend channels (a trendline plus a parallel line on the other side)
    Trendlines:  support / resistance lines touched by 3+ swing pivots
    Levels:      horizontal support / resistance (clusters of swing pivots)

Usage:
    python patterns.py                      # QQQ, SPY, SOXX, 2 years -> patterns.html
    python patterns.py NVDA SMH --period 5y
    python patterns.py QQQ --interval 4h --period 1y
    python patterns.py QQQ --pct 4          # fixed 4% zigzag swing instead of auto
    python patterns.py QQQ --refresh        # ignore cached prices
    python patterns.py QQQ --all            # include failed / expired patterns
    python patterns.py QQQ --no-browser

Prices are cached in data/<SYMBOL>_<period>.csv (git-ignored, refreshed once per day).
The report (patterns.html) is served by Vercel at /patterns; its lookup box calls
api/patterns.py to analyze any ticker on demand.
"""

import sys, os, re, json, argparse, webbrowser
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

ROOT     = os.path.dirname(os.path.abspath(__file__))
# Vercel functions can only write to /tmp
DATA_DIR = "/tmp/data" if os.environ.get("VERCEL") else os.path.join(ROOT, "data")
REPORT   = os.path.join(ROOT, "patterns.html")
def _watchlist():
    """Tickers shown on the formations page and covered by the market briefs.
    Edit watchlist.txt (one ticker per line, # for comments) to change both."""
    try:
        with open(os.path.join(ROOT, "watchlist.txt"), encoding="utf-8") as f:
            syms = [l.split("#")[0].strip().upper() for l in f]
        return [x for x in syms if x] or ["QQQ", "SPY", "SOXX"]
    except OSError:
        return ["QQQ", "SPY", "SOXX"]


DEFAULT_SYMBOLS = _watchlist()
PERIODS  = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y")
MONTHS   = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60, "10y": 120}
WARMUP_PERIOD = {"1mo": "2y", "3mo": "2y", "6mo": "2y", "1y": "2y", "2y": "5y", "5y": "10y", "10y": "max"}
INTERVALS = ("1d", "4h", "2h", "1h")
# Yahoo keeps ~730 days of hourly bars; cap the window so charts stay responsive
INTRADAY_MAX = {"1h": "6mo", "2h": "1y", "4h": "2y"}
BARS_PER_DAY = {"1d": 1, "4h": 2, "2h": 4, "1h": 7}


def ts(t):
    """Timestamp label: date for daily bars, date + time for intraday bars."""
    return t.strftime("%Y-%m-%d") if (t.hour, t.minute) == (0, 0) else t.strftime("%Y-%m-%d %H:%M")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_prices(sym, period="2y", refresh=False, interval="1d", fill_today=False):
    """OHLCV in exchange-local time, cached per symbol+interval+period.
    Daily cache is reused for the same calendar day, intraday cache for 15 minutes.
    2h / 4h bars are built from hourly bars, starting at each session's open."""
    os.makedirs(DATA_DIR, exist_ok=True)
    daily = interval == "1d"
    fetch_int, fetch_per = ("1d", period) if daily else ("60m", "730d")
    path = os.path.join(DATA_DIR, f"{sym}_{fetch_int}_{fetch_per}.csv")
    df = None
    if not refresh and os.path.exists(path):
        mt = os.path.getmtime(path)
        age = datetime.now().timestamp() - mt
        fresh = (date.fromtimestamp(mt) == date.today() and age < 3600) if daily else age < 900
        if fresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df is None:
        df = yf.Ticker(sym).history(period=fetch_per, interval=fetch_int, auto_adjust=True)
        if df.empty:
            raise ValueError(f"no price data for {sym}")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = df.index.tz_localize(None)
        df.to_csv(path)
    if daily and fill_today:
        df = _append_today(df, sym)
    if interval in ("2h", "4h"):
        df = _session_bars(df, int(interval[0]))
    if len(df) < 60:
        raise ValueError(f"only {len(df)} bars of data for {sym}")
    return df


def _append_today(df, sym):
    """Yahoo's daily bar for the current session is often missing or empty until
    well after the close; build it from today's regular-hours 5-minute bars."""
    try:
        h = yf.Ticker(sym).history(period="1d", interval="5m")
    except Exception:
        return df
    if h.empty:
        return df
    h.index = h.index.tz_localize(None)
    day = h.index[-1].normalize()
    if len(df) and df.index[-1] >= day:
        return df
    row = pd.DataFrame({"Open": [h["Open"].iloc[0]], "High": [h["High"].max()], "Low": [h["Low"].min()],
                        "Close": [h["Close"].iloc[-1]], "Volume": [h["Volume"].sum()]}, index=[day])
    return pd.concat([df, row])


def _session_bars(df, hours):
    """Combine hourly bars into `hours`-hour bars, counted from each day's first bar
    (US stocks: 4h = 9:30-13:30 and 13:30-16:00, like most charting platforms)."""
    t = df.index.to_series()
    day = t.dt.normalize()
    block = ((t - t.groupby(day).transform("min")).dt.total_seconds() // (hours * 3600)).astype(int)
    g = df.assign(_t=t).groupby([day, block])
    out = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                Close=("Close", "last"), Volume=("Volume", "sum"), _t=("_t", "first"))
    return out.set_index("_t").rename_axis(None)


def atr_pct(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float((tr.rolling(period).mean() / c).median())


# ── Swing pivots ───────────────────────────────────────────────────────────────
def zigzag(df, pct):
    """Swing highs/lows: a reversal of at least `pct` (fraction) from the running extreme.
    Returns list of dicts {i, price, kind: 'H'|'L', confirmed}. The last pivot is
    the still-running extreme and is marked unconfirmed."""
    hi, lo = df["High"].values, df["Low"].values
    piv, trend = [], None
    cand_hi, cand_lo = (0, hi[0]), (0, lo[0])
    ext = None
    for i in range(1, len(df)):
        if trend is None:
            if hi[i] > cand_hi[1]: cand_hi = (i, hi[i])
            if lo[i] < cand_lo[1]: cand_lo = (i, lo[i])
            if cand_hi[1] >= cand_lo[1] * (1 + pct) and cand_lo[0] < cand_hi[0]:
                piv.append({"i": cand_lo[0], "price": float(cand_lo[1]), "kind": "L"})
                trend, ext = "up", cand_hi
            elif cand_lo[1] <= cand_hi[1] * (1 - pct) and cand_hi[0] < cand_lo[0]:
                piv.append({"i": cand_hi[0], "price": float(cand_hi[1]), "kind": "H"})
                trend, ext = "down", cand_lo
        elif trend == "up":
            if hi[i] > ext[1]:
                ext = (i, hi[i])
            elif lo[i] <= ext[1] * (1 - pct):
                piv.append({"i": ext[0], "price": float(ext[1]), "kind": "H"})
                trend, ext = "down", (i, lo[i])
        else:
            if lo[i] < ext[1]:
                ext = (i, lo[i])
            elif hi[i] >= ext[1] * (1 + pct):
                piv.append({"i": ext[0], "price": float(ext[1]), "kind": "L"})
                trend, ext = "up", (i, hi[i])
    for p in piv:
        p["confirmed"] = True
    if ext is not None:
        piv.append({"i": ext[0], "price": float(ext[1]), "kind": "H" if trend == "up" else "L",
                    "confirmed": False})
    return piv


# ── Pattern helpers ────────────────────────────────────────────────────────────
def _pattern(kind, cat, bias, pivots, df, status, lines, target=None, note=""):
    d = df.index
    return {
        "type": kind, "cat": cat, "bias": bias, "status": status,
        "open": not (status.startswith("confirmed") or status.startswith("broke")
                     or status in ("failed", "expired")),
        "start": ts(d[pivots[0]["i"]]), "end": ts(d[pivots[-1]["i"]]),
        "i0": int(pivots[0]["i"]), "i1": int(pivots[-1]["i"]),
        "pivots": [{"date": ts(d[p["i"]]), "price": round(p["price"], 2)} for p in pivots],
        "lines": lines,       # [[date0, price0, date1, price1], ...]
        "target": None if target is None else round(float(target), 2),
        "note": note,
    }


def _seg(df, i0, p0, i1, p1):
    return [ts(df.index[i0]), round(float(p0), 2), ts(df.index[i1]), round(float(p1), 2)]


def _slope_word(m, price, flat):
    s = m / price
    return "horizontal" if abs(s) < flat else ("rising" if s > 0 else "falling")


# ── Reversal patterns ──────────────────────────────────────────────────────────
def _resolve(df, i, neck, top, invalid, max_wait):
    """Walk the bars after pivot i. Returns (outcome, bar):
    'confirmed' close through the neckline, 'failed' price went past `invalid` first,
    'expired' nothing happened within max_wait bars, 'open' still undecided."""
    c, h, l = df["Close"].values, df["High"].values, df["Low"].values
    for k in range(i + 1, len(c)):
        if k - i > max_wait:
            return "expired", k
        if (h[k] > invalid) if top else (l[k] < invalid):
            return "failed", k
        if (c[k] < neck(k)) if top else (c[k] > neck(k)):
            return "confirmed", k
    return "open", None


def _status(df, outcome, k, running, what):
    if outcome == "confirmed":
        return f"confirmed {ts(df.index[k])}"
    if outcome == "open":
        return f"possible ({what} still forming)" if running else "forming (neckline not broken)"
    return outcome


def find_double(df, piv, tol, min_bars=10):
    out, last = [], len(df) - 1
    for j in range(1, len(piv) - 2):
        pre, a, m, b = piv[j - 1], piv[j], piv[j + 1], piv[j + 2]
        if a["kind"] != b["kind"] or b["i"] - a["i"] < min_bars:
            continue
        avg = (a["price"] + b["price"]) / 2
        if abs(a["price"] - b["price"]) / avg > tol:
            continue
        top = a["kind"] == "H"
        neck = m["price"]
        # needs a prior trend into the pattern: rally before a top, decline before a bottom
        if (pre["price"] >= neck) if top else (pre["price"] <= neck):
            continue
        ext = max(a["price"], b["price"]) if top else min(a["price"], b["price"])
        invalid = ext * (1 + tol / 2) if top else ext * (1 - tol / 2)
        outcome, k = _resolve(df, b["i"], lambda _: neck, top, invalid,
                              max(15, b["i"] - a["i"]))
        status = _status(df, outcome, k, not b["confirmed"], "2nd peak" if top else "2nd trough")
        height = abs(avg - neck)
        out.append(_pattern(
            "Double top" if top else "Double bottom", "reversal", "bearish" if top else "bullish",
            [a, m, b], df, status,
            [_seg(df, a["i"], neck, k if k is not None else last, neck)],
            neck - height if top else neck + height,
            f"peaks {abs(a['price']-b['price'])/avg:.1%} apart, {b['i']-a['i']} bars, neckline {neck:.2f}"))
    return out


def find_head_shoulders(df, piv, tol, min_bars=15):
    out, last = [], len(df) - 1
    for j in range(1, len(piv) - 4):
        pre = piv[j - 1]
        ls, n1, hd, n2, rs = piv[j:j + 5]
        if not (ls["kind"] == hd["kind"] == rs["kind"]) or rs["i"] - ls["i"] < min_bars:
            continue
        top = hd["kind"] == "H"
        sgn = 1 if top else -1
        # head must stand out from both shoulders; shoulders roughly level
        if sgn * (hd["price"] - max(ls["price"], rs["price"], key=lambda x: sgn * x)) <= 0:
            continue
        sh_avg = (ls["price"] + rs["price"]) / 2
        if abs(ls["price"] - rs["price"]) / sh_avg > tol * 1.5:
            continue
        if sgn * (hd["price"] - sh_avg) / sh_avg < tol * 0.5:
            continue
        # prior trend into the pattern
        if sgn * (pre["price"] - min(n1["price"], n2["price"], key=lambda x: sgn * x)) >= 0:
            continue
        slope = (n2["price"] - n1["price"]) / (n2["i"] - n1["i"])
        neck = lambda k, n1=n1, s=slope: n1["price"] + s * (k - n1["i"])
        outcome, k = _resolve(df, rs["i"], neck, top, hd["price"],
                              max(15, (rs["i"] - ls["i"]) // 2))
        status = _status(df, outcome, k, not rs["confirmed"], "right shoulder")
        end_i = k if k is not None else last
        height = abs(hd["price"] - neck(hd["i"]))
        target = neck(end_i) - height if top else neck(end_i) + height
        out.append(_pattern(
            "Head & shoulders" if top else "Inverse head & shoulders", "reversal",
            "bearish" if top else "bullish", [ls, n1, hd, n2, rs], df, status,
            [_seg(df, ls["i"], neck(ls["i"]), end_i, neck(end_i))], target,
            f"{rs['i']-ls['i']} bars, neckline slope {slope / neck(end_i) * 100:+.2f}%/bar"))
    return out


# ── Triangles / wedges / short channels (5-pivot windows) ──────────────────────
def _fit(points):
    x = np.array([p["i"] for p in points], float)
    y = np.array([p["price"] for p in points], float)
    m, b = np.polyfit(x, y, 1)
    resid = np.max(np.abs(y - (m * x + b)) / y)
    return m, b, resid


def classify_lines(mu, ml, price, flat):
    """Name the formation from upper/lower line slopes (in price units per bar)."""
    su, sl = mu / price, ml / price          # fraction per bar
    fu, fl = abs(su) < flat, abs(sl) < flat
    if fu and fl:                   return "Rectangle", "neutral"
    if fu and sl > 0:               return "Ascending triangle", "bullish"
    if fl and su < 0:               return "Descending triangle", "bearish"
    if su < 0 < sl:                 return "Symmetrical triangle", "neutral"
    if su > 0 > sl:                 return "Broadening formation", "neutral"
    para = abs(su - sl) < flat
    if su > 0 and sl > 0:
        if para:                    return "Rising channel", "bullish"
        return ("Rising wedge", "bearish") if sl > su else ("Broadening (rising)", "neutral")
    if su < 0 and sl < 0:
        if para:                    return "Falling channel", "bearish"
        return ("Falling wedge", "bullish") if su < sl else ("Broadening (falling)", "neutral")
    return None, None


def find_line_patterns(df, piv, tol, flat, window=5, min_bars=20):
    """Slide a window of `window` alternating pivots (>=2 highs and >=2 lows), fit
    trendlines to highs and lows, and classify. Non-overlapping, latest first."""
    out, j, last = [], len(piv) - window, len(df) - 1
    c = df["Close"].values
    while j >= 0:
        w = piv[j:j + window]
        hs = [p for p in w if p["kind"] == "H"]
        ls = [p for p in w if p["kind"] == "L"]
        ok = False
        if len(hs) >= 2 and len(ls) >= 2:
            mu, bu, ru = _fit(hs)
            ml, bl, rl = _fit(ls)
            price = np.mean([p["price"] for p in w])
            if max(ru, rl) <= tol * 0.6:
                name, bias = classify_lines(mu, ml, price, flat)
                i0, i1 = w[0]["i"], w[-1]["i"]
                if name and i1 - i0 >= min_bars:
                    upper = lambda k: mu * k + bu
                    lower = lambda k: ml * k + bl
                    # breakout: first close clearly outside the lines, within max_wait bars
                    max_wait = max(15, i1 - i0)
                    k, d = None, None
                    for x in range(i1 + 1, min(len(c), i1 + max_wait + 1)):
                        if c[x] > upper(x) * (1 + tol / 3): k, d = x, "up"; break
                        if c[x] < lower(x) * (1 - tol / 3): k, d = x, "down"; break
                    if k is not None:
                        status, end_i = f"broke {d} {ts(df.index[k])}", k
                    elif last - i1 > max_wait:
                        status, end_i = "expired", i1 + max_wait
                    else:
                        status, end_i = "active (price inside lines)", last
                    # stop drawing lines past where they cross (triangle apex)
                    if abs(mu - ml) > 1e-12:
                        apex = (bl - bu) / (mu - ml)
                        if i1 < apex < end_i:
                            end_i = int(apex)
                    target = None
                    if k is not None and not name.startswith("Broadening"):
                        if "channel" in name or name == "Rectangle":
                            height = upper(k) - lower(k)          # channel width
                        else:
                            height = abs(hs[0]["price"] - ls[0]["price"])   # base of triangle/wedge
                        target = c[k] + height if d == "up" else c[k] - height
                    cat = "channel" if ("channel" in name or name == "Rectangle") else "triangle"
                    out.append(_pattern(name, cat, bias, w, df, status,
                        [_seg(df, i0, upper(i0), end_i, upper(end_i)),
                         _seg(df, i0, lower(i0), end_i, lower(end_i))], target,
                        f"{i1-i0} bars, upper {mu/price*100:+.2f}%/bar, lower {ml/price*100:+.2f}%/bar"))
                    ok = True
        j = j - (window - 1) if ok else j - 1
    return out


# ── Trendlines and parallel channels ───────────────────────────────────────────
def _trendline_candidates(df, piv, tol, min_touches=3, min_bars=20):
    """Every line through two same-side pivots that (a) price respected between
    them and (b) 3+ pivots touch. Break = first close beyond the line by tol/2."""
    c = df["Close"].values
    n = len(c)
    touch_tol = tol * 0.5
    cands = []
    for kind in ("L", "H"):
        support = kind == "L"
        pts = [p for p in piv if p["kind"] == kind]
        for a in range(len(pts)):
            p = pts[a]
            ks = np.arange(p["i"] + 1, n)
            for b in range(a + 1, len(pts)):
                q = pts[b]
                if q["i"] - p["i"] < 5:
                    continue
                m = (q["price"] - p["price"]) / (q["i"] - p["i"])
                line = p["price"] + m * (ks - p["i"])
                if np.any(line <= 0):
                    continue
                beyond = (c[p["i"] + 1:] < line * (1 - touch_tol)) if support \
                    else (c[p["i"] + 1:] > line * (1 + touch_tol))
                hit = np.flatnonzero(beyond)
                brk = int(ks[hit[0]]) if hit.size else None
                if brk is not None and brk <= q["i"]:
                    continue                      # violated before the 2nd touch
                end = brk if brk is not None else n - 1
                touches = [r for r in pts[a:] if r["i"] <= end and
                           abs(r["price"] - (p["price"] + m * (r["i"] - p["i"]))) / r["price"] <= touch_tol]
                if len(touches) < min_touches or touches[-1]["i"] - touches[0]["i"] < min_bars:
                    continue
                cands.append({"support": support, "m": m, "p": p, "touches": touches,
                              "brk": brk, "end": end})
    return cands


def _select_lines(cands, limit):
    """Greedy: most touches first (then most recent); drop lines sharing 2+ touch pivots."""
    cands = sorted(cands, key=lambda t: (len(t["touches"]), t["touches"][-1]["i"]), reverse=True)
    chosen = []
    for t in cands:
        ids = {r["i"] for r in t["touches"]}
        if all(len(ids & {r["i"] for r in s["touches"]}) < 2 for s in chosen):
            chosen.append(t)
        if len(chosen) >= limit:
            break
    return chosen


def find_trendlines(df, piv, tol, flat, cands=None, limit=8):
    out = []
    cands = cands if cands is not None else _trendline_candidates(df, piv, tol)
    for t in _select_lines(cands, limit):
        p, m, tch = t["p"], t["m"], t["touches"]
        line = lambda k, p=p, m=m: p["price"] + m * (k - p["i"])
        word = _slope_word(m, p["price"], flat)
        role = "support" if t["support"] else "resistance"
        bias = ("bullish" if word == "rising" else "neutral") if t["support"] \
            else ("bearish" if word == "falling" else "neutral")
        if t["brk"] is not None:
            status = f"broke {'down' if t['support'] else 'up'} {ts(df.index[t['brk']])}"
        elif abs(df["Close"].iloc[-1] - line(t["end"])) / df["Close"].iloc[-1] > 0.15:
            status = "expired"                   # never broken, but price has moved far away
        else:
            status = "active"
        out.append(_pattern(
            f"{word.capitalize()} {role} trendline", "trendline", bias, tch, df, status,
            [_seg(df, tch[0]["i"], line(tch[0]["i"]), t["end"], line(t["end"]))], None,
            f"{len(tch)} touches over {tch[-1]['i']-tch[0]['i']} bars, {m/p['price']*100:+.2f}%/bar"
            + (f", now at {line(t['end']):.2f}" if t["brk"] is None else "")))
    return out


def find_parallel_channels(df, piv, tol, flat, cands, limit=4):
    """A trendline plus a parallel line through the opposite-side pivots, which
    must be touched at least twice. Channel ends where price closes outside either line."""
    c = df["Close"].values
    n, touch_tol = len(c), tol * 0.5
    found = []
    for t in cands:
        p, m, tch = t["p"], t["m"], t["touches"]
        line = lambda k, p=p, m=m: p["price"] + m * (k - p["i"])
        i0, end = tch[0]["i"], t["end"]
        opp = [r for r in piv if r["kind"] == ("H" if t["support"] else "L") and i0 <= r["i"] <= end]
        if len(opp) < 2:
            continue
        # offset of the parallel line: the furthest opposite pivot
        offs = [r["price"] - line(r["i"]) for r in opp]
        off = max(offs) if t["support"] else min(offs)
        if abs(off) / p["price"] < tol * 1.5:
            continue                              # too narrow to be a channel
        par = lambda k, line=line, off=off: line(k) + off
        ptch = [r for r in opp if abs(r["price"] - par(r["i"])) / r["price"] <= touch_tol]
        if len(ptch) < 2:
            continue
        # break of the parallel side ends the channel too (base-line break is t["brk"])
        last_touch = max(tch[-1]["i"], ptch[-1]["i"])
        ks = np.arange(last_touch + 1, n)
        other = (c[last_touch + 1:] > (line(ks) + off) * (1 + touch_tol)) if t["support"] \
            else (c[last_touch + 1:] < (line(ks) + off) * (1 - touch_tol))
        hit = np.flatnonzero(other)
        brk_par = int(ks[hit[0]]) if hit.size else None
        brks = [(b, d) for b, d in [(t["brk"], "down" if t["support"] else "up"),
                                    (brk_par, "up" if t["support"] else "down")] if b is not None]
        brk, d = min(brks) if brks else (None, None)
        found.append({"t": t, "line": line, "off": off, "ptch": ptch, "brk": brk, "dir": d,
                      "score": len(tch) + len(ptch)})
    found.sort(key=lambda f: (f["score"], f["t"]["touches"][-1]["i"]), reverse=True)

    out, used = [], []
    for f in found:
        t = f["t"]
        ids = {r["i"] for r in t["touches"] + f["ptch"]}
        if any(len(ids & u) >= 3 for u in used):
            continue
        used.append(ids)
        p, m, line, off = t["p"], t["m"], f["line"], f["off"]
        pts = sorted(t["touches"] + f["ptch"], key=lambda r: r["i"])
        i0 = pts[0]["i"]
        end = f["brk"] if f["brk"] is not None else n - 1
        word = _slope_word(m, p["price"], flat)
        bias = {"rising": "bullish", "falling": "bearish"}.get(word, "neutral")
        status = f"broke {f['dir']} {ts(df.index[f['brk']])}" if f["brk"] is not None else "active"
        lo_off, hi_off = (0, off) if t["support"] else (off, 0)
        width = abs(off) / line(end) * 100
        out.append(_pattern(
            f"Parallel channel ({word})", "channel", bias, pts, df, status,
            [_seg(df, i0, line(i0) + lo_off, end, line(end) + lo_off),
             _seg(df, i0, line(i0) + hi_off, end, line(end) + hi_off)], None,
            f"{len(t['touches'])}+{len(f['ptch'])} touches, width {width:.1f}%, "
            f"{m/p['price']*100:+.2f}%/bar"
            + (f", now {line(end)+lo_off:.2f}-{line(end)+hi_off:.2f}" if f["brk"] is None else "")))
        if len(out) >= limit:
            break
    return out


# ── Consolidation ranges (boxes) and Wyckoff candidates ───────────────────────
BOX_ATR = 4.0       # a range: closes stay within 4 x ATR ...
BOX_MIN_BARS = 15   # ... for at least 15 bars
WYCKOFF_MIN_BARS = 20


def _atr_series(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean().values


def _find_boxes(c, atr):
    """Greedy scan: from each start, extend while the closing range stays within
    BOX_ATR x ATR(start); keep runs of at least BOX_MIN_BARS bars, non-overlapping."""
    n, out, i = len(c), [], 0
    while i < n - BOX_MIN_BARS:
        lim, hi, lo, j = BOX_ATR * atr[i], c[i], c[i], i
        while j + 1 < n and max(hi, c[j + 1]) - min(lo, c[j + 1]) <= lim:
            j += 1
            hi, lo = max(hi, c[j]), min(lo, c[j])
        if j - i + 1 >= BOX_MIN_BARS and hi > lo:
            out.append((i, j, float(hi), float(lo)))
            i = j + 1
        else:
            i += 1
    return out


def _touches(x, i, j, level, above, gap=3):
    """Separate tests of a level (bars at/through it, at least `gap` bars apart)."""
    n, last = 0, -gap - 1
    for k in range(i, j + 1):
        if (x[k] >= level) if above else (x[k] <= level):
            if k - last > gap:
                n += 1
            last = k
    return n


def _box_dict(kind, cat, bias, df, i, j, end, hi, lo, status, is_open, target, note, events=None):
    d = df.index
    return {
        "type": kind, "cat": cat, "bias": bias, "status": status, "open": is_open,
        "start": ts(d[i]), "end": ts(d[j]), "i0": int(i), "i1": int(j), "pivots": [],
        "lines": [_seg(df, i, hi, end, hi), _seg(df, i, lo, end, lo)],
        "box": [ts(d[i]), ts(d[j]), round(lo, 2), round(hi, 2)],
        "events": events or [],
        "target": None if target is None else round(float(target), 2), "note": note,
    }


def find_ranges(df):
    """Consolidation boxes plus Wyckoff accumulation / distribution candidates."""
    c, h, l = df["Close"].values, df["High"].values, df["Low"].values
    v = df["Volume"].values.astype(float)
    has_vol = v.sum() > 0
    atr = _atr_series(df)
    n, last, idx = len(c), len(c) - 1, df.index
    vavg = lambda a, b: float(v[max(a, 0):b].mean()) if b > max(a, 0) else float("nan")
    out = []
    for i, j, hi, lo in _find_boxes(c, atr):
        a = atr[i]
        brk = j + 1 if j + 1 < n else None
        d = None if brk is None else ("up" if c[brk] > hi else "down")
        back = None                       # false break: a close back inside within 3 bars
        if brk is not None:
            back = next((k for k in range(brk + 1, min(n, brk + 4)) if lo <= c[k] <= hi), None)
        if brk is None:
            status, end = "active (inside range)", last
        elif back is not None:
            status, end = f"false break {d} {ts(idx[brk])}, back inside {ts(idx[back])}", back
        else:
            status, end = f"broke {d} {ts(idx[brk])}", brk
        width = hi - lo
        t_top, t_bot = _touches(h, i, j, hi - 0.25 * a, True), _touches(l, i, j, lo + 0.25 * a, False)
        drift = abs(np.polyfit(np.arange(j - i + 1), c[i:j + 1], 1)[0]) * (j - i)
        if t_top < 2 or t_bot < 2 or drift > 0.5 * width:
            continue                      # a slow trend or a one-sided drift, not a range
        box_vol = vavg(i, j + 1)
        vr = box_vol / vavg(i - 20, i) if has_vol and i >= 5 else float("nan")
        note = (f"{j - i + 1} bars, closing range {lo:.2f}-{hi:.2f} ({width / lo * 100:.1f}%, {width / a:.1f}x ATR), "
                f"top tested {t_top}x, bottom {t_bot}x"
                + (f", volume {vr:.2f}x the prior 20 bars" if vr == vr else ""))
        target = None if brk is None or back is not None else (hi + width if d == "up" else lo - width)
        out.append(_box_dict("Consolidation range", "range", "neutral", df, i, j, end, hi, lo,
                             status, brk is None, target, note))

        # ── Wyckoff: a range that follows a clear trend ──
        if j - i + 1 < WYCKOFF_MIN_BARS or i < 10:
            continue
        look = max(0, i - 60)             # trend into the range: move from the recent high / low
        down, up = c[i] / c[look:i].max() - 1, c[i] / c[look:i].min() - 1
        prior = up if up >= -down else down
        if abs(prior) < 0.75 * width / lo:
            continue                      # no clear trend into the range
        acc = prior < 0
        ev, why = [], []
        # climax: wide, high-volume bar at the extreme around the start of the range
        clim = None
        for k in range(max(1, i - 10), min(j, i + 10) + 1):
            v20 = vavg(k - 20, k)
            wide = h[k] - l[k] >= 1.8 * atr[k]
            loud = (not has_vol) or (v20 > 0 and v[k] >= 1.8 * v20)
            near = (l[k] <= lo + 0.5 * a) if acc else (h[k] >= hi - 0.5 * a)
            if wide and loud and near and (clim is None or (l[k] < l[clim] if acc else h[k] > h[clim])):
                clim = k
        if clim is not None:
            vx = v[clim] / vavg(clim - 20, clim) if has_vol else None
            ev.append({"date": ts(idx[clim]), "price": round(float(l[clim] if acc else h[clim]), 2),
                       "label": "SC" if acc else "BC"})
            why.append(f"{'selling' if acc else 'buying'} climax {ts(idx[clim])}" + (f" (volume {vx:.1f}x)" if vx else ""))
            seg = range(clim + 1, min(j, clim + 10) + 1)
            if len(seg):
                ar = max(seg, key=lambda k: h[k]) if acc else min(seg, key=lambda k: l[k])
                ev.append({"date": ts(idx[ar]), "price": round(float(h[ar] if acc else l[ar]), 2), "label": "AR"})
        # spring (accumulation) / upthrust (distribution): poke through the edge, close back inside
        trap = None
        for k in range(i + (j - i + 1) // 3, min(n, j + 4)):
            poke = (l[k] < lo - 0.25 * a) if acc else (h[k] > hi + 0.25 * a)
            if poke and any((c[m] >= lo) if acc else (c[m] <= hi) for m in range(k, min(n, k + 4))):
                if trap is None or (l[k] < l[trap] if acc else h[k] > h[trap]):
                    trap = k
        if trap is not None:
            vt = v[trap] / box_vol if has_vol and box_vol else None
            lbl = "Spring" if acc else ("UTAD" if trap > i + (j - i) // 2 else "UT")
            ev.append({"date": ts(idx[trap]), "price": round(float(l[trap] if acc else h[trap]), 2), "label": lbl})
            why.append(f"{lbl.lower() if lbl == 'Spring' else lbl} {ts(idx[trap])}"
                       + (f" on {'light' if vt < 1 else 'heavy'} volume ({vt:.1f}x range avg)" if vt else ""))
        # breakout in the Wyckoff direction (sign of strength / weakness) and its retest
        phase, status, is_open = ("C" if trap is not None else "B"), None, brk is None
        if brk is not None and back is None:
            if (d == "up") == acc:
                vb = v[brk] / box_vol if has_vol and box_vol else None
                strong = vb is None or vb >= 1.3
                ev.append({"date": ts(idx[brk]), "price": round(float(c[brk]), 2), "label": "SOS" if acc else "SOW"})
                why.append(f"{'SOS breakout' if acc else 'SOW breakdown'} {ts(idx[brk])}"
                           + (f" on {'expanding' if strong else 'weak'} volume ({vb:.1f}x)" if vb else ""))
                seg = list(range(brk + 1, min(n, brk + 16)))
                if seg:
                    rt = min(seg, key=lambda k: l[k]) if acc else max(seg, key=lambda k: h[k])
                    held = (l[rt] >= hi - 0.5 * a) if acc else (h[rt] <= lo + 0.5 * a)
                    if held and rt < last:
                        ev.append({"date": ts(idx[rt]), "price": round(float(l[rt] if acc else h[rt]), 2),
                                   "label": "LPS" if acc else "LPSY"})
                        why.append(f"{'LPS' if acc else 'LPSY'} retest held {ts(idx[rt])}")
                after = c[brk:].max() if acc else c[brk:].min()
                phase = "E" if ((after >= hi + width) if acc else (after <= lo - width)) else "D"
                status = f"confirmed {ts(idx[brk])} ({'SOS breakout' if acc else 'SOW breakdown'}, Phase {phase})"
            else:
                status = f"failed: broke {d} {ts(idx[brk])}"
                why.append("range broke the other way" + (" (more likely re-accumulation)" if not acc else " (more likely re-distribution)"))
        if status is None:
            status = f"Phase {phase}, {'still in range' if brk is None else 'back in range after a false break'}"
        name = "Wyckoff accumulation (candidate)" if acc else "Wyckoff distribution (candidate)"
        note = f"Phase {phase} · prior trend {prior * 100:+.0f}% · " + (" · ".join(why) if why else "no climax / spring / upthrust found yet")
        out.append(_box_dict(name, "wyckoff", "bullish" if acc else "bearish", df, i, j,
                             end, hi, lo, status, is_open, None, note, ev))
    return out


# ── Volume climaxes (capitulation) ─────────────────────────────────────────────
VOL_SPIKE = 2.0     # volume at least 2x the average of the previous 20 bars


def _event_dict(kind, cat, bias, df, k, k_end, status, is_open, note, events):
    d = df.index
    return {"type": kind, "cat": cat, "bias": bias, "status": status, "open": is_open,
            "start": ts(d[k]), "end": ts(d[k_end]), "i0": int(k), "i1": int(k_end), "pivots": [],
            "lines": [], "events": events, "target": None, "note": note}


def find_volume_events(df):
    """Volume spikes; a spike on a wide bar at the end of a slide (rally) is a selling
    capitulation (buying climax). It is a confirmed turning point when, within 5 bars,
    volume is back to normal and price reverses without breaking the spike's extreme."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].values.astype(float)
    if v.sum() == 0:
        return []
    atr = _atr_series(df)
    avg = pd.Series(v).rolling(20).mean().shift(1).values      # previous 20 bars
    n, last, idx, out = len(c), len(c) - 1, df.index, []
    for k in range(20, n):
        if not avg[k] or v[k] < VOL_SPIKE * avg[k]:
            continue
        ratio, wide = v[k] / avg[k], (h[k] - l[k]) / atr[k]
        lo15, hi15 = l[max(0, k - 15):k + 1].min(), h[max(0, k - 15):k + 1].max()
        slide = c[k] / c[max(0, k - 15):k].max() - 1 if k else 0
        rally = c[k] / c[max(0, k - 15):k].min() - 1 if k else 0
        atrp = atr[k] / c[k]
        sell = wide >= 1.3 and l[k] <= lo15 * 1.002 and slide <= -3 * atrp
        buy = wide >= 1.3 and h[k] >= hi15 * 0.998 and rally >= 3 * atrp
        seg = range(k + 1, min(n, k + 6))
        calm = next((m for m in seg if avg[m] and v[m] < 1.2 * avg[m]), None)
        if sell or buy:
            ext = l[k] if sell else h[k]
            broke = next((m for m in seg if ((c[m] < ext) if sell else (c[m] > ext))), None)
            turn = next((m for m in seg if ((c[m] > h[k]) if sell else (c[m] < l[k]))), None)
            if broke is not None and (turn is None or broke < turn):
                status, is_open = f"failed: {'low' if sell else 'high'} broken {ts(idx[broke])}", False
            elif turn is not None and calm is not None:
                status, is_open = f"confirmed {ts(idx[max(turn, calm)])} (turning point)", False
            elif k > last - 5:
                status, is_open = "pending (watch the next sessions)", True
            else:
                status, is_open = "unconfirmed (no clear reversal)", False
            kind = "Selling capitulation" if sell else "Buying climax (blow-off)"
            note = (f"volume {ratio:.1f}x the 20-bar average, range {wide:.1f}x ATR, after a "
                    f"{(slide if sell else rally) * 100:+.1f}% move"
                    + (f"; volume back to normal {ts(idx[calm])}" if calm is not None else "")
                    + (f"; close beyond the spike bar {ts(idx[turn])}" if turn is not None else ""))
            out.append(_event_dict(kind, "volume", "bullish" if sell else "bearish", df, k,
                                   turn if turn is not None else k, status, is_open, note,
                                   [{"date": ts(idx[k]), "price": round(float(ext), 2),
                                     "label": f"{'Capitulation' if sell else 'Climax'} {ratio:.1f}x"}]))
        else:
            up = c[k] >= o[k]
            out.append(_event_dict("Volume spike", "volume", "neutral", df, k, k,
                                   f"{'up' if up else 'down'} day on {ratio:.1f}x volume", False,
                                   f"volume {ratio:.1f}x the 20-bar average, {'up' if up else 'down'} "
                                   f"{(c[k] / c[k - 1] - 1) * 100:+.1f}%, range {wide:.1f}x ATR"
                                   + (f"; back to normal {ts(idx[calm])}" if calm is not None else ""),
                                   [{"date": ts(idx[k]), "price": round(float(h[k] if up else l[k]), 2),
                                     "label": f"Vol {ratio:.1f}x"}]))
    return out


# ── Candlestick patterns (in context, confirmed) ──────────────────────────────
def find_candles(df):
    """Classic reversal candles that appear after a matching move (5-bar change of at
    least 1 ATR) and are confirmed by a close beyond the pattern's high (bullish) or
    low (bearish) within the next 2 bars. Unconfirmed ones are dropped, except on the
    last 2 bars where they are shown as awaiting confirmation."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    atr = _atr_series(df)
    n, last, idx, out = len(c), len(c) - 1, df.index, []
    body = abs(c - o)
    up_w = h - np.maximum(o, c)
    lo_w = np.minimum(o, c) - l
    for k in range(8, n):
        a = atr[k]
        down = c[k - 1] - c[k - 6] <= -a      # prior move into the candle(s)
        upm = c[k - 1] - c[k - 6] >= a
        found = []                             # (name, bullish, first bar of pattern)
        rng = h[k] - l[k]
        if rng >= 0.8 * a:
            small_top, small_bot = up_w[k] <= 0.35 * max(body[k], 0.1 * rng), lo_w[k] <= 0.35 * max(body[k], 0.1 * rng)
            if lo_w[k] >= 2 * body[k] and small_top and lo_w[k] >= 0.55 * rng:
                if down: found.append(("Hammer", True, k))
                if upm: found.append(("Hanging man", False, k))
            if up_w[k] >= 2 * body[k] and small_bot and up_w[k] >= 0.55 * rng:
                if down: found.append(("Inverted hammer", True, k))
                if upm: found.append(("Shooting star", False, k))
        b0, b1 = c[k - 1] - o[k - 1], c[k] - o[k]
        if down and b0 < 0 and b1 > 0 and o[k] <= c[k - 1] and c[k] >= o[k - 1] and body[k] > body[k - 1]:
            found.append(("Bullish engulfing", True, k - 1))
        if upm and b0 > 0 and b1 < 0 and o[k] >= c[k - 1] and c[k] <= o[k - 1] and body[k] > body[k - 1]:
            found.append(("Bearish engulfing", False, k - 1))
        if down and b0 < 0 and body[k - 1] >= 0.6 * a and b1 > 0 and o[k] < c[k - 1] and (o[k - 1] + c[k - 1]) / 2 < c[k] < o[k - 1]:
            found.append(("Piercing line", True, k - 1))
        if upm and b0 > 0 and body[k - 1] >= 0.6 * a and b1 < 0 and o[k] > c[k - 1] and o[k - 1] < c[k] < (o[k - 1] + c[k - 1]) / 2:
            found.append(("Dark cloud cover", False, k - 1))
        b2 = c[k - 2] - o[k - 2]
        down2, up2 = c[k - 2] - c[k - 7] <= -a, c[k - 2] - c[k - 7] >= a
        if down2 and b2 < 0 and body[k - 2] >= 0.6 * a and body[k - 1] <= 0.35 * body[k - 2] and b1 > 0 and c[k] > (o[k - 2] + c[k - 2]) / 2:
            found.append(("Morning star", True, k - 2))
        if up2 and b2 > 0 and body[k - 2] >= 0.6 * a and body[k - 1] <= 0.35 * body[k - 2] and b1 < 0 and c[k] < (o[k - 2] + c[k - 2]) / 2:
            found.append(("Evening star", False, k - 2))
        three = range(k - 2, k + 1)
        if all(c[m] > o[m] and body[m] >= 0.5 * a and h[m] - c[m] <= 0.3 * (h[m] - l[m]) for m in three) and c[k - 2] < c[k - 1] < c[k] and down2:
            found.append(("Three white soldiers", True, k - 2))
        if all(c[m] < o[m] and body[m] >= 0.5 * a and c[m] - l[m] <= 0.3 * (h[m] - l[m]) for m in three) and c[k - 2] > c[k - 1] > c[k] and up2:
            found.append(("Three black crows", False, k - 2))
        for name, bull, k0 in found:
            ph, pl = h[k0:k + 1].max(), l[k0:k + 1].min()
            conf = next((m for m in range(k + 1, min(n, k + 3)) if ((c[m] > ph) if bull else (c[m] < pl))), None)
            if conf is None and k <= last - 2:
                continue                       # never confirmed: dropped
            swing = (l[k0:k + 1].min() <= l[max(0, k - 10):min(n, k + 6)].min()) if bull else \
                    (h[k0:k + 1].max() >= h[max(0, k - 10):min(n, k + 6)].max())
            status = f"confirmed {ts(idx[conf])}" if conf is not None else "awaiting confirmation"
            note = (f"after a {(c[k - 1] / c[k - 6] - 1) * 100:+.1f}% 5-bar move; confirms on a close "
                    f"{'above' if bull else 'below'} {ph if bull else pl:.2f}" + ("; at a swing " + ("low" if bull else "high") if swing else ""))
            out.append(_event_dict(name, "candle", "bullish" if bull else "bearish", df, k0,
                                   conf if conf is not None else k, status, conf is None, note,
                                   [{"date": ts(idx[k]), "price": round(float(pl if bull else ph), 2),
                                     "label": "".join(w[0] for w in name.split()).upper()}]))
    return out


# ── Plain-language read of the last N bars ─────────────────────────────────────
READ_BARS = 30


def read_recent(df, pats, piv, n=READ_BARS):
    """Observations about the last n bars, each with a sign (+1 good / -1 bad / 0 neutral)
    and the reason, plus an overall verdict. Rule-based, so every line traces to the data."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].values.astype(float)
    has_vol = v.sum() > 0
    N, idx = len(c), df.index
    s0 = max(1, N - n)                       # first bar of the window
    atr = _atr_series(df)
    vavg = pd.Series(v).rolling(20).mean().shift(1).values
    unit = "days" if (idx[-1].hour, idx[-1].minute) == (0, 0) else "bars"
    obs = []
    add = lambda sign, text, kind: obs.append({"sign": sign, "text": text, "kind": kind})

    # 1) performance and where it closed in the window's range
    chg = c[-1] / c[s0 - 1] - 1
    hi, lo = h[s0:].max(), l[s0:].min()
    pos = (c[-1] - lo) / (hi - lo) if hi > lo else 0.5
    add(1 if chg > 0.02 else -1 if chg < -0.02 else 0,
        f"{chg * 100:+.1f}% over the last {N - s0} {unit}; closed in the {'upper' if pos > 0.67 else 'lower' if pos < 0.33 else 'middle'} "
        f"third of its {lo:.2f}-{hi:.2f} range", "performance")

    # 2) structure: higher highs / higher lows from the swing points
    H = [q for q in piv if q["kind"] == "H"]
    L = [q for q in piv if q["kind"] == "L"]
    if len(H) >= 2 and len(L) >= 2:
        hh, hl = H[-1]["price"] > H[-2]["price"], L[-1]["price"] > L[-2]["price"]
        dH, dL = ts(idx[H[-1]["i"]]), ts(idx[L[-1]["i"]])
        if hh and hl:
            add(1, f"higher highs and higher lows (low {L[-2]['price']:.2f} → {L[-1]['price']:.2f} on {dL}): uptrend structure intact", "structure")
        elif not hh and not hl:
            add(-1, f"lower highs and lower lows (high {H[-2]['price']:.2f} → {H[-1]['price']:.2f} on {dH}): downtrend structure", "structure")
        elif hl and not hh:
            add(0, f"higher low ({L[-1]['price']:.2f} on {dL}) but a lower high ({H[-1]['price']:.2f}): range-bound / coiling", "structure")
        else:
            add(0, f"higher high ({H[-1]['price']:.2f}) but a lower low ({L[-1]['price']:.2f} on {dL}): widening swings", "structure")
    prior_low = l[max(0, s0 - n):s0].min() if s0 > 1 else None
    if prior_low is not None:
        if lo >= prior_low:
            add(1, f"no lower low: the window's low {lo:.2f} held above the previous {n}-{unit[:-1]} low {prior_low:.2f}", "structure")
        else:
            add(-1, f"made a lower low ({lo:.2f}) below the previous {n}-{unit[:-1]} low {prior_low:.2f}", "structure")

    # 1b) the latest bar and the last 5 bars (fresh moves the 30-bar view can hide)
    atrp = atr[-1] / c[-1]
    d1 = c[-1] / c[-2] - 1
    if abs(d1) >= atrp:
        vr = v[-1] / vavg[-1] if has_vol and vavg[-1] else None
        add(1 if d1 > 0 else -1, f"latest {unit[:-1]} ({ts(idx[-1])}) {d1 * 100:+.1f}%"
            + (f" on {vr:.1f}x average volume" if vr else "") + f", a move of {abs(d1) / atrp:.1f}x the usual range", "recent")
    hi5, lo5 = h[-6:].max(), l[-6:].min()
    if c[-1] / hi5 - 1 <= -2 * atrp:
        k = N - 6 + int(np.argmax(h[-6:]))
        add(-1, f"sharp pullback: {(c[-1] / hi5 - 1) * 100:.1f}% from the {ts(idx[k])} high {hi5:.2f} within 5 {unit}", "recent")
    elif c[-1] / lo5 - 1 >= 2 * atrp:
        k = N - 6 + int(np.argmin(l[-6:]))
        add(1, f"sharp rebound: {(c[-1] / lo5 - 1) * 100:+.1f}% from the {ts(idx[k])} low {lo5:.2f} within 5 {unit}", "recent")

    # 3) gaps (open beyond the prior close by >= 0.75 ATR), with volume and whether filled
    quiet_filled = 0
    for k in range(s0, N):
        g = o[k] - c[k - 1]
        if abs(g) < 0.75 * atr[k - 1]:
            continue
        up = g > 0
        vr = v[k] / vavg[k] if has_vol and vavg[k] else None
        filled = (l[k:].min() <= c[k - 1]) if up else (h[k:].max() >= c[k - 1])
        loud = vr is not None and vr >= 1.5
        if filled and not loud:
            quiet_filled += 1
            continue
        txt = (f"gap {'up' if up else 'down'} on {ts(idx[k])} ({g / c[k - 1] * 100:+.1f}%"
               + (f", volume {vr:.1f}x average" if vr else "") + "), "
               + ("filled since" if filled else "still unfilled"))
        if up:
            sign = 0 if filled else 1
            txt += " → buyers in control" if not filled and loud else (" → gap support below at " + f"{c[k - 1]:.2f}" if not filled else "")
        else:
            sign = 0 if filled else -1
            txt += " → sellers in control" if not filled and loud else (" → gap resistance above at " + f"{c[k - 1]:.2f}" if not filled else "")
        add(sign, txt, "gap")
    if quiet_filled:
        add(0, f"{quiet_filled} other gap{'s' if quiet_filled > 1 else ''} on normal volume already filled", "gap")

    # 4) volume: up-day vs down-day volume, biggest volume day
    if has_vol:
        upv = v[s0:][c[s0:] >= o[s0:]].sum()
        dnv = v[s0:][c[s0:] < o[s0:]].sum()
        if upv + dnv > 0:
            ratio = upv / dnv if dnv else float("inf")
            if ratio >= 1.3:
                add(1, f"more volume on up {unit} than down {unit} ({ratio:.1f}x): accumulation", "volume")
            elif ratio <= 1 / 1.3:
                add(-1, f"more volume on down {unit} than up {unit} ({1 / ratio:.1f}x): distribution", "volume")
            else:
                add(0, f"up- and down-{unit[:-1]} volume roughly balanced ({ratio:.1f}x)", "volume")
        k = s0 + int(np.argmax(v[s0:]))
        if vavg[k] and v[k] >= 1.8 * vavg[k]:
            up = c[k] >= o[k]
            add(1 if up else -1, f"heaviest volume on {ts(idx[k])} ({v[k] / vavg[k]:.1f}x average) was an "
                f"{'up' if up else 'down'} {unit[:-1]} ({(c[k] / c[k - 1] - 1) * 100:+.1f}%)", "volume")

    # 5) patterns, volume events and candles inside the window or still open
    t0 = ts(idx[s0])
    for pt in pats:
        sign = {"bullish": 1, "bearish": -1}.get(pt["bias"], 0)
        st = pt["status"]
        recent_event = pt["start"] >= t0 or pt["end"] >= t0 or any(d >= t0 for d in re.findall(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?", st))
        if pt["cat"] == "trendline":
            continue                      # trendlines are covered by levels; too many to list
        if pt["cat"] in ("volume", "candle"):
            if pt["start"] < t0 or pt["type"] == "Volume spike":
                continue
            if st.startswith("confirmed"):
                add(sign, f"{pt['type']} on {pt['start']}, {st}", pt["cat"])
            elif st.startswith("failed"):
                add(-sign, f"{pt['type']} on {pt['start']} {st}", pt["cat"])
            elif pt["open"]:
                add(0, f"{pt['type']} on {pt['start']}, {st}", pt["cat"])
            continue
        if pt["open"]:
            add(sign if pt["cat"] != "range" else 0, f"{pt['type']} ({pt['start']} → {pt['end']}): {st}", "formation")
        elif recent_event:
            broke_dir = "up" if " up " in f" {st} " else "down" if " down " in f" {st} " else None
            if st.startswith("confirmed"):
                add(sign, f"{pt['type']} {st}", "formation")
            elif st.startswith("failed"):
                add(-sign if sign else 0, f"{pt['type']} {st}", "formation")
            elif st.startswith("false break") and broke_dir:
                # a breakout that slipped back inside is a failed move: a warning in that direction
                add(-1 if broke_dir == "up" else 1, f"{pt['type']}: {st} (failed {'breakout' if broke_dir == 'up' else 'breakdown'})", "formation")
            elif broke_dir and pt["cat"] in ("range", "triangle", "channel"):
                add(1 if broke_dir == "up" else -1, f"{pt['type']} {st}", "formation")

    # 6) trend and momentum
    cs = pd.Series(c)
    e9, e21 = cs.ewm(span=9, adjust=False).mean(), cs.ewm(span=21, adjust=False).mean()
    s50 = cs.rolling(50).mean()
    above21, above50 = c[-1] > e21.iloc[-1], (not np.isnan(s50.iloc[-1])) and c[-1] > s50.iloc[-1]
    add(1 if above21 and above50 else -1 if not above21 and not above50 else 0,
        f"price {'above' if above21 else 'below'} the 21-{unit[:-1]} EMA ({e21.iloc[-1]:.2f})"
        + ("" if np.isnan(s50.iloc[-1]) else f" and {'above' if above50 else 'below'} the 50-{unit[:-1]} average ({s50.iloc[-1]:.2f})"), "trend")
    x = np.sign(e9 - e21).diff().fillna(0).values
    xk = [k for k in range(s0, N) if x[k] != 0]
    if xk:
        k = xk[-1]
        add(1 if x[k] > 0 else -1, f"EMA9 crossed {'above' if x[k] > 0 else 'below'} EMA21 on {ts(idx[k])}", "trend")
    d = cs.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + g / ls)).values
    if rsi[-1] >= 70:
        add(0, f"RSI {rsi[-1]:.0f}: overbought, stretched short term", "momentum")
    elif rsi[-1] <= 30:
        add(0, f"RSI {rsi[-1]:.0f}: oversold, a bounce is common but not guaranteed", "momentum")
    if c[-1] >= h[s0:].max() * 0.995 and rsi[-1] < rsi[s0:].max() - 5:
        add(-1, f"price near the window high but RSI {rsi[-1]:.0f} is below its peak {rsi[s0:].max():.0f}: bearish divergence", "momentum")
    if c[-1] <= l[s0:].min() * 1.005 and rsi[-1] > rsi[s0:].min() + 5:
        add(1, f"price near the window low but RSI {rsi[-1]:.0f} is above its low {rsi[s0:].min():.0f}: bullish divergence", "momentum")

    score = sum(o_["sign"] for o_ in obs)
    pos_n, neg_n = sum(o_["sign"] > 0 for o_ in obs), sum(o_["sign"] < 0 for o_ in obs)
    verdict = ("Bullish" if score >= 3 else "Leaning bullish" if score >= 1 else
               "Bearish" if score <= -3 else "Leaning bearish" if score <= -1 else "Mixed")
    return {"bars": N - s0, "from": t0, "verdict": verdict, "score": score,
            "positives": int(pos_n), "negatives": int(neg_n), "observations": obs}


# ── Support / resistance ───────────────────────────────────────────────────────
def find_levels(piv, price, tol, min_touches=3):
    """Cluster pivot prices; keep clusters touched >= min_touches times."""
    pts = sorted(p["price"] for p in piv)
    clusters, cur = [], [pts[0]] if pts else []
    for x in pts[1:]:
        if (x - np.mean(cur)) / np.mean(cur) <= tol:
            cur.append(x)
        else:
            clusters.append(cur); cur = [x]
    if cur: clusters.append(cur)
    lv = [{"price": round(float(np.mean(cl)), 2), "touches": len(cl),
           "role": "support" if np.mean(cl) < price else "resistance"}
          for cl in clusters if len(cl) >= min_touches
          and abs(np.mean(cl) - price) / price <= 0.25]
    return sorted(lv, key=lambda l: abs(l["price"] - price))


# ── Analysis per symbol ────────────────────────────────────────────────────────
def analyze(sym, period="2y", pct=None, refresh=False, show_all=False, interval="1d", fill_today=False):
    daily = interval == "1d"
    note = ""
    if not daily and MONTHS[period] > MONTHS[INTRADAY_MAX[interval]]:
        note = f"{interval} charts are limited to {INTRADAY_MAX[interval]} (Yahoo keeps ~2 years of hourly data)"
        period = INTRADAY_MAX[interval]
    # download a longer period so moving averages (up to 200+ bars) are already
    # warmed up at the start of the chart; patterns use only the requested period
    full = load_prices(sym, WARMUP_PERIOD[period] if daily else None, refresh, interval, fill_today)
    start = full.index[-1] - pd.DateOffset(months=MONTHS[period])
    df = full[full.index > start]
    warm = full[full.index <= start].tail(300)
    a = atr_pct(df)
    bpd = BARS_PER_DAY[interval]
    swing = pct / 100 if pct else max(0.03 if daily else 0.01, round(2.5 * a, 3))
    tol = max(0.015 if daily else 0.006, swing * 0.4)   # "equal" price tolerance
    flat = 0.0004 / bpd                  # |slope| < 0.04%/bar counts as flat
    piv = zigzag(df, swing)
    tl_cands = _trendline_candidates(df, piv, tol)
    pats = (find_double(df, piv, tol) + find_head_shoulders(df, piv, tol)
            + find_line_patterns(df, piv, tol, flat)
            + find_parallel_channels(df, piv, tol, flat, tl_cands)
            + find_trendlines(df, piv, tol, flat, tl_cands)
            + find_ranges(df) + find_volume_events(df) + find_candles(df))
    if not show_all:   # hide patterns that never resolved or were invalidated
        pats = [p for p in pats if p["status"] not in ("failed", "expired")]
    pats.sort(key=lambda p: p["i1"], reverse=True)
    price = float(df["Close"].iloc[-1])
    return {
        "symbol": sym, "period": period, "interval": interval, "note": note,
        "price": round(price, 2), "asof": ts(df.index[-1]),
        "atr_pct": round(a * 100, 2), "swing_pct": round(swing * 100, 1),
        "pivots": [{"date": ts(df.index[p["i"]]), "price": round(p["price"], 2),
                    "kind": p["kind"], "confirmed": p["confirmed"]} for p in piv],
        "patterns": pats,
        "levels": find_levels(piv, price, tol)[:6],
        "read": read_recent(df, pats, piv),
        "warm": warm["Close"].round(2).tolist(),   # closes before the window, for MA warm-up
        "ohlc": {
            "x": [ts(d) for d in df.index],
            "volume": df["Volume"].astype(float).round(0).tolist(),
            "open": df["Open"].round(2).tolist(), "high": df["High"].round(2).tolist(),
            "low": df["Low"].round(2).tolist(), "close": df["Close"].round(2).tolist(),
        },
    }


# ── Report ─────────────────────────────────────────────────────────────────────
HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chart Formations</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e6e6;--mute:#8a93a5;--bull:#26a69a;--bear:#ef5350;--neu:#f5b041;--line:#2a2f3a;--acc:#5c8dff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1300px;margin:auto;padding:16px}
a{color:var(--acc)}
h1{font-size:20px;margin:4px 0 2px} h2{font-size:17px;margin:0}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.sub{color:var(--mute);font-size:12px}
.lookup{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 4px;align-items:center}
.lookup input,.lookup select,.lookup button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 10px;font:inherit}
.lookup input{width:150px;text-transform:uppercase}
.lookup button{background:var(--acc);border-color:var(--acc);color:#fff;cursor:pointer}
.lookup button:disabled{opacity:.6;cursor:wait}
#msg{font-size:13px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.x{cursor:pointer;color:var(--mute);background:none;border:0;font-size:18px}
.chart{height:580px}
.filters{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 2px;font-size:13px}
.filters label{cursor:pointer;user-select:none;color:var(--mute)} .filters input{vertical-align:-2px}
.filters select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;font:inherit}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)} th{color:var(--mute);font-weight:500}
tr.pat{cursor:pointer} tr.pat:hover{background:#1f2430}
.bullish{color:var(--bull)} .bearish{color:var(--bear)} .neutral{color:var(--neu)}
.lv{display:inline-block;margin:6px 8px 0 0;padding:2px 8px;border-radius:12px;border:1px solid var(--line);font-size:12px}
.tag{font-size:11px;padding:1px 6px;border-radius:8px;background:#232836;color:var(--mute)}
.wrap{overflow-x:auto}
.ma-bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0 4px;font-size:13px;color:var(--mute)}
.ma-bar input[type=text]{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 8px;font:inherit;width:240px}
.ma-bar label{cursor:pointer;user-select:none}
.state{display:grid;grid-template-columns:minmax(200px,260px) 1fr;gap:16px;margin:6px 0;padding:10px 12px;border:1px solid var(--line);border-radius:8px;font-size:13px}
@media(max-width:700px){.state{grid-template-columns:1fr}}
.state table{margin:0} .state td,.state th{padding:3px 8px}
.verdict{font-size:19px;font-weight:600;margin:2px 0}
.meter{height:6px;border-radius:3px;background:linear-gradient(90deg,#ef5350,#8a93a5,#26a69a);position:relative;margin:8px 0 4px}
.meter i{position:absolute;top:-4px;width:3px;height:14px;background:#fff;border-radius:2px}
.crosses{margin-top:6px;line-height:1.6}
.ivs{display:inline-flex;gap:4px;margin-left:10px;vertical-align:2px}
.ivs button{background:#232836;color:var(--mute);border:1px solid var(--line);border-radius:4px;padding:1px 7px;font:12px system-ui,Segoe UI,sans-serif;cursor:pointer}
.ivs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.vbtn{margin-left:10px;background:#232836;color:var(--fg);border:1px solid var(--acc);border-radius:4px;padding:1px 9px;font:12px system-ui,Segoe UI,sans-serif;cursor:pointer;vertical-align:2px}
.val{border:1px solid var(--line);border-radius:8px;padding:12px;margin:10px 0;font-size:13px}
.val h3{margin:0 0 6px;font-size:15px}
.vsum{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 10px}
.vsum div{min-width:120px} .vsum b{font-size:17px;display:block}
.vin{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin:8px 0}
.vin label{display:flex;flex-direction:column;color:var(--mute);font-size:12px}
.vin input{width:84px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 6px;font:inherit}
.vin button{background:#232836;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 10px;cursor:pointer}
.sens td,.sens th{text-align:right;padding:3px 7px} .sens .now{outline:1px solid var(--acc)}
.flag{color:var(--neu);margin:3px 0}
.read{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0;font-size:13px}
.read summary{cursor:pointer;font-weight:600}
.read ul{margin:6px 0 0;padding-left:0;list-style:none}
.read li{margin:3px 0;padding-left:20px;text-indent:-20px}
.read .ic{display:inline-block;width:16px;text-indent:0;font-weight:700}
.note{color:var(--neu);font-size:12px}
</style></head><body><main>
<div class="top">
  <div><h1>Chart Formations</h1>
  <div class="sub">Generated __GEN__ · swing-pivot pattern detection · mechanical candidates, not trade advice. Click a table row to zoom.</div></div>
  <div><a href="/brief">Market brief</a> · <a href="/">Daily signals</a></div>
</div>
<form class="lookup" id="lookup">
  <input id="sym" placeholder="Ticker, e.g. NVDA" autocomplete="off" spellcheck="false" required>
  <select id="per">__PERIODS__</select>
  <select id="iv"><option value="1d">1D</option><option value="4h">4H</option><option value="2h">2H</option><option value="1h">1H</option></select>
  <button id="go">Analyze</button>
  <span id="msg" class="sub"></span>
</form>
<div id="root"></div>
<section id="seas" hidden></section></main>
<script>
const DATA = __DATA__;
const SEAS = __SEAS__;
const COL = {bullish:'#26a69a', bearish:'#ef5350', neutral:'#f5b041'};
const CATS = [['reversal','Reversals'],['triangle','Triangles & wedges'],['channel','Channels'],['trendline','Trendlines'],['range','Ranges'],['wyckoff','Wyckoff'],['volume','Volume'],['candle','Candles']];
const root = document.getElementById('root');
let seq = 0;

// ── Moving averages (computed in the browser so they are configurable) ──
const MA_DEFAULT = 'EMA9, EMA21, SMA50, SMA200, WMA20, HMA55, AVWAP swing, RSI14, MACD12/26/9';
const MA_COLORS = ['#ffd54f','#4fc3f7','#ba68c8','#ff8a65','#e0e0e0','#81c784'];
let maCfg = MA_DEFAULT;
try { maCfg = localStorage.getItem('maCfg') || MA_DEFAULT; } catch (e) {}
if (['EMA9,EMA21,SMA50,SMA200', 'EMA9,EMA21,SMA50,SMA200,RSI14,MACD12/26/9'].includes(maCfg.replace(/\s/g, '')))
  maCfg = MA_DEFAULT;   // upgrade old saved defaults
function rsi(c, n) {   // Wilder's RSI
  const out = Array(c.length).fill(null);
  if (c.length <= n) return out;
  let g = 0, l = 0;
  for (let i = 1; i <= n; i++) { const d = c[i] - c[i-1]; d > 0 ? g += d : l -= d; }
  g /= n; l /= n;
  out[n] = l == 0 ? 100 : 100 - 100 / (1 + g / l);
  for (let i = n + 1; i < c.length; i++) {
    const d = c[i] - c[i-1];
    g = (g * (n - 1) + Math.max(d, 0)) / n; l = (l * (n - 1) + Math.max(-d, 0)) / n;
    out[i] = l == 0 ? 100 : 100 - 100 / (1 + g / l);
  }
  return out;
}
function macd(c, f, sl, sg) {
  const a = ema(c, f), b = ema(c, sl);
  const line = c.map((_, i) => a[i] != null && b[i] != null ? a[i] - b[i] : null);
  const first = line.findIndex(v => v != null);
  const sig = Array(c.length).fill(null);
  if (first >= 0) ema(line.slice(first), sg).forEach((v, k) => sig[first + k] = v);
  return {line, sig, hist: line.map((v, i) => v != null && sig[i] != null ? v - sig[i] : null)};
}
function parseOsc(txt) {
  const r = txt.match(/RSI\s*(\d+)/i), m = txt.match(/MACD\s*(\d+)\s*\/\s*(\d+)\s*\/\s*(\d+)/i);
  return {rsi: r ? Math.min(Math.max(+r[1], 2), 100) : null,
          macd: m && +m[1] < +m[2] ? [+m[1], +m[2], +m[3]] : null};
}
function sma(c, n) {
  const out = Array(c.length).fill(null); let sum = 0;
  for (let i = 0; i < c.length; i++) { sum += c[i]; if (i >= n) sum -= c[i-n]; if (i >= n-1) out[i] = sum / n; }
  return out;
}
function ema(c, n) {
  const out = Array(c.length).fill(null), k = 2 / (n + 1);
  if (c.length < n) return out;
  let v = c.slice(0, n).reduce((a, b) => a + b, 0) / n; out[n-1] = v;
  for (let i = n; i < c.length; i++) { v = c[i] * k + v * (1 - k); out[i] = v; }
  return out;
}
function wma(c, n) {   // linearly weighted: newest bar weight n, oldest weight 1
  const out = Array(c.length).fill(null), den = n * (n + 1) / 2;
  for (let i = n - 1; i < c.length; i++) {
    let s = 0; for (let k = 0; k < n; k++) s += c[i - k] * (n - k);
    out[i] = s / den;
  }
  return out;
}
function hma(c, n) {   // Hull: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
  const a = wma(c, Math.max(1, Math.round(n / 2))), b = wma(c, n);
  const first = b.findIndex(v => v != null), out = Array(c.length).fill(null);
  if (first < 0) return out;
  const d = [];
  for (let i = first; i < c.length; i++) d.push(2 * a[i] - b[i]);
  wma(d, Math.max(1, Math.round(Math.sqrt(n)))).forEach((v, k) => out[first + k] = v);
  return out;
}
const MA_FN = {EMA: ema, SMA: sma, WMA: wma, HMA: hma};
function vwapFrom(o, i0, session) {   // cumulative typical-price x volume / volume
  const out = Array(o.close.length).fill(null);
  let pv = 0, vv = 0, day = null;
  for (let i = Math.max(i0, 0); i < o.close.length; i++) {
    if (session && o.x[i].slice(0, 10) != day) { day = o.x[i].slice(0, 10); pv = 0; vv = 0; }
    const v = o.volume ? o.volume[i] : 0;
    pv += (o.high[i] + o.low[i] + o.close[i]) / 3 * v; vv += v;
    out[i] = vv > 0 ? pv / vv : null;
  }
  return out;
}
function parseVwap(txt) {
  return {session: /(^|[^A-Z])VWAP(?!\s*\d)/i.test(txt), swing: /AVWAP\s*SWING/i.test(txt),
          anchors: [...txt.matchAll(/AVWAP\s*(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?)/gi)].map(m => m[1].replace('T', ' ')).slice(0, 4)};
}
const parseTs = x => Date.parse(x.length > 10 ? x.replace(' ', 'T') + ':00Z' : x + 'T00:00:00Z');
const tsStr = ms => new Date(ms).toISOString().replace('T', ' ').slice(0, 16);
const EXTRA_COLORS = ['#f48fb1','#aed581','#80deea','#ffcc80','#ce93d8'];
function parseMA(txt) {   // "EMA9, SMA 50, WMA20, HMA55" -> [{kind, n, name}], sorted fast -> slow
  const seen = new Set(), out = [];
  for (const m of txt.matchAll(/(EMA|SMA|WMA|HMA|MA)\s*(\d+)/gi)) {
    const k = m[1].toUpperCase(), kind = k == 'MA' ? 'SMA' : k, n = +m[2];
    if (n < 2 || n > 400 || seen.has(kind + n)) continue;
    seen.add(kind + n); out.push({kind, n, name: kind + n});
  }
  return out.sort((a, b) => a.n - b.n || a.kind.localeCompare(b.kind)).slice(0, 6);
}
const SLOPE_BARS = 5;     // slope = change of the MA over the last 5 bars
function verdict(sc, prev) {
  if (sc == null) return ['No data', 'neutral'];
  const d = prev == null ? 0 : sc - prev;
  if (sc >= 0.5)  return d <= -0.4 ? ['Bullish, weakening', 'neutral'] : ['Bullish', 'bullish'];
  if (sc <= -0.5) return d >= 0.4 ? ['Bearish, improving', 'neutral'] : ['Bearish', 'bearish'];
  if (d >= 0.3)  return ['Turning bullish', 'bullish'];
  if (d <= -0.3) return ['Turning bearish', 'bearish'];
  return ['Neutral', 'neutral'];
}

// ── Multi-timeframe turn ladder ──
async function toggleLadder(sym, el) {
  if (!el.hidden) { el.hidden = true; return; }
  el.hidden = false;
  if (location.protocol == 'file:') { el.innerHTML = `<div class="sub">Runs on the website. Locally: python mtf.py ${sym}</div>`; return; }
  el.innerHTML = `<div class="sub">Loading 5m / 15m / 1h / 4h / daily for ${sym}…</div>`;
  try {
    const res = await fetch(`/api/mtf?symbol=${encodeURIComponent(sym)}`);
    const l = await res.json();
    if (!res.ok || l.error) throw new Error(l.error || res.statusText);
    renderLadder(l, el);
  } catch (err) { el.innerHTML = `<div class="flag">${sym}: ${err.message}</div>`; }
}
function renderLadder(l, el) {
  const s = l.summary, word = d => d == 'up' ? 'above' : 'below';
  const casc = !s ? '' : `<div class="vsum">${s.steps.map(x => `<div>${x.tf}<b class="${x.status == 'confirmed' ? (s.dir == 'up' ? 'bullish' : 'bearish') : x.status == 'pending' ? 'neutral' : ''}">${
      x.status == 'confirmed' ? '✓ confirmed' : x.status == 'pending' ? 'pending' : x.status}</b><span class="sub">${
      x.status == 'confirmed' ? x.when : x.status == 'pending' && x.level != null ? `close ${word(s.dir)} ${x.level}${x.distance_pct != null ? ` (${x.distance_pct > 0 ? '+' : ''}${x.distance_pct}%)` : ''}` : x.when ? 'since ' + x.when : ''}</span></div>`).join('')}</div>`;
  const rows = l.timeframes.map(r => {
    const t = r.last_turn, p = r.pending;
    return `<tr><td><b>${r.tf}</b></td><td class="${r.trend == 'up' ? 'bullish' : r.trend == 'down' ? 'bearish' : 'neutral'}">${r.trend}</td>
      <td>${t ? `<b class="${t.dir == 'up' ? 'bullish' : 'bearish'}">${t.dir}</b> (${t.kind}) ${t.when}<div class="sub">closed ${word(t.dir)} ${t.level} after the ${t.dir == 'up' ? 'low' : 'high'} ${t.swing.price} (${t.swing.when})</div>` : '–'}</td>
      <td>${p ? `${p.kind} ${p.dir}: close ${word(p.dir)} <b>${p.level}</b> <span class="sub">(${p.distance_pct > 0 ? '+' : ''}${p.distance_pct}%)</span>` : '–'}</td>
      <td>${r.rsi}</td><td class="${r.macd_hist > 0 ? 'bullish' : 'bearish'}">${r.macd_hist > 0 ? '+' : '−'} ${r.macd_rising ? '↑' : '↓'}</td>
      <td class="sub">${r.ema_cross ? `EMA9/21 ${r.ema_cross.dir} ${r.ema_cross.when}` : ''}</td></tr>`;
  }).join('');
  el.innerHTML = `<h3>${l.symbol} — turn ladder ${s ? `<span class="sub">latest 5m turn ${s.dir} from the ${s.from_swing.price} swing at ${s.from_swing.when}</span>` : ''}</h3>
    ${casc}
    <div class="wrap"><table><tr><th>TF</th><th>Trend</th><th>Last confirmed turn</th><th>Next confirmation</th><th>RSI</th><th>MACD</th><th></th></tr>${rows}</table></div>
    ${Object.entries(l.errors || {}).map(([k, e]) => `<div class="flag">${k}: ${e}</div>`).join('')}
    <div class="sub" style="margin-top:8px">A turn is confirmed on a timeframe when, after a swing low (high), a bar closes above the swing high (below the swing low) that came before it. A bottom usually confirms on 5m first, then 15m, 1h, 4h; "pending" shows the close each slower timeframe still needs. "Already trending" means that timeframe was already moving that way, so the drop (rise) was only a pullback there. Yahoo intraday data can be delayed ~15 min. Not investment advice.</div>`;
}

// ── Valuation (intrinsic value estimate) ──
function dcfVal(f0, g1, g2, r, tg, cash, debt, sh) {
  if (r <= tg || sh <= 0) return null;
  let pv = 0, f = f0;
  for (let y = 1; y <= 10; y++) { f *= 1 + (y <= 5 ? g1 : g2); pv += f / Math.pow(1 + r, y); }
  return (pv + f * (1 + tg) / (r - tg) / Math.pow(1 + r, 10) + cash - debt) / sh;
}
function impliedG(price, f0, r, tg, cash, debt, sh) {
  const f = g => dcfVal(f0, g, (g + tg) / 2, r, tg, cash, debt, sh) - price;
  let lo = -0.5, hi = 1.5;
  if (f0 <= 0 || f(lo) > 0 || f(hi) < 0) return null;
  for (let k = 0; k < 80; k++) { const m = (lo + hi) / 2; f(m) < 0 ? lo = m : hi = m; }
  return (lo + hi) / 2;
}
const pctS = (x, d = 1) => x == null ? '–' : (x >= 0 ? '+' : '') + (x * 100).toFixed(d) + '%';
const big = x => x == null ? '–' : Math.abs(x) >= 1e12 ? (x / 1e12).toFixed(2) + 'T' : Math.abs(x) >= 1e9 ? (x / 1e9).toFixed(1) + 'B' : (x / 1e6).toFixed(0) + 'M';
async function toggleValuation(sym, el) {
  if (!el.hidden) { el.hidden = true; return; }
  el.hidden = false;
  if (el.dataset.loaded) return;
  if (location.protocol == 'file:') { el.innerHTML = `<div class="sub">Valuation runs on the website. Locally: python valuation.py ${sym}</div>`; return; }
  el.innerHTML = `<div class="sub">Loading financials for ${sym}… (5–10 s)</div>`;
  try {
    const res = await fetch(`/api/valuation?symbol=${encodeURIComponent(sym)}`);
    const v = await res.json();
    if (!res.ok || v.error) throw new Error(v.error || res.statusText);
    el.dataset.loaded = 1;
    renderValuation(v, el);
  } catch (err) { el.innerHTML = `<div class="flag">${sym}: ${err.message}</div>`; }
}
function renderValuation(v, el) {
  const flags = v.flags.map(f => `<div class="flag">⚠ ${f}</div>`).join('');
  const foot = `<div class="sub" style="margin-top:8px">Estimates from Yahoo Finance data and the assumptions shown; small changes in growth or discount rate move the result a lot. Not investment advice.</div>`;
  if (v.etf) {
    const e = v.etf, erp = e.equity_risk_premium;
    el.innerHTML = `<h3>${v.name} — valuation vs bonds</h3>
      <div class="vsum"><div>P/E<b>${e.pe ?? '–'}</b></div><div>Earnings yield<b>${e.earnings_yield == null ? '–' : (e.earnings_yield * 100).toFixed(2) + '%'}</b></div>
        <div>10Y Treasury<b>${(v.ten_year * 100).toFixed(2)}%</b></div>
        <div>Equity risk premium<b class="${erp == null ? '' : erp < 0 ? 'bearish' : erp < 0.02 ? 'neutral' : 'bullish'}">${erp == null ? '–' : (erp >= 0 ? '+' : '') + (erp * 100).toFixed(2) + 'pp'}</b></div></div>
      <div class="sub">An ETF holds many companies, so it has no single intrinsic value. The earnings yield (1 ÷ P/E) is what the index earns per dollar invested; comparing it with the 10-year Treasury yield shows how richly stocks are priced against risk-free bonds. Historically the gap has usually been positive; near zero or negative means investors are paying up for expected growth.</div>${flags}${foot}`;
    return;
  }
  const i = v.inputs || {}, m = v.multiples || {}, sc = v.scenarios;
  const hist = (m.history || []).map(h => `<tr><td>${h.fiscal_year_end}</td><td>${h.price}</td><td>${h.pe ?? '–'}</td><td>${h.p_fcf ?? '–'}</td></tr>`).join('');
  const multHtml = `<h3 style="margin-top:12px">Multiples</h3><div class="wrap"><table><tr><th>Fiscal year end</th><th>Price</th><th>P/E</th><th>P/FCF</th></tr>${hist}
      <tr><td><b>Now</b></td><td>${v.price}</td><td>${m.pe_now == null ? '–' : m.pe_now.toFixed(1)} <span class="sub">(forward ${m.forward_pe == null ? '–' : m.forward_pe.toFixed(1)})</span></td><td>${m.p_fcf_now ?? '–'}</td></tr></table></div>`;
  if (!sc) { el.innerHTML = `<h3>${v.name} — valuation</h3>${flags}${multHtml}${foot}`; return; }
  const fcfHist = Object.entries(i.fcf_history || {}).map(([d, x]) => `${d.slice(0, 4)}: ${big(x)}`).join(' · ');
  el.innerHTML = `<h3>${v.name} — intrinsic value estimate (DCF)</h3>
    <div class="vsum">
      <div>Price<b>${v.price}</b></div>
      <div>Base value<b class="vbase"></b><span class="sub vbase2"></span></div>
      <div>Bear – bull<b>${sc.bear.value} – ${sc.bull.value}</b><span class="sub">${pctS(sc.bear.g1, 0)} … ${pctS(sc.bull.g1, 0)} growth</span></div>
      <div>Market implies<b class="vimp"></b><span class="sub">FCF growth/yr for 5 years, then about half</span></div>
    </div>
    <div class="vin">
      <label>FCF, ${v.currency || ''} bn<input type="number" step="0.1" data-k="fcf" value="${(i.fcf / 1e9).toFixed(1)}"></label>
      <label>Growth yrs 1–5 %<input type="number" step="0.5" data-k="g1" value="${(i.growth_base * 100).toFixed(1)}"></label>
      <label>Growth yrs 6–10 %<input type="number" step="0.5" data-k="g2" value="${((i.growth_base + i.terminal_growth) / 2 * 100).toFixed(1)}"></label>
      <label>Discount rate %<input type="number" step="0.25" data-k="r" value="${(i.discount_rate * 100).toFixed(2)}"></label>
      <label>Terminal growth %<input type="number" step="0.25" data-k="tg" value="${(i.terminal_growth * 100).toFixed(2)}"></label>
      <button class="vreset">Reset</button>
    </div>
    <div class="sub">FCF base: ${i.fcf_source}; history ${fcfHist}. Growth default: ${i.growth_source}${i.fcf_cagr != null ? `; FCF grew ${pctS(i.fcf_cagr, 0)}/yr historically` : ''}.
      Discount rate = 10Y ${(i.risk_free * 100).toFixed(2)}% + beta ${i.beta} × ${(i.erp * 100).toFixed(0)}% risk premium (kept within 8–14%). Cash ${big(i.cash)}, debt ${big(i.debt)}, shares ${big(i.shares)}.</div>
    <h3 style="margin-top:12px">Sensitivity <span class="sub">value per share by discount rate (rows) and growth in years 1–5 (columns); green = above today's price</span></h3>
    <div class="wrap"><table class="sens"></table></div>
    ${multHtml}${flags}${foot}`;
  const inp = k => el.querySelector(`[data-k=${k}]`);
  const read = () => ({fcf: +inp('fcf').value * 1e9, g1: +inp('g1').value / 100, g2: +inp('g2').value / 100, r: +inp('r').value / 100, tg: +inp('tg').value / 100});
  let g2Touched = false;
  function update() {
    const a = read(), val = dcfVal(a.fcf, a.g1, a.g2, a.r, a.tg, i.cash, i.debt, i.shares);
    const vs = val == null ? null : val / v.price - 1;
    el.querySelector('.vbase').innerHTML = val == null ? '–' : `<span class="${vs >= 0 ? 'bullish' : 'bearish'}">${val.toFixed(2)}</span>`;
    el.querySelector('.vbase2').textContent = vs == null ? '' : `${pctS(vs)} vs price` + (vs > 0 ? ` · margin of safety ${(vs / (1 + vs) * 100).toFixed(0)}%` : '');
    const ig = impliedG(v.price, a.fcf, a.r, a.tg, i.cash, i.debt, i.shares);
    el.querySelector('.vimp').textContent = ig == null ? 'n/a' : pctS(ig);
    const rs = [-0.02, -0.01, 0, 0.01, 0.02].map(d => a.r + d), gs = [-0.10, -0.05, 0, 0.05, 0.10].map(d => a.g1 + d);
    el.querySelector('.sens').innerHTML = `<tr><th></th>${gs.map(g => `<th>${pctS(g, 0)}</th>`).join('')}</tr>` +
      rs.map((r, ri) => `<tr><th>${(r * 100).toFixed(2)}%</th>${gs.map((g, gi) => {
        const x = dcfVal(a.fcf, g, (g + a.tg) / 2, r, a.tg, i.cash, i.debt, i.shares);
        return `<td class="${x == null ? '' : x >= v.price ? 'bullish' : 'bearish'}${ri == 2 && gi == 2 ? ' now' : ''}">${x == null ? '–' : x.toFixed(0)}</td>`;
      }).join('')}</tr>`).join('');
  }
  el.querySelectorAll('.vin input').forEach(x => x.oninput = () => {
    if (x.dataset.k == 'g2') g2Touched = true;
    if ((x.dataset.k == 'g1' || x.dataset.k == 'tg') && !g2Touched) inp('g2').value = ((+inp('g1').value + +inp('tg').value) / 2).toFixed(1);
    update();
  });
  el.querySelector('.vreset').onclick = () => { delete el.dataset.loaded; el.hidden = true; toggleValuation(v.symbol, el); };
  update();
}

function stripColor(sc) {
  if (sc == null) return 'rgba(0,0,0,0)';
  if (sc >= 0.5) return '#26a69a'; if (sc >= 0.2) return '#1b6b63';
  if (sc <= -0.5) return '#ef5350'; if (sc <= -0.2) return '#8e3634';
  return '#4a5060';
}

function render(r, prepend) {
  const n = seq++;
  const s = document.createElement('section');
  const open = r.patterns.filter(p => p.open).length;
  const IV = r.interval || '1d', daily = IV == '1d';
  const unit = daily ? 'trading days' : 'bars', bu = daily ? 'd' : ' bars';
  s.innerHTML = `<div class="head"><h2>${r.symbol} <span class="sub">${r.price} · ${r.asof} · ${IV.toUpperCase()} · ${r.period} · ATR ${r.atr_pct}%/bar · swing ${r.swing_pct}% · ${r.patterns.length} patterns (${open} open)</span>
      <span class="ivs">${['1d','4h','2h','1h'].map(v => `<button data-iv="${v}" class="${v == IV ? 'on' : ''}" title="Open ${r.symbol} on ${v.toUpperCase()} bars">${v.toUpperCase()}</button>`).join('')}</span><button class="vbtn" title="Intrinsic value estimate">Valuation</button><button class="vbtn tbtn" title="Has a turn been confirmed on 5m, 15m, 1h, 4h and daily?">Timeframes</button></h2>
      ${prepend ? '<button class="x" title="Remove">&times;</button>' : ''}</div>
    ${r.note ? `<div class="note">${r.note}</div>` : ''}
    <div class="val" hidden></div>
    <div class="val mtf" hidden></div>
    <div>${r.levels.map(l => `<span class="lv ${l.role=='support'?'bullish':'bearish'}">${l.role} ${l.price} ×${l.touches}</span>`).join('')}</div>
    <div class="filters">${CATS.map(([k,t]) => `<label><input type="checkbox" data-cat="${k}" checked> ${t} </label>`).join('')}
      <label><input type="checkbox" data-opt="levels" checked> S/R levels</label>
      <label>Show <select data-opt="show"><option value="recent">open + last 6 months</option><option value="open">open / active only</option><option value="all">all</option></select></label></div>
    <div class="ma-bar">Moving averages <input type="text" class="ma-cfg" spellcheck="false" title="Comma-separated. Up to 6 of EMA9 / SMA50 (scored) / WMA20 / HMA55 (shown, not scored), plus VWAP (session, intraday), AVWAP swing (from the last swing low and high), AVWAP 2025-04-07 (anchored; or Shift+click the chart), RSI14, MACD12/26/9">
      <label><input type="checkbox" class="ma-show" checked> lines</label>
      <label><input type="checkbox" class="ma-strip" checked> trend strip</label>
      <label><input type="checkbox" class="ma-vol" checked> volume</label>
      <span>· hover the chart to see the state on any bar, click to pin it, Shift+click to anchor a VWAP there</span></div>
    ${r.read ? `<details class="read" open><summary>Last ${r.read.bars} ${daily ? 'days' : 'bars'}: <span class="${r.read.score > 0 ? 'bullish' : r.read.score < 0 ? 'bearish' : 'neutral'}">${r.read.verdict}</span>
      <span class="sub">(${r.read.positives} positive · ${r.read.negatives} negative · since ${r.read.from})</span></summary>
      <ul>${r.read.observations.map(o => `<li><span class="ic ${o.sign > 0 ? 'bullish' : o.sign < 0 ? 'bearish' : 'sub'}">${o.sign > 0 ? '▲' : o.sign < 0 ? '▼' : '•'}</span>${o.text}</li>`).join('')}</ul>
      <div class="sub" style="margin-top:6px">Rule-based read of the recent candles, formations, gaps and volume; the verdict counts positives minus negatives. Not investment advice.</div></details>` : ''}
    <div class="state"></div>
    <div class="chart"></div>
    <div class="wrap"><table><thead><tr><th>Pattern</th><th>Bias</th><th>From</th><th>To</th><th>Status</th><th>Target</th><th>Detail</th></tr></thead><tbody></tbody></table></div>`;
  prepend ? root.prepend(s) : root.appendChild(s);
  const div = s.querySelector('.chart'), tbody = s.querySelector('tbody');
  const boxes = [...s.querySelectorAll('.filters input, .filters select')];
  const T0 = parseTs(r.ohlc.x[0]), T1 = parseTs(r.ohlc.x[r.ohlc.x.length - 1]);
  const cutoff = T1 - (daily ? 183 * 864e5 : (T1 - T0) / 3);   // "recent" window
  const lastDate = p => Math.max(parseTs(p.end), ...p.lines.map(L => parseTs(L[2])));
  s.querySelectorAll('.ivs button').forEach(b => b.onclick = () => {
    const v = b.dataset.iv;
    lookup(r.symbol, v == '1d' ? '2y' : {'4h': '1y', '2h': '6mo', '1h': '3mo'}[v], v);
  });
  if (prepend) s.querySelector('.x').onclick = () => { Plotly.purge(div); s.remove(); };
  s.querySelector('.vbtn').onclick = () => toggleValuation(r.symbol, s.querySelector('.val'));
  s.querySelector('.tbtn').onclick = () => toggleLadder(r.symbol, s.querySelector('.mtf'));

  const C = r.ohlc.close, D = r.ohlc.x, last = C.length - 1;
  const TS = D.map(parseTs);
  const cfgIn = s.querySelector('.ma-cfg'), stateDiv = s.querySelector('.state');
  cfgIn.value = maCfg;
  let mas = [], extras = [], score = [], pairs = [], osc = {}, pinned = last;
  const VOL = r.ohlc.volume || [], hasVolume = VOL.some(v => v > 0);
  const volAvg = sma(VOL.length ? VOL : C.map(() => 0), 20);
  const volRatio = VOL.map((v, i) => i >= 20 && volAvg[i - 1] ? v / volAvg[i - 1] : null);   // vs previous 20 bars
  const fmtVol = v => v >= 1e9 ? (v / 1e9).toFixed(2) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : String(v);

  function computeMA() {
    const W = r.warm || [], CW = W.concat(C);   // warm-up closes so long MAs exist from day 1
    mas = parseMA(cfgIn.value).map((m, k) => ({...m, color: MA_COLORS[k],
      scored: m.kind == 'EMA' || m.kind == 'SMA', v: MA_FN[m.kind](CW, m.n).slice(W.length)}));
    // WMA / HMA are drawn and listed but not scored: they lag much less than EMA/SMA,
    // so the "faster MA above slower MA" check would misread them. HMA is read by its turns.
    const sm = mas.filter(m => m.scored);
    mas.filter(m => m.kind == 'HMA').forEach(m => {
      m.turns = [];
      for (let i = 2; i < C.length; i++) {
        if (m.v[i-2] == null) continue;
        const d0 = m.v[i-1] - m.v[i-2], d1 = m.v[i] - m.v[i-1];
        if (d0 <= 0 && d1 > 0) m.turns.push({i, up: true});
        if (d0 >= 0 && d1 < 0) m.turns.push({i, up: false});
      }
    });
    // VWAP lines are shown in the table but not used in the MA score
    const vw = parseVwap(cfgIn.value), hasVol = (r.ohlc.volume || []).some(v => v > 0);
    extras = [];
    if (vw.session) extras.push(daily ? {name: 'VWAP', why: 'session VWAP needs 4H/2H/1H bars'}
      : hasVol ? {name: 'VWAP', v: vwapFrom(r.ohlc, 0, true)} : {name: 'VWAP', why: 'no volume data'});
    if (vw.swing) {   // anchored at the last confirmed swing low and swing high
      ['L', 'H'].forEach(kind => {
        const pv = r.pivots.filter(q => q.kind == kind && q.confirmed).pop();
        const name = `AVWAP swing ${kind == 'L' ? 'low' : 'high'}${pv ? ' ' + pv.date : ''}`;
        const i0 = pv ? D.indexOf(pv.date) : -1;
        if (!hasVol) extras.push({name, why: 'no volume data'});
        else if (i0 < 0) extras.push({name, why: 'no confirmed swing in range'});
        else extras.push({name, v: vwapFrom(r.ohlc, i0, false), i0});
      });
    }
    vw.anchors.forEach(a => {
      const name = 'AVWAP ' + a, i0 = D.findIndex(d => d >= a);
      if (!hasVol) extras.push({name, why: 'no volume data'});
      else if (i0 < 0 || (i0 == 0 && a < D[0])) extras.push({name, why: 'anchor is outside the chart range'});
      else extras.push({name, v: vwapFrom(r.ohlc, i0, false), i0});
    });
    extras.forEach((e, k) => e.color = EXTRA_COLORS[k % EXTRA_COLORS.length]);
    // daily score in [-1, 1]: price above/below each MA, each MA rising/falling,
    // and whether each faster MA is above the next slower one
    score = C.map((c, i) => {
      let pts = 0, tot = 0;
      sm.forEach((m, k) => {
        const v = m.v[i]; if (v == null) return;
        pts += c > v ? 1 : -1; tot++;
        const pv = m.v[i - SLOPE_BARS];
        if (pv != null) { pts += v > pv ? 1 : -1; tot++; }
        const nx = sm[k + 1];
        if (nx && nx.v[i] != null) { pts += v > nx.v[i] ? 1 : -1; tot++; }
      });
      return tot ? pts / tot : null;
    });
    // crosses of the two fastest and the two slowest MAs
    pairs = [];
    const add = (a, b) => {
      if (!a || !b || pairs.some(p => p.a == a && p.b == b)) return;
      const ev = [];
      for (let i = 1; i < C.length; i++) {
        if (a.v[i-1] == null || b.v[i-1] == null) continue;
        const x0 = a.v[i-1] - b.v[i-1], x1 = a.v[i] - b.v[i];
        if (x0 <= 0 && x1 > 0) ev.push({i, up: true});
        if (x0 >= 0 && x1 < 0) ev.push({i, up: false});
      }
      const gd = a.n == 50 && b.n == 200;
      pairs.push({a, b, ev, label: up => gd ? (up ? 'Golden cross' : 'Death cross') : (up ? 'bullish cross' : 'bearish cross')});
    };
    if (sm.length >= 2) add(sm[0], sm[1]);
    if (sm.length >= 3) add(sm[sm.length - 2], sm[sm.length - 1]);
    // RSI / MACD from the same warm-up series
    const cfg = parseOsc(cfgIn.value);
    osc = {cfg};
    if (cfg.rsi) osc.rsi = rsi(CW, cfg.rsi).slice(W.length);
    if (cfg.macd) {
      const m = macd(CW, ...cfg.macd);
      osc.macd = {line: m.line.slice(W.length), sig: m.sig.slice(W.length), hist: m.hist.slice(W.length), ev: []};
      const {line, sig} = osc.macd;
      for (let i = 1; i < C.length; i++) {
        if ([line[i-1], sig[i-1], line[i], sig[i]].includes(null)) continue;
        const x0 = line[i-1] - sig[i-1], x1 = line[i] - sig[i];
        if (x0 <= 0 && x1 > 0) osc.macd.ev.push({i, up: true});
        if (x0 >= 0 && x1 < 0) osc.macd.ev.push({i, up: false});
      }
    }
  }

  function oscHtml(i) {
    const out = [], b = (cls, t) => `<b class="${cls}">${t}</b>`;
    if (osc.rsi) {
      const v = osc.rsi[i], pv = osc.rsi[i - SLOPE_BARS];
      if (v == null) out.push(`RSI${osc.cfg.rsi}: not enough history`);
      else {
        const zone = v >= 70 ? b('bearish', 'overbought') : v <= 30 ? b('bullish', 'oversold')
                   : v >= 50 ? b('bullish', 'above 50') : b('bearish', 'below 50');
        const d = pv == null ? null : v - pv;
        const dir = d == null ? '' : Math.abs(d) < 2 ? ` · flat over 5${bu}` :
          ` · ${d > 0 ? b('bullish', 'rising') : b('bearish', 'falling')} ${d > 0 ? '+' : ''}${d.toFixed(1)} over 5${bu}`;
        // simple divergence: price at a 20-day high/low but RSI is not
        let div = '';
        if (i >= 20) {
          const win = C.slice(i - 20, i + 1), rw = osc.rsi.slice(i - 20, i + 1).filter(x => x != null);
          if (C[i] >= Math.max(...win) && v < Math.max(...rw) - 3) div = ' · ' + b('bearish', `bearish divergence (new 20${bu} high, weaker RSI)`);
          if (C[i] <= Math.min(...win) && v > Math.min(...rw) + 3) div = ' · ' + b('bullish', `bullish divergence (new 20${bu} low, firmer RSI)`);
        }
        out.push(`RSI${osc.cfg.rsi} <b>${v.toFixed(1)}</b> · ${zone}${dir}${div}`);
      }
    }
    if (osc.macd) {
      const {line, sig, hist, ev} = osc.macd, [f, sl, sg] = osc.cfg.macd;
      if (line[i] == null || sig[i] == null) out.push(`MACD ${f}/${sl}/${sg}: not enough history`);
      else {
        const h = hist[i], ph = hist[i - 1], e = ev.filter(x => x.i <= i).pop();
        const mom = ph == null ? '' : h > 0 ? (h > ph ? b('bullish', 'histogram growing') : b('neutral', 'histogram shrinking (momentum fading)'))
                                            : (h < ph ? b('bearish', 'histogram growing negative') : b('neutral', 'histogram shrinking (selling easing)'));
        out.push(`MACD ${f}/${sl}/${sg} <b>${line[i].toFixed(2)}</b> vs signal ${sig[i].toFixed(2)} · ` +
          `${line[i] > sig[i] ? b('bullish', 'above signal') : b('bearish', 'below signal')} · ` +
          `${line[i] > 0 ? b('bullish', 'above zero') : b('bearish', 'below zero')} · ${mom}` +
          (e ? `<br>&nbsp;&nbsp;last MACD ${e.up ? b('bullish', 'bullish cross') : b('bearish', 'bearish cross')} ${D[e.i]} (${i - e.i} ${unit} before)` : ''));
      }
    }
    return out.join('<br>');
  }

  function showState(i) {
    if (i == null || i < 0) i = last;
    const sc = score[i], prev = i >= SLOPE_BARS ? score[i - SLOPE_BARS] : null;
    const [txt, cls] = verdict(sc, prev);
    const pct = x => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
    const rows = mas.concat(extras).map(m => {
      if (m.why) return `<tr><td style="color:${m.color}">${m.name}</td><td colspan="3" class="sub">${m.why}</td></tr>`;
      const v = m.v[i];
      if (v == null) return `<tr><td style="color:${m.color}">${m.name}</td><td colspan="3" class="sub">${m.i0 != null && i < m.i0 ? 'before the anchor' : 'not enough history'}</td></tr>`;
      const pv = m.v[i - SLOPE_BARS], sl = pv != null ? v / pv - 1 : null;
      const dir = sl == null ? '' : Math.abs(sl) < 0.001 ? '→ flat' : sl > 0 ? '↗ rising' : '↘ falling';
      return `<tr><td style="color:${m.color}">${m.name}${m.scored === false || !m.kind ? ' <span class="sub">· not scored</span>' : ''}</td><td>${v.toFixed(2)}</td>
        <td class="${C[i] > v ? 'bullish' : 'bearish'}">price ${C[i] > v ? 'above' : 'below'} ${pct(C[i] / v - 1)}</td>
        <td class="${sl == null ? 'sub' : sl > 0.001 ? 'bullish' : sl < -0.001 ? 'bearish' : 'neutral'}">${dir}${sl == null ? '' : ' ' + pct(sl)}</td></tr>`;
    }).join('');
    const cr = pairs.map(p => {
      if (p.a.v[i] == null || p.b.v[i] == null) return '';
      const e = p.ev.filter(x => x.i <= i).pop();
      const now = p.a.v[i] > p.b.v[i] ? 'above' : 'below';
      return `${p.a.name} is <b class="${now == 'above' ? 'bullish' : 'bearish'}">${now}</b> ${p.b.name}` +
        (e ? ` · last ${p.label(e.up)} ${D[e.i]} (${i - e.i} ${unit} before)` : ' · no cross in range');
    }).filter(Boolean).concat(mas.filter(m => m.turns).map(m => {
      const e = m.turns.filter(x => x.i <= i).pop();
      return e ? `${m.name} ${e.up ? '<b class="bullish">turned up</b>' : '<b class="bearish">turned down</b>'} ${D[e.i]} (${i - e.i} ${unit} before)` : '';
    }).filter(Boolean)).join('<br>');
    stateDiv.innerHTML = `<div>
        <div class="sub">${i == pinned && i != last ? 'pinned · ' : ''}${D[i]} · close ${C[i].toFixed(2)}</div>
        <div class="verdict ${cls}">${txt}</div>
        <div class="meter">${sc == null ? '' : `<i style="left:calc(${(sc + 1) * 50}% - 1px)"></i>`}</div>
        <div class="sub">MA score ${sc == null ? '–' : sc.toFixed(2)} (5 bars earlier ${prev == null ? '–' : prev.toFixed(2)}) · −1 bearish … +1 bullish</div>
        ${hasVolume ? `<div style="margin-top:8px">Volume <b>${fmtVol(VOL[i])}</b>${volRatio[i] == null ? '' :
          ` · <b class="${volRatio[i] >= 2 ? 'neutral' : ''}">${volRatio[i].toFixed(1)}×</b> the 20-${daily ? 'day' : 'bar'} average${volRatio[i] >= 2 ? ' (spike)' : ''}`}</div>` : ''}
        ${r.patterns.filter(p => (p.cat == 'volume' || p.cat == 'candle') && p.events.some(e => e.date == D[i]))
           .map(p => `<div class="sub"><b class="${p.bias}">${p.type}</b> · ${p.status}</div>`).join('')}</div>
      <div><table><tr><th>MA</th><th>Value</th><th>Price vs MA</th><th>Slope (5 bars)</th></tr>${rows ||
        '<tr><td colspan="4" class="sub">Enter moving averages, e.g. EMA9, EMA21, SMA50, SMA200, RSI14, MACD12/26/9</td></tr>'}</table>
        <div class="crosses">${cr}</div>
        <div class="crosses">${oscHtml(i)}</div></div>`;
  }

  function visible() {
    const on = new Set(boxes.filter(b => b.dataset.cat && b.checked).map(b => b.dataset.cat));
    const show = s.querySelector('[data-opt=show]').value;
    return r.patterns.filter(p => on.has(p.cat) &&
      (show == 'all' || p.open || (show == 'recent' && lastDate(p) >= cutoff)));
  }
  function draw() {
    const pats = visible(), o = r.ohlc;
    const traces = [
      {type:'candlestick', x:o.x, open:o.open, high:o.high, low:o.low, close:o.close, name:r.symbol,
       increasing:{line:{color:'#26a69a'}}, decreasing:{line:{color:'#ef5350'}}},
      {type:'scatter', mode:'lines', x:r.pivots.map(p=>p.date), y:r.pivots.map(p=>p.price),
       line:{color:'#5c6bc0', width:1, dash:'dot'}, name:'swings', hoverinfo:'skip'}
    ];
    if (s.querySelector('.ma-show').checked) mas.forEach(m => traces.push({type:'scatter', mode:'lines',
      x:D, y:m.v, name:m.name, line:{color:m.color, width:1.3}, hovertemplate:m.name + ' %{y:.2f}<extra></extra>'}));
    if (s.querySelector('.ma-show').checked) extras.filter(e => e.v).forEach(e => traces.push({type:'scatter', mode:'lines',
      x:D, y:e.v, name:e.name, line:{color:e.color, width:1.5, dash:'dot'}, hovertemplate:e.name + ' %{y:.2f}<extra></extra>'}));
    pairs.forEach(p => p.ev.forEach(e => traces.push({type:'scatter', mode:'markers', x:[D[e.i]], y:[p.b.v[e.i]],
      marker:{symbol: e.up ? 'triangle-up' : 'triangle-down', size:11, color: e.up ? '#26a69a' : '#ef5350', line:{color:'#fff', width:1}},
      hovertemplate:`${p.a.name}/${p.b.name} ${p.label(e.up)}<br>%{x}<extra></extra>`})));
    const strip = s.querySelector('.ma-strip').checked, volOn = s.querySelector('.ma-vol').checked && hasVolume;
    if (volOn) {
      const o = r.ohlc;
      traces.push({type:'bar', x:D, y:VOL, yaxis:'y3', name:'volume',
        marker:{color:VOL.map((v, i) => volRatio[i] >= 2 ? (o.close[i] >= o.open[i] ? '#26a69a' : '#ef5350')
                                              : (o.close[i] >= o.open[i] ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)'))},
        customdata:volRatio.map(x => x == null ? '' : x.toFixed(1) + 'x avg'),
        hovertemplate:'Volume %{y:.3s} · %{customdata}<extra></extra>', showlegend:false});
      traces.push({type:'scatter', mode:'lines', x:D, y:volAvg, yaxis:'y3', line:{color:'#e0e0e0', width:1},
        hovertemplate:'20-bar avg %{y:.3s}<extra></extra>', showlegend:false});
    }
    if (strip) traces.push({type:'bar', x:D, y:D.map((_, i) => score[i] == null ? 0 : 1), yaxis:'y2',
      marker:{color:score.map(stripColor)}, hovertext:score.map((sc, i) => verdict(sc, score[i - SLOPE_BARS])[0]),
      hoverinfo:'text+x', showlegend:false});
    const shapes = [], ann = [];
    pats.forEach(p => {
      if (!p.box && p.events && p.events.length) {   // volume climaxes / candle patterns: labelled markers
        const c = COL[p.bias], bull = p.bias == 'bullish';
        traces.push({type:'scatter', mode:'markers+text', x:p.events.map(e => e.date), y:p.events.map(e => e.price),
          text:p.events.map(e => e.label), textposition: bull ? 'bottom center' : 'top center', textfont:{color:c, size: p.cat == 'candle' ? 9 : 10},
          marker:{color:c, size: p.cat == 'candle' ? 5 : 8, symbol: p.cat == 'candle' ? 'circle' : (bull ? 'star-triangle-up' : 'star-triangle-down')},
          hovertemplate:p.type + ' (' + p.status + ')<br>%{x}<extra></extra>', showlegend:false});
        return;
      }
      if (p.box) {   // consolidation range / Wyckoff candidate
        const c = p.cat == 'wyckoff' ? COL[p.bias] : '#8fa3c7';
        const [x0, x1, lo, hi] = p.box, wy = p.cat == 'wyckoff';
        shapes.push({type:'rect', x0, x1, y0:lo, y1:hi, line:{color:c, width:1, dash: wy ? 'dot' : 'solid'},
                     fillcolor: wy ? 'rgba(0,0,0,0)' : 'rgba(143,163,199,0.12)', layer:'below'});
        const xe = p.lines[0][2];   // extend the edges to the breakout / last bar
        if (xe > x1) [lo, hi].forEach(y => shapes.push({type:'line', x0:x1, x1:xe, y0:y, y1:y, line:{color:c, width:1, dash:'dot'}}));
        if (wy && p.events.length) traces.push({type:'scatter', mode:'markers+text', x:p.events.map(e => e.date), y:p.events.map(e => e.price),
          text:p.events.map(e => e.label), textposition: p.bias == 'bullish' ? 'bottom center' : 'top center',
          textfont:{color:c, size:10}, marker:{color:c, size:6, symbol:'diamond'}, showlegend:false,
          hovertemplate:p.type + ': %{text}<br>%{x}: %{y}<extra></extra>'});
        ann.push({x:x0, y: wy ? lo : hi, text: wy ? p.type.replace(' (candidate)', '?') : 'Range', showarrow:false,
                  xanchor:'left', yanchor: wy ? 'top' : 'bottom', font:{color:c, size:10}});
        return;
      }
      const c = COL[p.bias], dash = p.cat == 'trendline' ? 'dash' : 'solid';
      p.lines.forEach(L => shapes.push({type:'line', x0:L[0], y0:L[1], x1:L[2], y1:L[3], line:{color:c, width:p.cat=='trendline'?1.5:2, dash}}));
      traces.push({type:'scatter', mode:'markers', x:p.pivots.map(q=>q.date), y:p.pivots.map(q=>q.price),
        marker:{color:c, size:8, symbol:'circle-open', line:{width:2}}, showlegend:false,
        hovertemplate:p.type + '<br>%{x}: %{y}<extra></extra>'});
      if (p.cat != 'trendline') {
        const top = p.pivots.reduce((a,b) => b.price > a.price ? b : a);
        ann.push({x:top.date, y:top.price, text:p.type, showarrow:true, arrowhead:0, ay:-26, font:{color:c, size:11}});
      }
    });
    if (s.querySelector('[data-opt=levels]').checked)
      r.levels.forEach(l => shapes.push({type:'line', xref:'paper', x0:0, x1:1, y0:l.price, y1:l.price,
        line:{color:'#8a93a5', width:1, dash:'dot'}}));
    const keep = div.layout && div.layout.xaxis ? div.layout.xaxis.range : undefined;
    Plotly.react(div, traces, {
      paper_bgcolor:'#171a21', plot_bgcolor:'#171a21', font:{color:'#e6e6e6'},
      margin:{l:50,r:20,t:30,b:30}, showlegend:false, shapes, annotations:ann,
      xaxis:{range:keep, rangeslider:{visible:false}, gridcolor:'#2a2f3a',
        // skip weekends; on intraday charts also skip the hours the US market is closed
        rangebreaks: daily ? [{bounds:['sat','mon']}] : [{bounds:['sat','mon']}, {pattern:'hour', bounds:[16, 9.5]}],
        rangeselector:{bgcolor:'#232836', activecolor:'#5c8dff', font:{color:'#e6e6e6'}, buttons:(daily ? [] :
          [{count:5,label:'5D',step:'day',stepmode:'backward'},{count:1,label:'1M',step:'month',stepmode:'backward'}]).concat([
          {count:3,label:'3M',step:'month',stepmode:'backward'},{count:6,label:'6M',step:'month',stepmode:'backward'},
          {count:1,label:'1Y',step:'year',stepmode:'backward'},{count:2,label:'2Y',step:'year',stepmode:'backward'},
          {step:'all',label:'All'}])}},
      yaxis:{gridcolor:'#2a2f3a', autorange:true, domain: [(strip ? 0.07 : 0) + (volOn ? 0.17 : 0) + (strip || volOn ? 0.02 : 0), 1]},
      yaxis2:{domain:[0, 0.05], visible:false, fixedrange:true, range:[0, 1]},
      yaxis3:{domain: strip ? [0.07, 0.24] : [0, 0.17], gridcolor:'#2a2f3a', showticklabels:false, fixedrange:true, rangemode:'tozero'}, bargap:0
    }, {responsive:true, displaylogo:false});
    if (!div._maEvents) {
      // map the mouse x-position to the nearest trading day (works anywhere on the chart)
      div._maEvents = true;
      const dayAt = ev => {
        const xa = div._fullLayout.xaxis, px = ev.clientX - div.getBoundingClientRect().left - xa._offset;
        if (px < 0 || px > xa._length) return null;
        const t = xa.p2c(px);
        let lo = 0, hi = last;                       // first bar at or after t, then pick the nearer one
        while (lo < hi) { const m = (lo + hi) >> 1; TS[m] < t ? lo = m + 1 : hi = m; }
        return lo > 0 && t - TS[lo - 1] < TS[lo] - t ? lo - 1 : lo;
      };
      // when the visible dates change, fit the price axis to the visible candles
      // (otherwise far-away level lines stretch it)
      div.on('plotly_relayout', ev => {
        let a = ev['xaxis.range[0]'], b = ev['xaxis.range[1]'];
        if (ev['xaxis.range']) [a, b] = ev['xaxis.range'];
        if (ev['xaxis.autorange']) [a, b] = [D[0], D[last]];
        if (a == null || b == null) return;
        const t0 = parseTs(String(a).slice(0, 16)), t1 = parseTs(String(b).slice(0, 16));
        let lo = Infinity, hi = -Infinity;
        TS.forEach((t, i) => { if (t >= t0 && t <= t1) { lo = Math.min(lo, r.ohlc.low[i]); hi = Math.max(hi, r.ohlc.high[i]); } });
        if (lo < hi) { const pad = (hi - lo) * 0.06; Plotly.relayout(div, {'yaxis.range': [lo - pad, hi + pad]}); }
      });
      let raf = 0, down = null;
      div.addEventListener('mousemove', ev => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => { const i = dayAt(ev); showState(i == null ? pinned : i); });
      });
      div.addEventListener('mouseleave', () => { cancelAnimationFrame(raf); showState(pinned); });
      div.addEventListener('mousedown', ev => down = [ev.clientX, ev.clientY]);
      div.addEventListener('mouseup', ev => {          // a click (not a drag-zoom) pins the day
        if (down && Math.abs(ev.clientX - down[0]) + Math.abs(ev.clientY - down[1]) < 4) {
          const i = dayAt(ev);
          if (i != null && ev.shiftKey) {               // Shift+click: anchored VWAP from this bar
            cfgIn.value = cfgIn.value.replace(/\s*,?\s*$/, '') + ', AVWAP ' + D[i];
            cfgIn.onchange();
          } else if (i != null) { pinned = i; showState(i); }
        }
        down = null;
      });
    }
    tbody.innerHTML = pats.map((p, k) => `<tr class="pat" data-k="${k}"><td>${p.type} <span class="tag">${p.cat}</span></td><td class="${p.bias}">${p.bias}</td><td>${p.start}</td><td>${p.end}</td><td>${p.status}</td><td>${p.target ?? ''}</td><td class="sub">${p.note}</td></tr>`).join('')
      || '<tr><td colspan="7" class="sub">No patterns for the selected filters.</td></tr>';
    tbody.querySelectorAll('tr.pat').forEach(tr => tr.onclick = () => {
      const p = pats[+tr.dataset.k], pad = 20 * (TS[last] - TS[0]) / last;   // ~20 bars
      Plotly.relayout(div, {'xaxis.range':[tsStr(parseTs(p.start) - pad), tsStr(lastDate(p) + pad)], 'yaxis.autorange':true});
      div.scrollIntoView({behavior:'smooth', block:'center'});
    });
  }
  boxes.forEach(b => b.onchange = draw);
  s.querySelectorAll('.ma-show, .ma-strip, .ma-vol').forEach(b => b.onchange = draw);
  cfgIn.onchange = () => {
    maCfg = cfgIn.value;
    try { localStorage.setItem('maCfg', maCfg); } catch (e) {}
    computeMA(); draw(); showState(pinned);
  };
  computeMA();
  draw();
  showState(last);
}

DATA.forEach(r => render(r, false));

// ── Sector seasonality ──
(function () {
  if (!SEAS || !SEAS.sectors || !SEAS.sectors.length) return;
  const el = document.getElementById('seas');
  el.hidden = false;
  const M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const mi = SEAS.month - 1, ni = mi % 12 + 1 > 11 ? 0 : mi + 1;
  const PAL = ['#e0e0e0','#4fc3f7','#ba68c8','#ffb74d','#f06292','#aed581','#64b5f6','#4db6ac','#e57373','#9575cd','#fff176','#ff8a65','#a1887f','#90a4ae','#81c784','#f48fb1','#ce93d8'];
  const DEFAULT_ON = new Set(['SPY','XLK','SMH','XLE','XLF','IYT','XLC','XLU']);
  const secs = SEAS.sectors.map((x, k) => ({...x, color: PAL[k % PAL.length]}));
  const fmt = v => v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(1) + '%';
  const cls = v => v == null ? '' : v > 0 ? 'bullish' : v < 0 ? 'bearish' : '';
  const heat = v => { if (v == null) return ''; const a = Math.min(Math.abs(v) / 4, 1) * 0.55; return v > 0 ? `background:rgba(38,166,154,${a})` : `background:rgba(239,83,80,${a})`; };
  const rows = secs.slice().sort((a, b) => b.month_avg[mi] - a.month_avg[mi]).map(x => `<tr>
      <td><span style="color:${x.color}">■</span> ${x.label} <span class="sub">${x.symbol} · ${x.years}y</span></td>
      <td class="${cls(x.month_avg[mi])}">${fmt(x.month_avg[mi])}</td><td>${x.month_up[mi]}%</td>
      <td class="${cls(x.month_so_far)}">${fmt(x.month_so_far)}</td>
      <td class="${cls(x.month_avg[ni])}">${fmt(x.month_avg[ni])}</td><td>${x.month_up[ni]}%</td>
      <td class="${cls(x.ytd)}">${fmt(x.ytd)}</td><td>${fmt(x.norm_today)}</td>
      <td class="${cls(x.ytd - x.norm_today)}">${x.ytd == null ? '–' : fmt(x.ytd - x.norm_today)}</td></tr>`).join('');
  const heatRows = secs.map(x => `<tr><td>${x.symbol}</td>${x.month_avg.map((v, k) =>
      `<td style="${heat(v)}${k == mi ? ';outline:1px solid #5c8dff' : ''}" title="${x.label} ${M[k]}: avg ${fmt(v)}, up in ${x.month_up[k]}% of years">${v == null ? '' : v.toFixed(1)}</td>`).join('')}</tr>`).join('');
  el.innerHTML = `<div class="head"><h2>Sector seasonality <span class="sub">average path through the year (last ${Math.max(...secs.map(x => x.years))} years) vs ${SEAS.year} so far</span></h2></div>
    <div class="filters"><label>Show <select id="seasMode"><option value="both">average + this year</option><option value="avg">average only</option><option value="cur">this year only</option></select></label>
      <span class="sub">click a name in the legend to show / hide it; double-click to show only that one</span></div>
    <div id="seasChart" style="height:520px"></div>
    <h3 style="font-size:15px;margin:14px 0 4px">Now: ${M[mi]} and ${M[ni]} <span class="sub">sorted by the usual ${M[mi]} return</span></h3>
    <div class="wrap"><table><tr><th>Sector</th><th>${M[mi]} avg</th><th>yrs up</th><th>${M[mi]} so far</th><th>${M[ni]} avg</th><th>yrs up</th><th>YTD ${SEAS.year}</th><th>usual by now</th><th>vs usual</th></tr>${rows}</table></div>
    <h3 style="font-size:15px;margin:14px 0 4px">Average return by month <span class="sub">green = usually up, red = usually down; hover for the share of positive years</span></h3>
    <div class="wrap"><table class="sens"><tr><th></th>${M.map(m => `<th>${m}</th>`).join('')}</tr>${heatRows}</table></div>
    <div class="sub" style="margin-top:8px">Seasonality is a historical average, not a forecast: individual years vary a lot (see "yrs up"). Sectors with shorter histories (e.g. XLC since 2018) use fewer years. Not investment advice.</div>`;
  function draw() {
    const mode = document.getElementById('seasMode').value, tr = [];
    secs.forEach(x => {
      const vis = DEFAULT_ON.has(x.symbol) ? true : 'legendonly';
      if (mode != 'cur') tr.push({type:'scatter', mode:'lines', x:SEAS.x, y:x.avg, name:x.symbol, legendgroup:x.symbol, visible:vis,
        line:{color:x.color, width: x.symbol == 'SPY' ? 2.5 : 1.6}, hovertemplate:`${x.label} average: %{y:.1f}%<extra>%{x|%b %d}</extra>`});
      if (mode != 'avg') tr.push({type:'scatter', mode:'lines', x:SEAS.x.slice(0, x.cur.length), y:x.cur, name:x.symbol + (mode == 'cur' ? '' : ' ' + SEAS.year),
        legendgroup:x.symbol, showlegend: mode == 'cur', visible:vis, line:{color:x.color, width:1.4, dash:'dot'},
        hovertemplate:`${x.label} ${SEAS.year}: %{y:.1f}%<extra>%{x|%b %d}</extra>`});
    });
    Plotly.react('seasChart', tr, {paper_bgcolor:'#171a21', plot_bgcolor:'#171a21', font:{color:'#e6e6e6'},
      margin:{l:50, r:20, t:10, b:40}, hovermode:'closest',
      xaxis:{gridcolor:'#2a2f3a', tickformat:'%b', dtick:'M1'}, yaxis:{gridcolor:'#2a2f3a', ticksuffix:'%', zeroline:true, zerolinecolor:'#555'},
      legend:{orientation:'h', y:-0.12},
      shapes:[{type:'line', x0:SEAS.today.replace(/^\d{4}/, SEAS.year), x1:SEAS.today.replace(/^\d{4}/, SEAS.year), yref:'paper', y0:0, y1:1, line:{color:'#5c8dff', width:1, dash:'dash'}}],
      annotations:[{x:SEAS.today.replace(/^\d{4}/, SEAS.year), yref:'paper', y:1, text:'today', showarrow:false, font:{color:'#5c8dff', size:11}, yanchor:'bottom'}]},
      {responsive:true, displaylogo:false});
  }
  document.getElementById('seasMode').onchange = draw;
  draw();
})();

document.getElementById('lookup').onsubmit = e => {
  e.preventDefault();
  lookup(document.getElementById('sym').value.trim().toUpperCase(),
         document.getElementById('per').value, document.getElementById('iv').value);
};
async function lookup(sym, per, iv) {
  const msg = document.getElementById('msg'), btn = document.getElementById('go');
  if (!sym) return;
  if (location.protocol == 'file:') { msg.textContent = `Lookup works on the website. Locally, run: python patterns.py ${sym} --interval ${iv} --period ${per}`; return; }
  btn.disabled = true; msg.textContent = `Analyzing ${sym} (${iv.toUpperCase()}, ${per})…`;
  try {
    const res = await fetch(`/api/patterns?symbol=${encodeURIComponent(sym)}&period=${per}&interval=${iv}`);
    const j = await res.json();
    if (!res.ok || j.error) throw new Error(j.error || res.statusText);
    render(j, true);
    msg.textContent = '';
    window.scrollTo({top: root.offsetTop - 10, behavior:'smooth'});
  } catch (err) { msg.textContent = `${sym}: ${err.message}`; }
  btn.disabled = false;
}
</script></body></html>"""


def write_report(results, path, seas=None):
    opts = "".join(f'<option{" selected" if p == "2y" else ""}>{p}</option>' for p in PERIODS)
    html = (HTML.replace("__GEN__", datetime.now().strftime("%Y-%m-%d %H:%M"))
                .replace("__PERIODS__", opts)
                .replace("__SEAS__", json.dumps(seas))
                .replace("__DATA__", json.dumps(results)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def print_summary(r):
    print(f"\n{r['symbol']}  {r['price']}  ({r['asof']})  ATR {r['atr_pct']}%  swing {r['swing_pct']}%")
    if r["levels"]:
        print("  levels: " + ", ".join(f"{l['role']} {l['price']} x{l['touches']}" for l in r["levels"]))
    if not r["patterns"]:
        print("  no patterns found")
    for p in r["patterns"]:
        tgt = f"  target {p['target']}" if p["target"] else ""
        print(f"  {p['start']} -> {p['end']}  {p['type']:<34} {p['bias']:<8} {p['status']}{tgt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--period", default="2y", choices=PERIODS)
    ap.add_argument("--interval", default="1d", choices=INTERVALS, help="bar size: 1d, 4h, 2h, 1h")
    ap.add_argument("--pct", type=float, help="zigzag swing size in %% (default: 2.5 x median ATR%%)")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--all", action="store_true", help="also list failed / expired patterns")
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-seasonality", action="store_true", help="skip the sector seasonality section")
    args = ap.parse_args()

    results = []
    for sym in [s.upper() for s in args.symbols]:
        try:
            r = analyze(sym, args.period, args.pct, args.refresh, args.all, args.interval,
                        fill_today=args.interval == "1d")
        except Exception as e:
            print(f"{sym}: ERROR {e}")
            continue
        print_summary(r)
        results.append(r)
    if not results:
        sys.exit(1)
    seas = None
    if not args.no_seasonality:
        try:
            import seasonality
            seas = seasonality.seasonality()
        except Exception as e:
            print(f"seasonality: ERROR {e}")
    write_report(results, args.out, seas)
    print(f"\nReport saved: {args.out}")
    if not args.no_browser:
        webbrowser.open("file:///" + os.path.abspath(args.out).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
