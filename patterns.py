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
    python patterns.py QQQ --pct 4          # fixed 4% zigzag swing instead of auto
    python patterns.py QQQ --refresh        # ignore cached prices
    python patterns.py QQQ --all            # include failed / expired patterns
    python patterns.py QQQ --no-browser

Prices are cached in data/<SYMBOL>_<period>.csv (git-ignored, refreshed once per day).
The report (patterns.html) is served by Vercel at /patterns; its lookup box calls
api/patterns.py to analyze any ticker on demand.
"""

import sys, os, json, argparse, webbrowser
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

ROOT     = os.path.dirname(os.path.abspath(__file__))
# Vercel functions can only write to /tmp
DATA_DIR = "/tmp/data" if os.environ.get("VERCEL") else os.path.join(ROOT, "data")
REPORT   = os.path.join(ROOT, "patterns.html")
DEFAULT_SYMBOLS = ["QQQ", "SPY", "SOXX"]
PERIODS  = ("6mo", "1y", "2y", "5y", "10y")
WARMUP_PERIOD = {"6mo": "2y", "1y": "2y", "2y": "5y", "5y": "10y", "10y": "max"}


# ── Data ───────────────────────────────────────────────────────────────────────
def load_prices(sym, period="2y", refresh=False):
    """Daily OHLCV, cached per symbol+period; re-downloaded if cache is from an earlier day."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{sym}_{period}.csv")
    if not refresh and os.path.exists(path):
        if date.fromtimestamp(os.path.getmtime(path)) == date.today():
            return pd.read_csv(path, index_col=0, parse_dates=True)
    df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"no price data for {sym}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_localize(None)
    if len(df) < 60:
        raise ValueError(f"only {len(df)} days of data for {sym}")
    df.to_csv(path)
    return df


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
        "start": str(d[pivots[0]["i"]].date()), "end": str(d[pivots[-1]["i"]].date()),
        "i0": int(pivots[0]["i"]), "i1": int(pivots[-1]["i"]),
        "pivots": [{"date": str(d[p["i"]].date()), "price": round(p["price"], 2)} for p in pivots],
        "lines": lines,       # [[date0, price0, date1, price1], ...]
        "target": None if target is None else round(float(target), 2),
        "note": note,
    }


def _seg(df, i0, p0, i1, p1):
    return [str(df.index[i0].date()), round(float(p0), 2), str(df.index[i1].date()), round(float(p1), 2)]


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
        return f"confirmed {df.index[k].date()}"
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
            f"{rs['i']-ls['i']} bars, neckline slope {slope / neck(end_i) * 100:+.2f}%/day"))
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
                        status, end_i = f"broke {d} {df.index[k].date()}", k
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
                        f"{i1-i0} bars, upper {mu/price*100:+.2f}%/day, lower {ml/price*100:+.2f}%/day"))
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
            status = f"broke {'down' if t['support'] else 'up'} {df.index[t['brk']].date()}"
        elif abs(df["Close"].iloc[-1] - line(t["end"])) / df["Close"].iloc[-1] > 0.15:
            status = "expired"                   # never broken, but price has moved far away
        else:
            status = "active"
        out.append(_pattern(
            f"{word.capitalize()} {role} trendline", "trendline", bias, tch, df, status,
            [_seg(df, tch[0]["i"], line(tch[0]["i"]), t["end"], line(t["end"]))], None,
            f"{len(tch)} touches over {tch[-1]['i']-tch[0]['i']} bars, {m/p['price']*100:+.2f}%/day"
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
        status = f"broke {f['dir']} {df.index[f['brk']].date()}" if f["brk"] is not None else "active"
        lo_off, hi_off = (0, off) if t["support"] else (off, 0)
        width = abs(off) / line(end) * 100
        out.append(_pattern(
            f"Parallel channel ({word})", "channel", bias, pts, df, status,
            [_seg(df, i0, line(i0) + lo_off, end, line(end) + lo_off),
             _seg(df, i0, line(i0) + hi_off, end, line(end) + hi_off)], None,
            f"{len(t['touches'])}+{len(f['ptch'])} touches, width {width:.1f}%, "
            f"{m/p['price']*100:+.2f}%/day"
            + (f", now {line(end)+lo_off:.2f}-{line(end)+hi_off:.2f}" if f["brk"] is None else "")))
        if len(out) >= limit:
            break
    return out


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
def analyze(sym, period="2y", pct=None, refresh=False, show_all=False):
    # download a longer period so moving averages (up to 200+ days) are already
    # warmed up at the start of the chart; patterns use only the requested period
    full = load_prices(sym, WARMUP_PERIOD[period], refresh)
    months = 6 if period == "6mo" else 12 * int(period[:-1])
    start = full.index[-1] - pd.DateOffset(months=months)
    df = full[full.index > start]
    warm = full["Close"][full.index <= start].tail(300).round(2).tolist()
    a = atr_pct(df)
    swing = pct / 100 if pct else max(0.03, round(2.5 * a, 3))
    tol = max(0.015, swing * 0.4)        # "equal" price tolerance
    flat = 0.0004                        # |slope| < 0.04%/day counts as flat
    piv = zigzag(df, swing)
    tl_cands = _trendline_candidates(df, piv, tol)
    pats = (find_double(df, piv, tol) + find_head_shoulders(df, piv, tol)
            + find_line_patterns(df, piv, tol, flat)
            + find_parallel_channels(df, piv, tol, flat, tl_cands)
            + find_trendlines(df, piv, tol, flat, tl_cands))
    if not show_all:   # hide patterns that never resolved or were invalidated
        pats = [p for p in pats if p["status"] not in ("failed", "expired")]
    pats.sort(key=lambda p: p["i1"], reverse=True)
    price = float(df["Close"].iloc[-1])
    return {
        "symbol": sym, "period": period, "price": round(price, 2), "asof": str(df.index[-1].date()),
        "atr_pct": round(a * 100, 2), "swing_pct": round(swing * 100, 1),
        "pivots": [{"date": str(df.index[p["i"]].date()), "price": round(p["price"], 2),
                    "kind": p["kind"], "confirmed": p["confirmed"]} for p in piv],
        "patterns": pats,
        "levels": find_levels(piv, price, tol)[:6],
        "warm": warm,   # closes before the chart window, for moving-average warm-up
        "ohlc": {
            "x": [str(d.date()) for d in df.index],
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
</style></head><body><main>
<div class="top">
  <div><h1>Chart Formations</h1>
  <div class="sub">Generated __GEN__ · swing-pivot pattern detection · mechanical candidates, not trade advice. Click a table row to zoom.</div></div>
  <a href="/">&larr; Daily signals</a>
</div>
<form class="lookup" id="lookup">
  <input id="sym" placeholder="Ticker, e.g. NVDA" autocomplete="off" spellcheck="false" required>
  <select id="per">__PERIODS__</select>
  <button id="go">Analyze</button>
  <span id="msg" class="sub"></span>
</form>
<div id="root"></div></main>
<script>
const DATA = __DATA__;
const COL = {bullish:'#26a69a', bearish:'#ef5350', neutral:'#f5b041'};
const CATS = [['reversal','Reversals'],['triangle','Triangles & wedges'],['channel','Channels'],['trendline','Trendlines']];
const root = document.getElementById('root');
let seq = 0;

// ── Moving averages (computed in the browser so they are configurable) ──
const MA_DEFAULT = 'EMA9, EMA21, SMA50, SMA200, RSI14, MACD12/26/9';
const MA_COLORS = ['#ffd54f','#4fc3f7','#ba68c8','#ff8a65','#e0e0e0','#81c784'];
let maCfg = MA_DEFAULT;
try { maCfg = localStorage.getItem('maCfg') || MA_DEFAULT; } catch (e) {}
if (maCfg.replace(/\s/g, '') == 'EMA9,EMA21,SMA50,SMA200') maCfg = MA_DEFAULT;   // upgrade old saved default
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
function parseMA(txt) {   // "EMA9, SMA 50, MA200" -> [{kind, n, name}], sorted fast -> slow
  const seen = new Set(), out = [];
  for (const m of txt.matchAll(/(EMA|SMA|MA)\s*(\d+)/gi)) {
    const kind = m[1].toUpperCase() == 'EMA' ? 'EMA' : 'SMA', n = +m[2];
    if (n < 2 || n > 400 || seen.has(kind + n)) continue;
    seen.add(kind + n); out.push({kind, n, name: kind + n});
  }
  return out.sort((a, b) => a.n - b.n || (a.kind == 'EMA' ? -1 : 1)).slice(0, 6);
}
const SLOPE_BARS = 5;     // slope = change of the MA over the last 5 trading days
function verdict(sc, prev) {
  if (sc == null) return ['No data', 'neutral'];
  const d = prev == null ? 0 : sc - prev;
  if (sc >= 0.5)  return d <= -0.4 ? ['Bullish, weakening', 'neutral'] : ['Bullish', 'bullish'];
  if (sc <= -0.5) return d >= 0.4 ? ['Bearish, improving', 'neutral'] : ['Bearish', 'bearish'];
  if (d >= 0.3)  return ['Turning bullish', 'bullish'];
  if (d <= -0.3) return ['Turning bearish', 'bearish'];
  return ['Neutral', 'neutral'];
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
  s.innerHTML = `<div class="head"><h2>${r.symbol} <span class="sub">${r.price} · ${r.asof} · ${r.period} · ATR ${r.atr_pct}% · swing ${r.swing_pct}% · ${r.patterns.length} patterns (${open} open)</span></h2>
      ${prepend ? '<button class="x" title="Remove">&times;</button>' : ''}</div>
    <div>${r.levels.map(l => `<span class="lv ${l.role=='support'?'bullish':'bearish'}">${l.role} ${l.price} ×${l.touches}</span>`).join('')}</div>
    <div class="filters">${CATS.map(([k,t]) => `<label><input type="checkbox" data-cat="${k}" checked> ${t} </label>`).join('')}
      <label><input type="checkbox" data-opt="levels" checked> S/R levels</label>
      <label>Show <select data-opt="show"><option value="recent">open + last 6 months</option><option value="open">open / active only</option><option value="all">all 2 years</option></select></label></div>
    <div class="ma-bar">Moving averages <input type="text" class="ma-cfg" spellcheck="false" title="Comma-separated: up to 6 of EMA9, SMA50 …, plus RSI14 and MACD12/26/9">
      <label><input type="checkbox" class="ma-show" checked> lines</label>
      <label><input type="checkbox" class="ma-strip" checked> trend strip</label>
      <span>· hover the chart to see the state on any day, click to pin it</span></div>
    <div class="state"></div>
    <div class="chart"></div>
    <div class="wrap"><table><thead><tr><th>Pattern</th><th>Bias</th><th>From</th><th>To</th><th>Status</th><th>Target</th><th>Detail</th></tr></thead><tbody></tbody></table></div>`;
  prepend ? root.prepend(s) : root.appendChild(s);
  const div = s.querySelector('.chart'), tbody = s.querySelector('tbody');
  const boxes = [...s.querySelectorAll('.filters input, .filters select')];
  const cutoff = new Date(new Date(r.asof) - 183*864e5);
  const lastDate = p => new Date(Math.max(+new Date(p.end), ...p.lines.map(L => +new Date(L[2]))));
  if (prepend) s.querySelector('.x').onclick = () => { Plotly.purge(div); s.remove(); };

  const C = r.ohlc.close, D = r.ohlc.x, last = C.length - 1;
  const idx = Object.fromEntries(D.map((d, i) => [d, i]));
  const cfgIn = s.querySelector('.ma-cfg'), stateDiv = s.querySelector('.state');
  cfgIn.value = maCfg;
  let mas = [], score = [], pairs = [], osc = {}, pinned = last;

  function computeMA() {
    const W = r.warm || [], CW = W.concat(C);   // warm-up closes so long MAs exist from day 1
    mas = parseMA(cfgIn.value).map((m, k) => ({...m, color: MA_COLORS[k],
      v: (m.kind == 'EMA' ? ema(CW, m.n) : sma(CW, m.n)).slice(W.length)}));
    // daily score in [-1, 1]: price above/below each MA, each MA rising/falling,
    // and whether each faster MA is above the next slower one
    score = C.map((c, i) => {
      let pts = 0, tot = 0;
      mas.forEach((m, k) => {
        const v = m.v[i]; if (v == null) return;
        pts += c > v ? 1 : -1; tot++;
        const pv = m.v[i - SLOPE_BARS];
        if (pv != null) { pts += v > pv ? 1 : -1; tot++; }
        const nx = mas[k + 1];
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
    if (mas.length >= 2) add(mas[0], mas[1]);
    if (mas.length >= 3) add(mas[mas.length - 2], mas[mas.length - 1]);
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
        const dir = d == null ? '' : Math.abs(d) < 2 ? ' · flat over 5d' :
          ` · ${d > 0 ? b('bullish', 'rising') : b('bearish', 'falling')} ${d > 0 ? '+' : ''}${d.toFixed(1)} over 5d`;
        // simple divergence: price at a 20-day high/low but RSI is not
        let div = '';
        if (i >= 20) {
          const win = C.slice(i - 20, i + 1), rw = osc.rsi.slice(i - 20, i + 1).filter(x => x != null);
          if (C[i] >= Math.max(...win) && v < Math.max(...rw) - 3) div = ' · ' + b('bearish', 'bearish divergence (new 20d high, weaker RSI)');
          if (C[i] <= Math.min(...win) && v > Math.min(...rw) + 3) div = ' · ' + b('bullish', 'bullish divergence (new 20d low, firmer RSI)');
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
          (e ? `<br>&nbsp;&nbsp;last MACD ${e.up ? b('bullish', 'bullish cross') : b('bearish', 'bearish cross')} ${D[e.i]} (${i - e.i} trading days before)` : ''));
      }
    }
    return out.join('<br>');
  }

  function showState(i) {
    if (i == null || i < 0) i = last;
    const sc = score[i], prev = i >= SLOPE_BARS ? score[i - SLOPE_BARS] : null;
    const [txt, cls] = verdict(sc, prev);
    const pct = x => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';
    const rows = mas.map(m => {
      const v = m.v[i];
      if (v == null) return `<tr><td style="color:${m.color}">${m.name}</td><td colspan="3" class="sub">not enough history</td></tr>`;
      const pv = m.v[i - SLOPE_BARS], sl = pv != null ? v / pv - 1 : null;
      const dir = sl == null ? '' : Math.abs(sl) < 0.001 ? '→ flat' : sl > 0 ? '↗ rising' : '↘ falling';
      return `<tr><td style="color:${m.color}">${m.name}</td><td>${v.toFixed(2)}</td>
        <td class="${C[i] > v ? 'bullish' : 'bearish'}">price ${C[i] > v ? 'above' : 'below'} ${pct(C[i] / v - 1)}</td>
        <td class="${sl == null ? 'sub' : sl > 0.001 ? 'bullish' : sl < -0.001 ? 'bearish' : 'neutral'}">${dir}${sl == null ? '' : ' ' + pct(sl)}</td></tr>`;
    }).join('');
    const cr = pairs.map(p => {
      if (p.a.v[i] == null || p.b.v[i] == null) return '';
      const e = p.ev.filter(x => x.i <= i).pop();
      const now = p.a.v[i] > p.b.v[i] ? 'above' : 'below';
      return `${p.a.name} is <b class="${now == 'above' ? 'bullish' : 'bearish'}">${now}</b> ${p.b.name}` +
        (e ? ` · last ${p.label(e.up)} ${D[e.i]} (${i - e.i} trading days before)` : ' · no cross in range');
    }).filter(Boolean).join('<br>');
    stateDiv.innerHTML = `<div>
        <div class="sub">${i == pinned && i != last ? 'pinned · ' : ''}${D[i]} · close ${C[i].toFixed(2)}</div>
        <div class="verdict ${cls}">${txt}</div>
        <div class="meter">${sc == null ? '' : `<i style="left:calc(${(sc + 1) * 50}% - 1px)"></i>`}</div>
        <div class="sub">MA score ${sc == null ? '–' : sc.toFixed(2)} (5 days earlier ${prev == null ? '–' : prev.toFixed(2)}) · −1 bearish … +1 bullish</div></div>
      <div><table><tr><th>MA</th><th>Value</th><th>Price vs MA</th><th>Slope (5d)</th></tr>${rows ||
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
    pairs.forEach(p => p.ev.forEach(e => traces.push({type:'scatter', mode:'markers', x:[D[e.i]], y:[p.b.v[e.i]],
      marker:{symbol: e.up ? 'triangle-up' : 'triangle-down', size:11, color: e.up ? '#26a69a' : '#ef5350', line:{color:'#fff', width:1}},
      hovertemplate:`${p.a.name}/${p.b.name} ${p.label(e.up)}<br>%{x}<extra></extra>`})));
    const strip = s.querySelector('.ma-strip').checked;
    if (strip) traces.push({type:'bar', x:D, y:D.map((_, i) => score[i] == null ? 0 : 1), yaxis:'y2',
      marker:{color:score.map(stripColor)}, hovertext:score.map((sc, i) => verdict(sc, score[i - SLOPE_BARS])[0]),
      hoverinfo:'text+x', showlegend:false});
    const shapes = [], ann = [];
    pats.forEach(p => {
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
      xaxis:{range:keep, rangeslider:{visible:false}, gridcolor:'#2a2f3a', rangebreaks:[{bounds:['sat','mon']}],
        rangeselector:{bgcolor:'#232836', activecolor:'#5c8dff', font:{color:'#e6e6e6'}, buttons:[
          {count:3,label:'3M',step:'month',stepmode:'backward'},{count:6,label:'6M',step:'month',stepmode:'backward'},
          {count:1,label:'1Y',step:'year',stepmode:'backward'},{count:2,label:'2Y',step:'year',stepmode:'backward'},
          {step:'all',label:'All'}]}},
      yaxis:{gridcolor:'#2a2f3a', autorange:true, domain: strip ? [0.09, 1] : [0, 1]},
      yaxis2:{domain:[0, 0.05], visible:false, fixedrange:true, range:[0, 1]}, bargap:0
    }, {responsive:true, displaylogo:false});
    if (!div._maEvents) {
      // map the mouse x-position to the nearest trading day (works anywhere on the chart)
      div._maEvents = true;
      const dayAt = ev => {
        const xa = div._fullLayout.xaxis, px = ev.clientX - div.getBoundingClientRect().left - xa._offset;
        if (px < 0 || px > xa._length) return null;
        const d = new Date(xa.p2c(px)).toISOString().slice(0, 10);
        let lo = 0, hi = last;                       // first trading day >= d
        while (lo < hi) { const m = (lo + hi) >> 1; D[m] < d ? lo = m + 1 : hi = m; }
        return lo;
      };
      let raf = 0, down = null;
      div.addEventListener('mousemove', ev => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => { const i = dayAt(ev); showState(i == null ? pinned : i); });
      });
      div.addEventListener('mouseleave', () => { cancelAnimationFrame(raf); showState(pinned); });
      div.addEventListener('mousedown', ev => down = [ev.clientX, ev.clientY]);
      div.addEventListener('mouseup', ev => {          // a click (not a drag-zoom) pins the day
        if (down && Math.abs(ev.clientX - down[0]) + Math.abs(ev.clientY - down[1]) < 4) {
          const i = dayAt(ev); if (i != null) { pinned = i; showState(i); }
        }
        down = null;
      });
    }
    tbody.innerHTML = pats.map((p, k) => `<tr class="pat" data-k="${k}"><td>${p.type} <span class="tag">${p.cat}</span></td><td class="${p.bias}">${p.bias}</td><td>${p.start}</td><td>${p.end}</td><td>${p.status}</td><td>${p.target ?? ''}</td><td class="sub">${p.note}</td></tr>`).join('')
      || '<tr><td colspan="7" class="sub">No patterns for the selected filters.</td></tr>';
    tbody.querySelectorAll('tr.pat').forEach(tr => tr.onclick = () => {
      const p = pats[+tr.dataset.k], pad = 20*864e5;
      const ends = p.lines.map(L => +new Date(L[2])).concat([+new Date(p.end)]);
      const a = new Date(+new Date(p.start) - pad), b = new Date(Math.max(...ends) + pad);
      Plotly.relayout(div, {'xaxis.range':[a.toISOString().slice(0,10), b.toISOString().slice(0,10)], 'yaxis.autorange':true});
      div.scrollIntoView({behavior:'smooth', block:'center'});
    });
  }
  boxes.forEach(b => b.onchange = draw);
  s.querySelectorAll('.ma-show, .ma-strip').forEach(b => b.onchange = draw);
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

document.getElementById('lookup').onsubmit = async e => {
  e.preventDefault();
  const sym = document.getElementById('sym').value.trim().toUpperCase();
  const per = document.getElementById('per').value;
  const msg = document.getElementById('msg'), btn = document.getElementById('go');
  if (!sym) return;
  if (location.protocol == 'file:') { msg.textContent = 'Lookup works on the website. Locally, run: python patterns.py ' + sym; return; }
  btn.disabled = true; msg.textContent = `Analyzing ${sym} (${per})…`;
  try {
    const res = await fetch(`/api/patterns?symbol=${encodeURIComponent(sym)}&period=${per}`);
    const j = await res.json();
    if (!res.ok || j.error) throw new Error(j.error || res.statusText);
    render(j, true);
    msg.textContent = '';
    window.scrollTo({top: root.offsetTop - 10, behavior:'smooth'});
  } catch (err) { msg.textContent = `${sym}: ${err.message}`; }
  btn.disabled = false;
};
</script></body></html>"""


def write_report(results, path):
    opts = "".join(f'<option{" selected" if p == "2y" else ""}>{p}</option>' for p in PERIODS)
    html = (HTML.replace("__GEN__", datetime.now().strftime("%Y-%m-%d %H:%M"))
                .replace("__PERIODS__", opts)
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
    ap.add_argument("--pct", type=float, help="zigzag swing size in %% (default: 2.5 x median ATR%%)")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--all", action="store_true", help="also list failed / expired patterns")
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    results = []
    for sym in [s.upper() for s in args.symbols]:
        try:
            r = analyze(sym, args.period, args.pct, args.refresh, args.all)
        except Exception as e:
            print(f"{sym}: ERROR {e}")
            continue
        print_summary(r)
        results.append(r)
    if not results:
        sys.exit(1)
    write_report(results, args.out)
    print(f"\nReport saved: {args.out}")
    if not args.no_browser:
        webbrowser.open("file:///" + os.path.abspath(args.out).replace(os.sep, "/"))


if __name__ == "__main__":
    main()
