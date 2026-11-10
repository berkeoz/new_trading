"""
Rotation watch — when the market turns, which groups should lead next given the economy, have they
turned yet, and are the previous cycle leaders just consolidating?

For every sector / theme ETF of the rotation map (rotation.py), using price returns vs SPY:
  * previous leader  = top third of the universe by 12-month excess return
  * consolidating    = within 15% of its 52-week high, 30-day range at most 4x its daily ATR... and its
                       1-month excess return <= 0 or the rotation map has it Weakening / Lagging
  * next group       = NOT a previous leader, rotation map Improving or Leading with rising relative
                       momentum, 1-month excess return > 0 and price above its 50-day average
  * everything else  = previous leader still leading / no signal
Each ETF also gets the buy & sell checklist (patterns.assess): verdict, buy zone, stop, reward/risk —
"is it at a good point to invest now?".

Economic conditions: the macro data (macro/data.json) gives a growth direction, an inflation direction and
the Fed path priced in futures; the classic economic-cycle map says which groups that regime favors, and
each ETF is marked favored / neutral / unfavored, plus whether it has turned yet (relative strength rising,
beating SPY over a month, above its 50-day) and what is still missing.

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


# ── Economic conditions: which groups the current regime favors ─────────────────────────────
# The classic economic-cycle map (growth direction x inflation direction), with a rates overlay
# from the Fed-funds futures path. These are textbook tendencies, not a tested model.
REGIME_FIT = {
    "Recovery (growth up, inflation down)": {
        "favored": ["XLY", "XLF", "XLI", "XLK", "IWM", "ITB", "IYT", "KRE", "SMH", "IGV", "ARKK"],
        "unfavored": ["XLU", "XLP", "XLE", "XOP"]},
    "Overheat (growth up, inflation up)": {
        "favored": ["XLE", "XOP", "XLB", "XLI", "ITA", "URA", "XLF", "KRE"],
        "unfavored": ["XLU", "XLRE", "ITB", "TAN", "ARKK"]},
    "Stagflation (growth down, inflation up)": {
        "favored": ["XLE", "XOP", "XLP", "XLV", "XLU", "ITA", "URA"],
        "unfavored": ["XLY", "IWM", "ITB", "ARKK", "TAN", "SMH", "KRE"]},
    "Slowdown (growth down, inflation down)": {
        "favored": ["XLU", "XLP", "XLV", "XLRE", "ITB", "XBI"],
        "unfavored": ["XLE", "XOP", "XLB", "XLI", "KRE", "SMH"]},
}
RATE_FIT = {"cuts": {"favored": ["ITB", "XLRE", "IWM", "XLU", "KRE", "TAN", "XBI"], "unfavored": []},
            "hikes": {"favored": ["XLF"], "unfavored": ["ITB", "XLRE", "TAN", "IWM", "ARKK", "XBI"]}}


def macro_regime(path=os.path.join(ROOT, "macro", "data.json")):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    S = {x["id"]: x for x in d.get("series", [])}
    v = lambda k, f="value": (S.get(k) or {}).get(f)
    g, gi, i, ii = 0, [], 0, []
    def sig(lst, ok, cond, text):
        if ok:
            lst.append(("+" if cond else "−") + " " + text)
            return 1 if cond else -1
        return 0
    g += sig(gi, v("A191RL1Q225SBEA") is not None, (v("A191RL1Q225SBEA") or 0) >= 2, f"real GDP {v('A191RL1Q225SBEA')}% annualized")
    g += sig(gi, v("CFNAIMA3") is not None, (v("CFNAIMA3") or 0) >= 0, f"Chicago Fed activity {v('CFNAIMA3')}")
    g += sig(gi, v("PAYEMS") is not None, (v("PAYEMS") or 0) >= 100, f"payrolls +{v('PAYEMS')}k")
    g += sig(gi, v("INDPRO", "ann3m") is not None, (v("INDPRO", "ann3m") or 0) > (v("INDPRO") or 0), f"industrial production {v('INDPRO', 'ann3m')}% 3-month pace vs {v('INDPRO')}% y/y")
    g += sig(gi, v("RSAFS", "ann3m") is not None, (v("RSAFS", "ann3m") or 0) > (v("RSAFS") or 0), f"retail sales {v('RSAFS', 'ann3m')}% 3-month pace vs {v('RSAFS')}% y/y")
    g += sig(gi, v("ICSA", "trend") is not None, v("ICSA", "trend") in ("falling", "stable"), f"jobless claims {v('ICSA')}k, {v('ICSA', 'trend')}")
    i += sig(ii, v("CPILFESL", "ann3m") is not None, (v("CPILFESL", "ann3m") or 0) > (v("CPILFESL") or 0), f"core CPI {v('CPILFESL', 'ann3m')}% 3-month pace vs {v('CPILFESL')}% y/y")
    i += sig(ii, v("PCEPILFE", "ann3m") is not None, (v("PCEPILFE", "ann3m") or 0) > (v("PCEPILFE") or 0), f"core PCE {v('PCEPILFE', 'ann3m')}% 3-month pace vs {v('PCEPILFE')}% y/y")
    i += sig(ii, v("DCOILWTICO") is not None, v("DCOILWTICO", "trend") == "rising", f"WTI ${v('DCOILWTICO')}, {v('DCOILWTICO', 'trend')}")
    i += sig(ii, v("T5YIE") is not None, v("T5YIE", "trend") == "rising", f"5-year inflation expectations {v('T5YIE')}%, {v('T5YIE', 'trend')}")
    ff = d.get("fed_futures") or {}
    path_ = ff.get("months") or []
    rates, rtxt = "neutral", "no clear Fed path"
    if ff.get("effr") is not None and path_:
        delta = path_[-1]["implied"] - ff["effr"]
        rates = "hikes" if delta >= 0.15 else "cuts" if delta <= -0.15 else "neutral"
        rtxt = f"futures price fed funds {path_[-1]['implied']:.2f}% by {path_[-1]['month']} vs {ff['effr']:.2f}% now ({'+' if delta >= 0 else ''}{delta * 100:.0f} bp)"
    gu, iu = g >= 0, i >= 0          # a tie counts as growth up / inflation up (the cautious reading for inflation)
    name = ("Recovery (growth up, inflation down)" if gu and not iu else "Overheat (growth up, inflation up)" if gu and iu
            else "Stagflation (growth down, inflation up)" if iu else "Slowdown (growth down, inflation down)")
    borderline = abs(g) <= 1 or abs(i) <= 1
    return {"regime": name, "borderline": borderline, "growth_score": g, "growth_signals": gi, "inflation_score": i, "inflation_signals": ii,
            "rates": rates, "rates_text": rtxt, "as_of": d.get("generated_et"), "pillars": d.get("regimes")}


def macro_fit(sym, m):
    if not m:
        return "neutral", ""
    r, rr = REGIME_FIT[m["regime"]], RATE_FIT.get(m["rates"], {"favored": [], "unfavored": []})
    score, why = 0, []
    if sym in r["favored"]:
        score += 1; why.append(m["regime"].split(" (")[0].lower() + " favors it")
    if sym in r["unfavored"]:
        score -= 1; why.append(m["regime"].split(" (")[0].lower() + " works against it")
    if sym in rr["favored"]:
        score += 1; why.append(f"rate {m['rates']} help it")
    if sym in rr["unfavored"]:
        score -= 1; why.append(f"rate {m['rates']} hurt it")
    return ("favored" if score > 0 else "unfavored" if score < 0 else "neutral"), "; ".join(why)


def run():
    px, hi, lo = _prices()
    syms = [e[0] for e in UNIVERSE if e[0] in px.columns]
    f = features(px, hi, lo)
    i = len(px) - 1
    cls = classify(f, i, syms)
    spy = px["SPY"]
    regime = P.market_regime(spy)
    macro = macro_regime()
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
        # economic conditions: is this a group the current regime favors, and has it turned yet?
        r["fit"], r["fit_why"] = macro_fit(sym, macro)
        turned = {"rotation map Improving or Leading with rising momentum": r["quadrant"] in ("Improving", "Leading") and r["rm"] > 0,
                  "beating SPY over the last month": r["ex1"] > 0,
                  "above its 50-day average": bool(f["above50"].iloc[i][sym])}
        r["turn_missing"] = [k for k, ok in turned.items() if not ok]
        if r["fit"] == "favored":
            if r["class"] == "Next group candidate":
                r["follow"] = "Favored and turning now"
            elif r["class"] == "Previous leader, still leading" or not r["turn_missing"]:
                r["follow"] = "Favored and already leading"
            else:
                r["follow"] = "Favored, not turned yet"
        else:
            r["follow"] = "Against the regime" if r["fit"] == "unfavored" else "Neutral for the regime"
        rows.append(r)
    try:
        th = json.load(open(os.path.join(ROOT, "thesis", "state.json"), encoding="utf-8")).get("outlook") or {}
        outlook = {"regime": th.get("regime"), "themes": [{"name": t.get("name"), "symbols": t.get("symbols")} for t in th.get("themes", [])], "id": th.get("id")}
    except Exception:
        outlook = None
    out = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "asof": f"{px.index[-1]:%Y-%m-%d}",
           "spy_ytd_pct": round(float(spy.iloc[-1] / spy[spy.index.year == px.index[-1].year].iloc[0] - 1) * 100, 1),
           "regime": regime, "macro": macro, "outlook": outlook, "rows": rows, "test": backtest(px, f, syms),
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
    m = d.get("macro")
    if m:
        L.append(f"  economic regime: {m['regime']} (growth score {m['growth_score']:+}, inflation score {m['inflation_score']:+}); rates: {m['rates_text']}")
        for k in ("Favored and turning now", "Favored, not turned yet", "Favored and already leading"):
            xs = [r for r in d["rows"] if r.get("follow") == k]
            if xs:
                L.append(f"  {k}: " + "; ".join(f"{r['symbol']} ({r['label']}; {r['fit_why']}; checklist {r.get('check', {}).get('verdict', '?')}"
                                                 + (f"; missing: {', '.join(r['turn_missing'])}" if k == 'Favored, not turned yet' else "") + ")" for r in xs))
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
