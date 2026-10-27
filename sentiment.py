"""
Sentiment & leadership layer: fear / complacency gauges, breadth and the intermarket ratios
that often turn before the S&P 500, combined into one "risk appetite" score.

Each indicator becomes a daily signal from -1 (risk-off) to +1 (risk-on):
  * levels (VIX, SKEW, MOVE): percentile vs the past 10 years, high = risk-off;
  * VIX term structure: VIX / VIX3M above 1 (backwardation) = panic;
  * ratios (small vs large caps, equal vs cap weight, semis, transports, discretionary vs
    staples, high beta vs low volatility, copper vs gold, junk vs Treasuries) and Bitcoin:
    position vs their 50-day average;
  * breadth: share of sector ETFs above their 50-day and 200-day averages.

Test (model): does each signal lead SPY's return over the next 21 trading days? Rank
correlation (IC) on 2012-2021 decides whether it counts and in which direction (a fear gauge
is often contrarian); the composite uses those weights and is then checked on 2022 onwards,
which the fitting never saw. The model is refit at most weekly.

Routine-sourced gauges (CNN Fear & Greed, AAII survey, put/call ratio, NAAIM) are written by
the market-brief routine to context/sentiment_ext.json and shown alongside, not in the model.

    python sentiment.py          # refresh context/sentiment.json (and the model if older than 7 days)
    python sentiment.py --refit  # refit the model now
"""

import os, sys, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(ROOT, "context")
OUT = os.path.join(CDIR, "sentiment.json")
MODEL = os.path.join(CDIR, "sentiment_model.json")
CACHE = os.path.join(ROOT, "data", "sentiment_prices.csv")
TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
H = 21
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
TICKERS = ["SPY", "^VIX", "^VIX3M", "^SKEW", "^MOVE", "RSP", "IWM", "SMH", "IYT", "SPHB", "SPLV",
           "HG=F", "GC=F", "HYG", "IEF", "BTC-USD"] + SECTORS

# key, label, group, how to read it
IND = [
    ("vix", "VIX (fear gauge)", "Fear & complacency", "percentile of the VIX vs 10 years; high = fear"),
    ("vix_term", "VIX vs 3-month VIX", "Fear & complacency", "above 1 = near-term panic (backwardation)"),
    ("skew", "SKEW (tail-risk demand)", "Fear & complacency", "percentile; high = investors paying up for crash protection"),
    ("move", "MOVE (bond volatility)", "Fear & complacency", "percentile; high = stress in rates"),
    ("breadth50", "Sectors above their 50-day", "Breadth", "share of the 11 sector ETFs above their 50-day average"),
    ("breadth200", "Sectors above their 200-day", "Breadth", "share of the 11 sector ETFs above their 200-day average"),
    ("rsp_spy", "Equal-weight vs cap-weight S&P (RSP/SPY)", "Breadth", "rising = broad participation, falling = narrow leadership"),
    ("iwm_spy", "Small caps vs S&P (IWM/SPY)", "Leaders", "small caps leading = risk appetite"),
    ("smh_spy", "Semis vs S&P (SMH/SPY)", "Leaders", "semis often lead the market"),
    ("iyt_spy", "Transports vs S&P (IYT/SPY)", "Leaders", "Dow-theory confirmation"),
    ("xly_xlp", "Discretionary vs staples (XLY/XLP)", "Leaders", "offense vs defense"),
    ("sphb_splv", "High beta vs low volatility (SPHB/SPLV)", "Leaders", "risk appetite inside equities"),
    ("cu_au", "Copper vs gold", "Intermarket", "growth expectations vs safety"),
    ("hyg_ief", "Junk bonds vs Treasuries (HYG/IEF)", "Intermarket", "credit risk appetite; often leads stocks"),
    ("btc", "Bitcoin", "Intermarket", "speculative risk appetite"),
]


def prices(refresh=False):
    """Daily closes for TICKERS since 2008, cached for the day."""
    if not refresh and os.path.exists(CACHE) and datetime.fromtimestamp(os.path.getmtime(CACHE)).date() == datetime.now().date():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    df = yf.download(TICKERS, start="2008-01-01", interval="1d", auto_adjust=True, progress=False)["Close"]
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[df["SPY"].notna()]                      # US trading days; BTC and futures aligned to them
    df = df.ffill(limit=5)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE)
    return df


def _pct_rank(s, win=2520, minp=500):
    return s.rolling(win, min_periods=minp).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)


def _vs_ma(s, n=50, scale=0.03):
    return ((s / s.rolling(n).mean() - 1) / scale).clip(-1, 1)


def signals(px):
    """Signal per indicator, -1 (risk-off) .. +1 (risk-on), plus the raw values for display."""
    sig, raw = {}, {}
    for k, t in (("vix", "^VIX"), ("skew", "^SKEW"), ("move", "^MOVE")):
        if t in px and px[t].notna().sum() > 600:
            p = _pct_rank(px[t])
            sig[k], raw[k] = -(p - 0.5) * 2, px[t]
    if "^VIX3M" in px:
        r = px["^VIX"] / px["^VIX3M"]
        sig["vix_term"], raw["vix_term"] = (-(r - 0.9) / 0.1).clip(-1, 1), r
    sec = px[[s for s in SECTORS if s in px]]
    above50 = (sec > sec.rolling(50).mean()).where(sec.notna()).mean(axis=1)
    above200 = (sec > sec.rolling(200).mean()).where(sec.notna()).mean(axis=1)
    sig["breadth50"], raw["breadth50"] = (above50 - 0.5) * 2, above50 * 100
    sig["breadth200"], raw["breadth200"] = (above200 - 0.5) * 2, above200 * 100
    for k, a, b in (("rsp_spy", "RSP", "SPY"), ("iwm_spy", "IWM", "SPY"), ("smh_spy", "SMH", "SPY"), ("iyt_spy", "IYT", "SPY"),
                    ("xly_xlp", "XLY", "XLP"), ("sphb_splv", "SPHB", "SPLV"), ("cu_au", "HG=F", "GC=F"), ("hyg_ief", "HYG", "IEF")):
        if a in px and b in px:
            r = px[a] / px[b]
            sig[k], raw[k] = _vs_ma(r), r
    if "BTC-USD" in px:
        sig["btc"], raw["btc"] = _vs_ma(px["BTC-USD"], 50, 0.15), px["BTC-USD"]
    return pd.DataFrame(sig), pd.DataFrame(raw)


def _ic(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 250:
        return None
    return float(a[m].rank().corr(b[m].rank()))


def fit(px, sig):
    spy = px["SPY"]
    fwd = spy.shift(-H) / spy - 1
    tr = sig.index <= TRAIN_END
    te = sig.index >= TEST_START
    model = {"horizon_days": H, "train": f"2012 to {TRAIN_END[:4]}", "test": f"{TEST_START[:4]} onwards", "indicators": {}}
    w = {}
    for k in sig:
        ic_tr = _ic(sig[k][tr & (sig.index >= "2012-01-01")], fwd[tr & (sig.index >= "2012-01-01")])
        ic_te = _ic(sig[k][te], fwd[te])
        # quintiles on the test period: SPY's average next-month return when the signal is lowest / highest
        q = {}
        s_te, f_te = sig[k][te], fwd[te]
        mm = s_te.notna() & f_te.notna()
        if mm.sum() > 200:
            cut = pd.qcut(s_te[mm].rank(method="first"), 5, labels=False)
            q = {"lowest": round(float(f_te[mm][cut == 0].mean()) * 100, 2), "highest": round(float(f_te[mm][cut == 4].mean()) * 100, 2)}
        # robustness: the same direction in both halves of the fitting period (no test data used)
        h1 = (sig.index >= "2012-01-01") & (sig.index <= "2016-12-31")
        h2 = (sig.index >= "2017-01-01") & tr
        ic1, ic2 = _ic(sig[k][h1], fwd[h1]), _ic(sig[k][h2], fwd[h2])
        keep = bool(ic_tr is not None and abs(ic_tr) >= 0.03 and ic1 is not None and ic2 is not None
                and np.sign(ic1) == np.sign(ic2) == np.sign(ic_tr) and min(abs(ic1), abs(ic2)) >= 0.02)
        model["indicators"][k] = {"ic_train": None if ic_tr is None else round(ic_tr, 3), "ic_test": None if ic_te is None else round(ic_te, 3),
                                  "ic_2012_16": None if ic1 is None else round(ic1, 3), "ic_2017_21": None if ic2 is None else round(ic2, 3),
                                  "used": keep, "direction": ("contrarian" if ic_tr < 0 else "trend") if keep else None, "test_quintiles": q}
        if keep:
            w[k] = ic_tr
    tot = sum(abs(x) for x in w.values()) or 1
    model["weights"] = {k: round(v / tot, 4) for k, v in w.items()}
    comp = composite(sig, model["weights"])
    model["composite"] = {"ic_train": round(_ic(comp[tr & (comp.index >= "2012-01-01")], fwd[tr & (comp.index >= "2012-01-01")]) or 0, 3),
                          "ic_test": round(_ic(comp[te], fwd[te]) or 0, 3)}
    mm = comp[te].notna() & fwd[te].notna()
    cut = pd.qcut(comp[te][mm].rank(method="first"), 5, labels=False)
    model["composite"]["test_quintiles"] = {f"q{i + 1}": round(float(fwd[te][mm][cut == i].mean()) * 100, 2) for i in range(5)}
    model["composite"]["spy_avg_month"] = round(float(fwd[te][mm].mean()) * 100, 2)
    model["fitted"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return model


def composite(sig, weights):
    """Weighted sum of the signals (weights carry the direction), scaled to -100..+100."""
    cols = [k for k in weights if k in sig]
    if not cols:
        return pd.Series(np.nan, index=sig.index)
    W = pd.Series({k: weights[k] for k in cols})
    s = sig[cols]
    num = (s * W).sum(axis=1, min_count=1)
    den = s.notna().mul(W.abs()).sum(axis=1)
    return (num / den.replace(0, np.nan) * 100).clip(-100, 100)


def _reading(k, s, model):
    m = model["indicators"].get(k, {})
    risk = "risk-on" if s > 0.25 else "risk-off" if s < -0.25 else "neutral"
    if m.get("direction") == "contrarian" and abs(s) > 0.6:
        return risk, ("extreme fear: historically bullish (contrarian)" if s < 0 else "complacency: historically weaker returns (contrarian)")
    return risk, ""


def refresh(refit=False):
    px = prices()
    sig, raw = signals(px)
    model = None
    if not refit and os.path.exists(MODEL):
        model = json.load(open(MODEL, encoding="utf-8"))
        if datetime.now() - datetime.strptime(model["fitted"], "%Y-%m-%d %H:%M") > timedelta(days=7):
            model = None
    if model is None:
        model = fit(px, sig)
        os.makedirs(CDIR, exist_ok=True)
        json.dump(model, open(MODEL, "w", encoding="utf-8"), indent=1)
    comp = composite(sig, model["weights"])
    last = sig.index[-1]
    rows = []
    for k, label, group, how in IND:
        if k not in sig or pd.isna(sig[k].iloc[-1]):
            continue
        s = float(sig[k].iloc[-1])
        s21 = float(sig[k].iloc[-22]) if len(sig) > 22 and not pd.isna(sig[k].iloc[-22]) else None
        v = float(raw[k].iloc[-1])
        risk, note = _reading(k, s, model)
        m = model["indicators"].get(k, {})
        rows.append({"key": k, "label": label, "group": group, "how": how, "value": round(v, 4 if v < 10 else 2),
                     "signal": round(s, 2), "signal_1m_ago": None if s21 is None else round(s21, 2), "reading": risk, "note": note,
                     "used": m.get("used", False), "direction": m.get("direction"), "weight": model["weights"].get(k),
                     "ic_train": m.get("ic_train"), "ic_test": m.get("ic_test")})
    c = float(comp.iloc[-1])
    hist = comp.dropna()
    pct = float((hist.iloc[-2520:] < c).mean() * 100)
    alerts = divergences(px, sig)
    out = {"asof": f"{last:%Y-%m-%d}", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "composite": {"score": round(c, 1), "percentile_10y": round(pct), "score_1w_ago": round(float(comp.iloc[-6]), 1),
                         "score_1m_ago": round(float(comp.iloc[-22]), 1),
                         "label": "strong risk appetite" if c > 40 else "risk-on" if c > 10 else "risk-off" if c < -10 else "neutral" if c >= -10 else "",
                         "history": [[f"{d:%Y-%m-%d}", round(float(x), 1)] for d, x in comp.dropna().iloc[-260:].items()]},
           "indicators": rows, "alerts": alerts,
           "model": {k: model[k] for k in ("horizon_days", "train", "test", "composite", "fitted")}}
    if out["composite"]["score"] < -40:
        out["composite"]["label"] = "fear / risk-off extreme"
    os.makedirs(CDIR, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    return out


def divergences(px, sig):
    """Index near its high while the leaders / breadth are not confirming (or the opposite)."""
    spy = px["SPY"]
    near_high = spy.iloc[-1] >= spy.iloc[-63:].max() * 0.98
    near_low = spy.iloc[-1] <= spy.iloc[-63:].min() * 1.03
    out = []
    weak = [k for k in ("rsp_spy", "iwm_spy", "smh_spy", "iyt_spy", "hyg_ief", "breadth50") if k in sig and sig[k].iloc[-1] < -0.25]
    strong = [k for k in ("rsp_spy", "iwm_spy", "smh_spy", "iyt_spy", "hyg_ief", "breadth50") if k in sig and sig[k].iloc[-1] > 0.25]
    names = {k: lab for k, lab, *_ in IND}
    if near_high and len(weak) >= 2:
        out.append("SPY is within 2% of its 3-month high but these are weakening: " + ", ".join(names[k] for k in weak) +
                   " (narrow / unconfirmed rally)")
    if near_low and len(strong) >= 2:
        out.append("SPY is near its 3-month low but these are improving: " + ", ".join(names[k] for k in strong) +
                   " (positive divergence)")
    if "vix_term" in sig and sig["vix_term"].iloc[-1] < -0.9:
        out.append("VIX is above the 3-month VIX (backwardation): near-term panic, which has often come near short-term lows")
    return out


def summary(d):
    """Plain text for the brief's data output."""
    c = d["composite"]
    L = [f"SENTIMENT & LEADERSHIP (as of {d['asof']})",
         f"  risk appetite score {c['score']:+.0f} (-100..+100; 10-year percentile {c['percentile_10y']}; 1 week ago {c['score_1w_ago']:+.0f}, "
         f"1 month ago {c['score_1m_ago']:+.0f}) — {c['label']}"]
    for r in d["indicators"]:
        L.append(f"  {r['label']}: {r['value']} · signal {r['signal']:+.2f} (1m ago {r['signal_1m_ago']}) · {r['reading']}"
                 + (f" · {r['note']}" if r["note"] else "") + (f" · weight {r['weight']:+.2f} ({r['direction']})" if r["used"] else " · not used (no lead in the test)"))
    for a in d["alerts"]:
        L.append(f"  DIVERGENCE: {a}")
    m = d["model"]["composite"]
    L.append(f"  model (indicators kept only if they led SPY the same way in 2012-16 and 2017-21): rank correlation with SPY's next month {m['ic_train']:+.3f} (2012-2021), {m['ic_test']:+.3f} (2022+); "
             f"SPY next-month avg by score quintile since 2022: " + ", ".join(f"{k} {v:+.2f}%" for k, v in m["test_quintiles"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = refresh(refit="--refit" in sys.argv)
    print(summary(d))
    print(json.dumps(json.load(open(MODEL, encoding="utf-8"))["indicators"], indent=0)[:3000])
