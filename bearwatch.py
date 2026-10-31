"""
Bear market watch: are we heading into a bear market? Classic technical and credit warning signs
for the S&P 500, checked every day, plus a test of how the same signs behaved since 1997.

Warning signs (each on / off):
  trend      SPY below its 200-day average · the 200-day average falling · 50-day below 200-day (death cross)
             · SPY below a falling 40-week average (Weinstein stage 4) · 12-month return negative
  damage     drawdown from the 52-week high of 10% or more (correction)
  breadth    fewer than 40% of the sector ETFs above their 200-day average
  Dow theory Dow Jones Industrials and Transports both below their 200-day averages
  credit     high-yield spread above its 1-year average and up 1 point+ from its 1-year low
  curve      10-year minus 3-month Treasury inverted at some point in the last 18 months and now
             re-steepening (the classic recession lead)

State: Bear market (SPY 20%+ below its high and below a falling 200-day), otherwise by the
number of signs: 0-2 Bull, 3-4 Caution, 5+ Bear-market warning (which then stays on until 2 or fewer remain).

Test since 1997: what SPY did next (6-month return, worst drop) in each state, how early the
warning came before the 2000, 2007, 2020 and 2022 bear markets, and how often it was a false alarm.

    python bearwatch.py      # writes context/bear.json
"""

import os, sys, json
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "context", "bear.json")
CACHE = os.path.join(ROOT, "data", "bear_prices.csv")
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]   # the original nine, since 1998
BEARS = [("2000-03-24", "Dot-com bust"), ("2007-10-09", "Financial crisis"), ("2020-02-19", "COVID crash"), ("2022-01-03", "2022 rate shock")]
SIGNS = [("below200", "SPY below its 200-day average"), ("ma200_falling", "200-day average falling"),
         ("death_cross", "50-day below 200-day (death cross)"), ("stage4", "Below a falling 40-week average (stage 4)"),
         ("mom12", "12-month return negative"), ("dd10", "10%+ below the 52-week high"),
         ("breadth", "Under 40% of sectors above their 200-day"), ("dow", "Dow Industrials and Transports both below 200-day"),
         ("credit", "High-yield spread widening"), ("curve", "Yield curve re-steepening after an inversion")]


def _with_latest(df):
    """Yahoo's bulk download sometimes has no close yet for the latest day; add / fill that day
    from the single-ticker SPY history (which builds it from intraday bars)."""
    try:
        import patterns as P
        spy = P.load_prices("SPY", "1y", fill_today=True)["Close"]
        d = spy.index[-1]
        if d not in df.index:
            df.loc[d] = df.iloc[-1]
        df.loc[d, "SPY"] = float(spy.iloc[-1])
    except Exception:
        pass
    return df.sort_index()


def _prices():
    if os.path.exists(CACHE) and datetime.fromtimestamp(os.path.getmtime(CACHE)).date() == datetime.now().date():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    df = yf.download(["SPY", "^DJI", "^DJT"] + SECTORS, start="1995-01-01", auto_adjust=True, progress=False)["Close"]
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[df["SPY"].notna()].ffill(limit=5)
    df = _with_latest(df)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE)
    return df


def _fred(sid):
    try:
        import macro
        s = macro.fred(sid, start="1995-01-01")
        return s if s is not None and len(s) else None
    except Exception:
        return None


def signs(px):
    spy = px["SPY"]
    s50, s200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    w40 = spy.rolling(200).mean()                      # 40 weeks ≈ 200 trading days; slope over 4 weeks
    f = pd.DataFrame(index=px.index)
    f["below200"] = spy < s200
    f["ma200_falling"] = s200 < s200.shift(20)
    f["death_cross"] = s50 < s200
    f["stage4"] = (spy < w40) & (w40 < w40.shift(20))
    f["mom12"] = spy / spy.shift(252) - 1 < 0
    dd = spy / spy.rolling(252, min_periods=60).max() - 1
    f["dd10"] = dd <= -0.10
    sec = px[[s for s in SECTORS if s in px]]
    f["breadth"] = (sec > sec.rolling(200).mean()).where(sec.notna()).mean(axis=1) < 0.40
    if "^DJI" in px and "^DJT" in px:
        f["dow"] = (px["^DJI"] < px["^DJI"].rolling(200).mean()) & (px["^DJT"] < px["^DJT"].rolling(200).mean())
    hy = _fred("BAMLH0A0HYM2")
    if hy is not None:
        hy = hy.reindex(px.index).ffill()
        f["credit"] = (hy > hy.rolling(252).mean()) & (hy - hy.rolling(252).min() >= 1.0)
    curve = _fred("T10Y3M")
    if curve is not None:
        c = curve.reindex(px.index).ffill()
        inverted_recently = (c < 0).astype(int).rolling(378, min_periods=1).max() > 0
        f["curve"] = inverted_recently & (c > c.rolling(63).min() + 0.5) & (c.rolling(63).min() < 0.3)
    return f.fillna(False), dd


def states(f, dd, px):
    n = f.sum(axis=1)
    spy = px["SPY"]
    s200 = spy.rolling(200).mean()
    bear = (dd <= -0.20) & (spy < s200) & (s200 < s200.shift(20))
    # hysteresis: a warning starts at 5 signs and stays on until 2 or fewer remain (fewer on/off flips)
    warn, on = [], False
    for x in n.values:
        on = x >= 5 or (on and x > 2)
        warn.append(on)
    warn = pd.Series(warn, index=f.index)
    st = pd.Series(np.where(bear, "Bear market", np.where(warn, "Bear-market warning", np.where(n >= 3, "Caution", "Bull"))), index=f.index)
    return st, n


def test(px, st, dd):
    spy = px["SPY"]
    fwd6 = spy.shift(-126) / spy - 1
    # worst drop over the next 6 months from today's close
    low6 = spy[::-1].rolling(126, min_periods=20).min()[::-1].shift(-1)
    worst6 = low6 / spy - 1
    m = st.index >= "1997-01-01"
    by = {}
    for s_ in ["Bull", "Caution", "Bear-market warning", "Bear market"]:
        k = m & (st == s_) & fwd6.notna()
        if k.sum():
            by[s_] = {"days_pct": round(100 * float((m & (st == s_)).sum() / m.sum()), 1), "fwd6_avg": round(float(fwd6[k].mean()) * 100, 1),
                      "fwd6_up_pct": round(100 * float((fwd6[k] > 0).mean())), "worst6_avg": round(float(worst6[k].mean()) * 100, 1)}
    # each bear: when did the first warning come, and how far below the peak was SPY by then
    bears = []
    for peak, name in BEARS:
        p = pd.Timestamp(peak)
        if p < st.index[0]:
            continue
        after = st[(st.index >= p - pd.Timedelta(days=60)) & (st.index <= p + pd.Timedelta(days=400))]
        w = after[after.isin(["Bear-market warning", "Bear market"])]
        if len(w):
            d = w.index[0]
            bears.append({"name": name, "peak": peak, "first_warning": f"{d:%Y-%m-%d}", "days_after_peak": int((d - p).days),
                          "spy_vs_peak_then": round(float(spy[d] / spy[spy.index <= p].iloc[-1] - 1) * 100, 1),
                          "max_drawdown": round(float(spy[(spy.index >= p) & (spy.index <= p + pd.Timedelta(days=760))].min() / spy[spy.index <= p].iloc[-1] - 1) * 100, 1)})
    # warning episodes that were not followed by a further 10% drop within 6 months = false alarms
    w = (st == "Bear-market warning") & m
    starts = w & ~w.shift(1, fill_value=False)
    eps = []
    for d in st.index[starts]:
        later = spy[(spy.index > d) & (spy.index <= d + pd.Timedelta(days=183))]
        eps.append(bool(len(later) and later.min() / spy[d] - 1 <= -0.10))
    # strategy: hold SPY, step aside while the state is a warning or a bear market (trade next day)
    out = st.isin(["Bear-market warning", "Bear market"]).shift(1, fill_value=False)
    r = spy.pct_change().fillna(0)
    strat = (r * (~out)).loc["1997-01-01":]
    bh = r.loc["1997-01-01":]
    def stats(x):
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        return {"cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "max_dd": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
                "sharpe": round(float(x.mean() / x.std() * np.sqrt(252)), 2)}
    return {"by_state": by, "bears": bears, "warning_episodes": len(eps), "false_alarms": len(eps) - sum(eps),
            "step_aside": stats(strat), "buy_hold": stats(bh), "since": "1997"}


def refresh():
    px = _prices()
    f, dd = signs(px)
    st, n = states(f, dd, px)
    t = test(px, st, dd)
    last = f.index[-1]
    spy = px["SPY"]
    vals = {"below200": f"SPY {spy.iloc[-1]:.2f} vs 200-day {spy.rolling(200).mean().iloc[-1]:.2f}",
            "dd10": f"{dd.iloc[-1] * 100:.1f}% from the 52-week high",
            "mom12": f"12-month return {(spy.iloc[-1] / spy.iloc[-253] - 1) * 100:+.1f}%"}
    out = {"asof": f"{last:%Y-%m-%d}", "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "state": st.iloc[-1], "signs_on": int(n.iloc[-1]),
           "signs_total": len([k for k, _ in SIGNS if k in f]),
           "signs": [{"key": k, "label": lab, "on": bool(f[k].iloc[-1]), "detail": vals.get(k, "")} for k, lab in SIGNS if k in f],
           "drawdown_pct": round(float(dd.iloc[-1]) * 100, 1),
           "history": [[f"{d:%Y-%m-%d}", int(x)] for d, x in n.iloc[-520:].items()],
           "state_history": [[f"{d:%Y-%m-%d}", s_] for d, s_ in st.iloc[-520:].items()],
           "test": t}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    return out


def summary(d):
    L = [f"BEAR MARKET WATCH (as of {d['asof']}): {d['state']} — {d['signs_on']} of {d['signs_total']} warning signs on; SPY {d['drawdown_pct']}% from its 52-week high"]
    L.append("  " + "; ".join(f"{'ON ' if s['on'] else 'off'} {s['label']}" + (f" ({s['detail']})" if s["detail"] else "") for s in d["signs"]))
    t = d["test"]
    L.append("  since 1997, SPY's next 6 months by state: " + "; ".join(f"{k} {v['fwd6_avg']:+}% avg, worst drop {v['worst6_avg']}% ({v['days_pct']}% of days)" for k, v in t["by_state"].items()))
    L.append(f"  warnings: {t['warning_episodes']} episodes, {t['false_alarms']} false alarms (no further 10% drop within 6 months); "
             f"stepping aside on warnings: CAGR {t['step_aside']['cagr']}% / max drawdown {t['step_aside']['max_dd']}% vs buy & hold {t['buy_hold']['cagr']}% / {t['buy_hold']['max_dd']}%")
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = refresh()
    print(summary(d))
    for b in d["test"]["bears"]:
        print("  ", b)
