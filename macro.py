"""
Macro dashboard and outlook trigger.

Data (FRED, St. Louis Fed): inflation, labor, growth, Fed & rates, financial
conditions, recession signals and oil — latest print, previous print, change,
a 12-month table, 10 years of monthly history, a trend label and where the
latest value sits in its 10-year range. Fed funds futures (Yahoo, ZQ contracts)
give a FedWatch-style market-implied rate path and hike / hold / cut odds for
the next FOMC meetings.

Outlook trigger: when a headline release (CPI, core CPI, PPI, PCE, core PCE,
payrolls, unemployment, JOLTS, GDP, retail sales) has a new observation, or an
FOMC decision / minutes date has passed, macro/state.json is marked stale with
the reasons. The market-brief routine then writes a new outlook
(macro/outlook-<id>.md) and marks it updated.

FRED access: with an API key (env FRED_API_KEY or a one-line file .fred_key in
the repo root, git-ignored) the official API is used; otherwise the public CSV
download, slowly, with retries. Every series is cached in data/fred/ for 6 hours.

Usage:
    python macro.py                 # refresh data, print the trigger state and a summary
    python macro.py --mark-updated <outlook_id>
"""

import os, sys, io, json, time, math, glob, argparse, urllib.request, urllib.parse
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MACRO_DIR = os.path.join(ROOT, "macro")
CACHE_DIR = "/tmp/data/fred" if os.environ.get("VERCEL") else os.path.join(ROOT, "data", "fred")
ET = ZoneInfo("America/New_York")
CACHE_HOURS = 6
HIST_YEARS = 10

# id, name, pillar, kind, tier1 (headline release), extra
#   kind: "pct"      index -> year-on-year % (also m/m % and 3-month annualized)
#         "payrolls" level -> monthly change (thousands)
#         "level"    the value itself (yoy=True also shows the year-on-year %)
SERIES = [
    ("CPIAUCSL", "CPI (headline)", "Inflation", "pct", True, {}),
    ("CPILFESL", "Core CPI", "Inflation", "pct", True, {}),
    ("PPIFIS", "PPI (final demand)", "Inflation", "pct", True, {}),
    ("PCEPI", "PCE prices", "Inflation", "pct", True, {}),
    ("PCEPILFE", "Core PCE (Fed target)", "Inflation", "pct", True, {}),
    ("T5YIE", "5-year breakeven inflation", "Inflation", "level", False, {"unit": "%"}),
    ("MICH", "UMich 1-year inflation expectations", "Inflation", "level", False, {"unit": "%"}),
    ("PAYEMS", "Nonfarm payrolls (monthly change)", "Labor", "payrolls", True, {"unit": "k"}),
    ("UNRATE", "Unemployment rate", "Labor", "level", True, {"unit": "%"}),
    ("ICSA", "Initial jobless claims (weekly)", "Labor", "level", False, {"unit": "", "scale": 1e-3, "suffix": "k"}),
    ("JTSJOL", "Job openings (JOLTS)", "Labor", "level", True, {"unit": "", "scale": 1e-3, "suffix": "M"}),
    ("CES0500000003", "Average hourly earnings", "Labor", "pct", False, {}),
    ("A191RL1Q225SBEA", "Real GDP (q/q annualized)", "Growth", "level", True, {"unit": "%"}),
    ("RSAFS", "Retail sales", "Growth", "pct", True, {}),
    ("INDPRO", "Industrial production", "Growth", "pct", False, {}),
    ("HOUST", "Housing starts (annualized)", "Growth", "level", False, {"unit": "", "scale": 1e-3, "suffix": "M"}),
    ("UMCSENT", "UMich consumer sentiment", "Growth", "level", False, {"unit": ""}),
    ("DFEDTARU", "Fed funds target (upper)", "Fed & rates", "level", False, {"unit": "%"}),
    ("DGS2", "2-year Treasury yield", "Fed & rates", "level", False, {"unit": "%"}),
    ("DGS10", "10-year Treasury yield", "Fed & rates", "level", False, {"unit": "%"}),
    ("T10Y2Y", "Curve: 10Y minus 2Y", "Fed & rates", "level", False, {"unit": "pp"}),
    ("T10Y3M", "Curve: 10Y minus 3M", "Fed & rates", "level", False, {"unit": "pp"}),
    ("DFII10", "10-year real yield (TIPS)", "Fed & rates", "level", False, {"unit": "%"}),
    ("BAMLH0A0HYM2", "High-yield credit spread", "Financial conditions", "level", False, {"unit": "pp"}),
    ("NFCI", "Chicago Fed financial conditions (<0 = loose)", "Financial conditions", "level", False, {"unit": ""}),
    ("DTWEXBGS", "Broad dollar index", "Financial conditions", "level", False, {"unit": "", "yoy": True}),
    ("M2SL", "M2 money supply", "Financial conditions", "pct", False, {}),
    ("WALCL", "Fed balance sheet", "Financial conditions", "level", False, {"unit": "", "scale": 1e-6, "suffix": "T", "yoy": True}),
    ("SAHMREALTIME", "Sahm rule (>= 0.5 = recession signal)", "Recession signals", "level", False, {"unit": "pp"}),
    ("DCOILWTICO", "WTI crude oil (spot)", "Oil", "level", False, {"unit": "$", "yoy": True}),
    ("DCOILBRENTEU", "Brent crude oil (spot, EIA)", "Oil", "level", False, {"unit": "$", "yoy": True}),
    ("OVXCLS", "Oil volatility index (OVX)", "Oil", "level", False, {"unit": ""}),
]
PILLARS = ["Inflation", "Labor", "Growth", "Fed & rates", "Financial conditions", "Recession signals", "Oil"]
EXTRA_IDS = ["DFF", "DFEDTARL"]          # used for the Fed-futures calculation only

# FOMC decision days (second day of each meeting); minutes come out 3 weeks later.
FOMC = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
MONTH_CODES = "FGHJKMNQUVXZ"


# ── FRED ──────────────────────────────────────────────────────────────────────
def _key():
    k = os.environ.get("FRED_API_KEY")
    if not k and os.path.exists(os.path.join(ROOT, ".fred_key")):
        k = open(os.path.join(ROOT, ".fred_key"), encoding="utf-8").read().strip()
    return k or None


def _get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")


_public_failures = 0     # circuit breaker for the keyless download page


def fred(sid, start="2013-01-01"):
    """A FRED series as a float Series indexed by date (cached CACHE_HOURS).
    Without an API key the public page is tried once with a short timeout, and after
    two failures in a run FRED is skipped (cached data only) so a brief never stalls."""
    global _public_failures
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{sid}.csv")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < CACHE_HOURS * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    key, err = _key(), None
    if not key and _public_failures >= 2:
        err = "skipped: FRED download page is blocking requests (add a FRED API key)"
    for attempt in range(3 if key else (0 if err else 1)):
        try:
            if key:
                q = urllib.parse.urlencode({"series_id": sid, "api_key": key, "file_type": "json", "observation_start": start})
                obs = json.loads(_get("https://api.stlouisfed.org/fred/series/observations?" + q))["observations"]
                s = pd.Series({pd.Timestamp(o["date"]): (float(o["value"]) if o["value"] not in (".", "") else np.nan) for o in obs})
            else:
                txt = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}", timeout=12)
                if not txt.lower().startswith(("observation_date", "date")):
                    raise ValueError("FRED returned a non-CSV page (rate limit / bot check)")
                df = pd.read_csv(io.StringIO(txt))
                s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(df.iloc[:, 0]))
            s = s.dropna()
            s.name = sid
            s.to_frame().to_csv(path)
            if not key:
                time.sleep(1.5)            # be gentle with the public download page
            return s
        except Exception as e:
            err = e
            if not key:
                _public_failures += 1
                break
            time.sleep(4 * (attempt + 1))
    if os.path.exists(path):              # stale cache is better than nothing
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    raise RuntimeError(f"{sid}: {err}")


# ── Transformations ───────────────────────────────────────────────────────────
def _monthly(s):
    """One value per calendar month; months with no observation stay NaN so that
    shifts are calendar-correct (e.g. the missing Oct-2025 CPI must not turn a
    12-month comparison into a 13-month one)."""
    return s.resample("ME").last()


def _primary(s, kind):
    m = _monthly(s)
    if kind == "pct":
        return (m / m.shift(12) - 1) * 100
    if kind == "payrolls":
        return m.diff()
    return m


def _trend(p, kind, pillar):
    """Rising / falling / stable from the last 3 months vs the 3 before, scaled by the
    series' own variability."""
    p = p.dropna()
    if len(p) < 12:
        return "n/a"
    a, b = p.iloc[-3:].mean(), p.iloc[-6:-3].mean()
    thr = 0.35 * p.iloc[-60:].diff().abs().median() * 3 if len(p) > 13 else 0
    if abs(a - b) <= thr:
        return "stable"
    up = a > b
    if pillar == "Inflation":
        return "accelerating" if up else "cooling"
    return "rising" if up else "falling"


def series_view(sid, name, pillar, kind, tier1, extra):
    raw = fred(sid)
    p = _primary(raw, kind)
    scale, suffix, unit = extra.get("scale", 1), extra.get("suffix", ""), extra.get("unit", "%" if kind == "pct" else "")
    pv = p.dropna()
    if pv.empty:
        raise ValueError("no data")
    cutoff = pv.index[-1] - pd.DateOffset(years=HIST_YEARS)
    hist = pv[pv.index > cutoff]
    out = {
        "id": sid, "name": name, "pillar": pillar, "kind": kind, "tier1": tier1,
        "unit": "% y/y" if kind == "pct" else ("k jobs" if kind == "payrolls" else (unit + suffix if unit != "$" else "$")),
        "latest_obs": str(raw.index[-1].date()), "latest_raw": round(float(raw.iloc[-1]) * scale, 3),
        "value": round(float(pv.iloc[-1]) * (scale if kind == "level" else 1), 3),
        "value_date": str(pv.index[-1].date()),
        "prev": round(float(pv.iloc[-2]) * (scale if kind == "level" else 1), 3) if len(pv) > 1 else None,
        "trend": _trend(pv, kind, pillar),
        "pct_rank_10y": round(float((hist < pv.iloc[-1]).mean() * 100), 0),
        "range_10y": [round(float(hist.min()) * (scale if kind == "level" else 1), 2), round(float(hist.max()) * (scale if kind == "level" else 1), 2)],
        "table12": [{"date": str(d.date())[:7], "value": round(float(v) * (scale if kind == "level" else 1), 3)} for d, v in pv.iloc[-12:].items()],
        "hist": {"x": [str(d.date())[:7] for d in hist.index], "y": [round(float(v) * (scale if kind == "level" else 1), 3) for v in hist.values]},
    }
    if kind == "pct":
        m = _monthly(raw)
        mom, a3 = m / m.shift(1) - 1, (m / m.shift(3)) ** 4 - 1
        out["mom"] = None if np.isnan(mom.iloc[-1]) else round(float(mom.iloc[-1] * 100), 2)
        out["ann3m"] = None if np.isnan(a3.iloc[-1]) else round(float(a3.iloc[-1] * 100), 2)
    if kind == "payrolls":
        out["avg3m"] = round(float(pv.iloc[-3:].mean()), 0)
    if extra.get("yoy"):
        m = _monthly(raw)
        y = m / m.shift(12) - 1
        out["yoy"] = None if np.isnan(y.iloc[-1]) else round(float(y.iloc[-1] * 100), 1)
    if raw.index.inferred_freq is None or (raw.index[-1] - raw.index[-2]).days <= 7:
        out["daily_or_weekly"] = True
    return out


# ── Regimes (deterministic labels per pillar) ─────────────────────────────────
def regimes(v):
    g = lambda sid, k="value": (v.get(sid) or {}).get(k)
    out = {}
    core, core3 = g("CPILFESL"), g("CPILFESL", "ann3m")
    if core is not None and core3 is not None:
        heat = "hot" if core >= 3 else "elevated" if core >= 2.5 else "near target"
        dirn = "re-accelerating" if core3 > core + 0.3 else "cooling" if core3 < core - 0.3 else "steady"
        out["Inflation"] = f"{heat}, {dirn} (core CPI {core:.1f}% y/y, {core3:.1f}% 3-month annualized)"
    ur, pay3 = g("UNRATE"), g("PAYEMS", "avg3m")
    sahm = g("SAHMREALTIME")
    if ur is not None and pay3 is not None:
        lab = "weak" if (sahm or 0) >= 0.5 or pay3 < 0 else "softening" if pay3 < 75 or g("UNRATE", "trend") == "rising" else "solid"
        out["Labor"] = f"{lab} (unemployment {ur:.1f}%, payrolls 3-month avg {pay3:+.0f}k)"
    gdp, rs = g("A191RL1Q225SBEA"), g("RSAFS")
    if gdp is not None:
        gr = "contracting" if gdp < 0 else "slow" if gdp < 1.5 else "solid"
        out["Growth"] = f"{gr} (real GDP {gdp:.1f}% annualized" + (f", retail sales {rs:+.1f}% y/y)" if rs is not None else ")")
    ff, cpce = g("DFEDTARU"), g("PCEPILFE")
    if ff is not None and cpce is not None:
        real = ff - cpce
        st = "restrictive" if real > 1.5 else "mildly restrictive" if real > 0.5 else "neutral to easy"
        out["Fed & rates"] = f"{st} (real policy rate ~{real:+.1f}pp = fed funds {ff:.2f}% minus core PCE {cpce:.1f}%)"
    nfci, hy = g("NFCI"), g("BAMLH0A0HYM2")
    if nfci is not None:
        out["Financial conditions"] = ("loose" if nfci < -0.3 else "neutral" if nfci < 0.2 else "tight") + \
            f" (NFCI {nfci:+.2f}" + (f", high-yield spread {hy:.2f}pp)" if hy is not None else ")")
    c10y2, c10y3 = g("T10Y2Y"), g("T10Y3M")
    if sahm is not None:
        sig = []
        if sahm >= 0.5: sig.append("Sahm rule triggered")
        if c10y3 is not None and c10y3 < 0: sig.append("10Y-3M curve inverted")
        if g("ICSA", "trend") == "rising": sig.append("jobless claims trending up")
        out["Recession signals"] = ("; ".join(sig) if sig else "none flashing") + f" (Sahm {sahm:.2f}, 10Y-2Y {c10y2:+.2f}pp)"
    oil, ovx = g("DCOILWTICO", "latest_raw"), g("OVXCLS", "latest_raw")
    if oil is not None:
        out["Oil"] = f"WTI ${oil:.0f} ({g('DCOILWTICO', 'yoy'):+.0f}% y/y), price trend {g('DCOILWTICO', 'trend')}" + \
            (f", oil volatility {ovx:.0f} ({g('OVXCLS', 'pct_rank_10y'):.0f}th percentile)" if ovx is not None else "")
    return out


# ── Fed funds futures (FedWatch-style) ─────────────────────────────────────────
def fed_futures(effr, today=None):
    """Market-implied average fed funds rate per month (100 - ZQ price) and hike / hold /
    cut odds for the next FOMC meetings in the FOMC list, CME FedWatch method."""
    import yfinance as yf
    today = today or date.today()
    months = []
    y, m = today.year, today.month
    for k in range(9):
        mm, yy = (m - 1 + k) % 12 + 1, y + (m - 1 + k) // 12
        sym = f"ZQ{MONTH_CODES[mm - 1]}{str(yy)[2:]}.CBT"
        try:
            h = yf.Ticker(sym).history(period="5d")["Close"].dropna()
            if not h.empty:
                months.append({"month": f"{yy}-{mm:02d}", "symbol": sym, "implied": round(100 - float(h.iloc[-1]), 3)})
        except Exception:
            pass
    rate = {x["month"]: x["implied"] for x in months}
    meetings, pre = [], effr
    for d in FOMC:
        md = date.fromisoformat(d)
        if md <= today:
            continue
        key, nxt = f"{md.year}-{md.month:02d}", (md.replace(day=1) + timedelta(days=32)).replace(day=1)
        nkey = f"{nxt.year}-{nxt.month:02d}"
        has_next_meeting = any(x[:7] == nkey for x in FOMC)
        if nkey in rate and not has_next_meeting:
            post = rate[nkey]                               # clean month after the meeting
        elif key in rate:
            D = (nxt - md.replace(day=1)).days
            n1 = md.day - 1                                 # days at the old rate (decision effective next day)
            post = (rate[key] * D - pre * n1) / (D - n1)
        else:
            break
        ch = post - pre
        p = {"hike": 0.0, "hold": 0.0, "cut": 0.0}
        steps = ch / 0.25
        if steps >= 0:
            p["hike"], p["hold"] = min(steps, 1.0), max(0.0, 1 - steps)
        else:
            p["cut"], p["hold"] = min(-steps, 1.0), max(0.0, 1 + steps)
        meetings.append({"date": d, "rate_before": round(pre, 3), "implied_after": round(post, 3),
                         "change_bp": round(ch * 100, 1), "p_hike": round(p["hike"] * 100), "p_hold": round(p["hold"] * 100),
                         "p_cut": round(p["cut"] * 100), "big_move": abs(steps) > 1})
        pre = post
    return {"effr": round(effr, 3), "months": months, "meetings": meetings,
            "note": "Implied from 30-day fed funds futures (Yahoo, ZQ contracts) with the CME FedWatch method; "
                    "approximate, assumes 25 bp steps."}


# ── Refresh, trigger state, outlook index ─────────────────────────────────────
def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def refresh():
    os.makedirs(MACRO_DIR, exist_ok=True)
    views, errors = {}, {}
    for sid, name, pillar, kind, tier1, extra in SERIES:
        try:
            views[sid] = series_view(sid, name, pillar, kind, tier1, extra)
        except Exception as e:
            errors[sid] = str(e)
    ff = None
    try:
        effr = float(fred("DFF").iloc[-1])
        ff = fed_futures(effr)
        lo = float(fred("DFEDTARL").iloc[-1])
        ff["target_range"] = [lo, float(views["DFEDTARU"]["latest_raw"]) if "DFEDTARU" in views else None]
    except Exception as e:
        errors["fed_futures"] = str(e)
    now = datetime.now(ET)
    data = {"generated_et": now.strftime("%Y-%m-%d %H:%M"), "pillars": PILLARS,
            "series": [views[s[0]] for s in SERIES if s[0] in views], "regimes": regimes(views),
            "fed_futures": ff, "errors": errors}
    json.dump(data, open(os.path.join(MACRO_DIR, "data.json"), "w", encoding="utf-8"), indent=1, default=float)

    # trigger: new headline prints or a passed FOMC decision / minutes date
    st_path = os.path.join(MACRO_DIR, "state.json")
    st = _load(st_path, None)
    first = st is None
    st = st or {"seen": {}, "fomc_seen": [], "stale": False, "reasons": [], "last_outlook": None}
    reasons = []
    for sid, v in views.items():
        if not v["tier1"]:
            continue
        old = st["seen"].get(sid)
        if old and v["latest_obs"] > old:
            reasons.append(f"new {v['name']} print (for {v['latest_obs'][:7]})")
        st["seen"][sid] = v["latest_obs"]
    for d in FOMC:
        for kind, when in (("decision", date.fromisoformat(d)), ("minutes", date.fromisoformat(d) + timedelta(days=21))):
            tag = f"{kind} {d}"
            passed = now.date() > when or (now.date() == when and now.hour >= (15 if kind == "decision" else 14))
            if passed and tag not in st["fomc_seen"]:
                if not first and now.date() - when <= timedelta(days=3):
                    reasons.append(f"FOMC {kind} ({when})")
                st["fomc_seen"].append(tag)
    if first:
        reasons.append("first macro outlook")
    if reasons:
        st["stale"] = True
        st["reasons"] = sorted(set(st.get("reasons", []) + reasons))
    st["checked_et"] = data["generated_et"]
    json.dump(st, open(st_path, "w", encoding="utf-8"), indent=1)
    return data, st


def mark_updated(outlook_id):
    st_path = os.path.join(MACRO_DIR, "state.json")
    st = _load(st_path, {"seen": {}, "fomc_seen": []})
    st.update({"stale": False, "reasons": [], "last_outlook": outlook_id})
    json.dump(st, open(st_path, "w", encoding="utf-8"), indent=1)
    rebuild_index()


def rebuild_index():
    items = []
    for f in sorted(glob.glob(os.path.join(MACRO_DIR, "outlook-*.md")), reverse=True):
        oid = os.path.basename(f)[8:-3]
        title = next((l.lstrip("# ").strip() for l in open(f, encoding="utf-8") if l.startswith("# ")), oid)
        items.append({"id": oid, "file": os.path.basename(f), "title": title})
    json.dump({"outlooks": items}, open(os.path.join(MACRO_DIR, "index.json"), "w", encoding="utf-8"), indent=1)


def summary(data, st):
    L = [f"MACRO_OUTLOOK_STALE={'yes' if st.get('stale') else 'no'}"]
    if st.get("stale"):
        L.append("MACRO_REASONS=" + "; ".join(st.get("reasons", [])))
        L.append(f"WRITE_OUTLOOK_TO=macro/outlook-{datetime.now(ET):%Y-%m-%d-%H%M}.md")
        prev = sorted(glob.glob(os.path.join(MACRO_DIR, "outlook-*.md")))
        if prev:
            L.append("PREVIOUS_OUTLOOK=macro/" + os.path.basename(prev[-1]))
    L.append("\nMACRO REGIMES")
    for k, x in data["regimes"].items():
        L.append(f"  {k}: {x}")
    L.append("\nMACRO INDICATORS (latest | previous | trend | 10y percentile | 12 months)")
    for p in PILLARS:
        for v in [s for s in data["series"] if s["pillar"] == p]:
            extra = ""
            if v.get("mom") is not None: extra += f", m/m {v['mom']:+.2f}%" + (f", 3m ann {v['ann3m']:.1f}%" if v.get("ann3m") is not None else "")
            if "avg3m" in v: extra += f", 3m avg {v['avg3m']:+.0f}k"
            if v.get("yoy") is not None: extra += f", {v['yoy']:+.1f}% y/y"
            L.append(f"  [{p}] {v['name']}: {v['value']} {v['unit']} ({v['value_date'][:7]}, obs {v['latest_obs']}) | prev {v['prev']}"
                     f" | {v['trend']} | {v['pct_rank_10y']:.0f}th pct (10y {v['range_10y'][0]}–{v['range_10y'][1]}){extra}"
                     f" | 12m: " + ", ".join(f"{t['date']} {t['value']}" for t in v["table12"]))
    ff = data.get("fed_futures")
    if ff:
        L.append(f"\nFED FUNDS FUTURES (effective rate {ff['effr']}%)")
        for m in ff["meetings"]:
            L.append(f"  FOMC {m['date']}: implied {m['implied_after']}% ({m['change_bp']:+.0f} bp) → hike {m['p_hike']}% / hold {m['p_hold']}% / cut {m['p_cut']}%")
        L.append("  implied average rate by month: " + ", ".join(f"{x['month']} {x['implied']}%" for x in ff["months"]))
    if data["errors"]:
        L.append("\nMACRO DATA ERRORS: " + "; ".join(f"{k}: {v}" for k, v in data["errors"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mark-updated", metavar="OUTLOOK_ID")
    a = ap.parse_args()
    if a.mark_updated:
        mark_updated(a.mark_updated)
        print(f"outlook {a.mark_updated} marked as current")
    else:
        d, s = refresh()
        print(summary(d, s))
