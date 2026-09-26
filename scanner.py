"""
Daily Pivot Scanner — QQQ / SPY / MAG7 / $100B+ tech
Computes entry/exit signals and shows a 90-day signal timeline overlaid on the price chart.

Usage:
    python scanner.py
    python scanner.py --no-browser
"""

import sys, os, json, math, webbrowser
from datetime import datetime

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("Missing dependencies. Run:  pip install yfinance pandas")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOLS = [
    # Broad market
    "QQQ", "SPY",
    # Sector ETFs — tech & semis
    "SOXX", "IGV", "DRAM", "SMH", "SOXL", "CHAT", "ARKG", "NVDL",
    # Sector ETFs — other
    "COPX", "UNG", "HYG", "SPCX",
    # Mega-cap tech (MAG7)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Large-cap tech
    "AVGO", "ORCL", "NFLX", "AMD", "CSCO", "QCOM", "TXN", "CRM", "ASML",
    # Memory & chip design
    "MU", "ARM", "MRVL", "WDC", "RMBS",
    # AI / data / cloud
    "PLTR", "SNOW", "NET", "RDDT", "TEM", "NBIS",
    # Software
    "ADBE", "NOW",
    # Cybersecurity
    "PANW",
    # Platform / ride-share
    "UBER", "GRAB",
    # Fintech / crypto-adjacent
    "CRCL",
    # Clean energy / nuclear
    "VST", "BE", "SMR",
    # Space & quantum
    "RKLB", "ASTS", "LUNR", "PL", "IONQ",
    # Healthcare
    "ISRG",
    # Transport
    "AAL",
    # Consumer / other
    "MCD", "GME",
    # Speculative semis
    "AAOI", "AEHR",
    # Cybersecurity
    "CRWD",
    # Observability / DevOps
    "DDOG",
    # AI networking
    "ANET",
    # AI servers
    "SMCI",
    # Broad market risk signal
    "IWM",
    # E-commerce
    "SHOP",
    # Semis / hardware additions
    "AMAT", "ON", "TSM", "NXPI", "DELL", "INTC", "BABA", "IREN", "AMBA",
    # Software additions
    "IBM", "DT",
    # Sector ETFs (comprehensive coverage)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XBI",
    # Commodities ETFs
    "GLD", "SLV",
    # Market / thematic ETFs
    "MAGS", "QTUM", "SQQQ",
    # Quantum computing
    "QBTS", "RGTI",
    # Healthcare / defense / energy
    "UNH", "HWM", "OKLO",
    # International / Korea
    "EWY",
    # Fintech / gaming / crypto
    "COIN", "HOOD", "SOFI", "TTWO",
    # Sector ETFs — additional
    "XLRE", "XLU", "XLY",
    # Consumer
    "COST", "LULU", "NKE",
    # Pharma / biotech
    "LLY", "ABBV", "PFE", "GEHC",
    # Defense
    "LMT", "ITA",
    # Energy / clean energy
    "CEG", "FSLR", "NEE",
    # AI infrastructure / optical
    "VRT", "CRDO", "ALAB", "COHR", "CRWV",
    # Bitcoin miners
    "MSTR", "MARA", "CORZ",
    # SaaS / collaboration
    "TEAM", "MNDY",
    # China ADRs
    "JD", "PDD",
    # Other
    "GLW",
]
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
OPEN_BROWSER = "--no-browser" not in sys.argv
HISTORY_DAYS = 90
RSI_LOWER    = 40
SIGNAL_MIN   = 60

# ── Indicators ─────────────────────────────────────────────────────────────────
def _rsi(close, period=14):
    delta = close.diff()
    g = delta.clip(lower=0)
    l = -delta.clip(upper=0)
    ag = g.ewm(com=period-1, min_periods=period).mean()
    al = l.ewm(com=period-1, min_periods=period).mean()
    rs = ag / al.replace(0, float('nan'))
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, sig=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

def _stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    rng = (hi - lo).replace(0, float('nan'))
    raw = 100 * (close - lo) / rng
    K = raw.rolling(d).mean()
    D = K.rolling(d).mean()
    return K, D

def _obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()

# ── Day extraction ─────────────────────────────────────────────────────────────
def _day(df, rsi_s, ml_s, sl_s, hist_s, sk_s, sd_s, obv_s, vol_ma_s,
         ma20_s, ma50_s, ma200_s, idx):
    n = len(df)
    def v(s, i=idx):
        pos = n + i if i < 0 else i
        if pos < 0 or pos >= n: return None
        try:
            f = float(s.iloc[i])
            return None if math.isnan(f) else f
        except: return None
    return {
        "close":          v(df["close"]),
        "ma20":           v(ma20_s),
        "ma50":           v(ma50_s),
        "ma200":          v(ma200_s),
        "rsi":            v(rsi_s),
        "rsi_prev":       v(rsi_s,    idx-1),
        "macd_hist":      v(hist_s),
        "macd_hist_prev": v(hist_s,   idx-1),
        "macd_line":      v(ml_s),
        "sig_line":       v(sl_s),
        "stoch_k":        v(sk_s),
        "stoch_d":        v(sd_s),
        "stoch_k_prev":   v(sk_s,     idx-1),
        "stoch_d_prev":   v(sd_s,     idx-1),
        "obv":            v(obv_s),
        "obv_prev":       v(obv_s,    idx-1),
        "volume":         v(df["volume"]),
        "vol_ma20":       v(vol_ma_s),
    }

# ── Entry scoring ──────────────────────────────────────────────────────────────
def _score_entry(d):
    c, ma50, ma200 = d.get("close"), d.get("ma50"), d.get("ma200")
    rsi, hist, hist_prev = d.get("rsi"), d.get("macd_hist"), d.get("macd_hist_prev")
    sk, sd = d.get("stoch_k"), d.get("stoch_d")
    sk_p, sd_p = d.get("stoch_k_prev"), d.get("stoch_d_prev")
    vol, vol_ma = d.get("volume"), d.get("vol_ma20")
    obv, obv_p = d.get("obv"), d.get("obv_prev")

    r1 = bool(rsi is not None and rsi < RSI_LOWER)
    r2 = bool(hist is not None and hist < 0 and hist_prev is not None and hist > hist_prev)
    r3 = bool(sk and sd and sk_p is not None and sd_p is not None
              and sk > sd and sk_p <= sd_p and sk < 50)
    r4 = bool(vol and vol_ma and vol > vol_ma)
    r5 = bool(c and ma50 and c <= ma50 * 1.01)
    r6 = bool(obv is not None and obv_p is not None and obv > obv_p)

    rules = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R6": r6}
    score = sum(20 for v in rules.values() if v)
    sig = "BUY" if score >= SIGNAL_MIN else "WATCH" if score >= 40 else "HOLD"

    macd_imp = hist is not None and hist_prev is not None and hist > hist_prev
    vol_ratio = (vol/vol_ma) if (vol and vol_ma and vol_ma>0) else None
    pct_ma50  = ((c/ma50-1)*100) if (c and ma50) else None

    rule_details = {
        "R1": {"passed": r1, "ok": f"RSI {_f(rsi)} &lt; {RSI_LOWER}", "fail": f"RSI {_f(rsi)}"},
        "R2": {"passed": r2, "ok": f"MACD hist {_f(hist,2)} &lt; 0 and rising",
               "fail": f"MACD hist {_f(hist,2)} ({'rising' if macd_imp else 'falling'})"},
        "R3": {"passed": r3, "ok": f"Stoch K {_f(sk)} crossed above D {_f(sd)} (&lt;50)",
               "fail": f"Stoch K={_f(sk)} D={_f(sd)}, no cross"},
        "R4": {"passed": r4, "ok": f"Volume {_fv(vol)} > avg {_fv(vol_ma)} ({_f(vol_ratio)}×)",
               "fail": f"Volume {_fv(vol)} < avg {_fv(vol_ma)}"},
        "R5": {"passed": r5, "ok": "Price ≤ MA50 × 1.01", "fail": f"Price {_f1(pct_ma50)}% above MA50"},
        "R6": {"passed": r6, "ok": "OBV rising", "fail": "OBV falling"},
    }
    return {"entry_score": score, "entry_signal": sig, "entry_rules": rule_details}

# ── Exit scoring ───────────────────────────────────────────────────────────────
def _score_exit(d):
    rsi, rsi_p = d.get("rsi"), d.get("rsi_prev")
    hist, hist_p = d.get("macd_hist"), d.get("macd_hist_prev")
    sk, sd, sk_p, sd_p = d.get("stoch_k"), d.get("stoch_d"), d.get("stoch_k_prev"), d.get("stoch_d_prev")
    obv, obv_p = d.get("obv"), d.get("obv_prev")
    c, ma200, ma20 = d.get("close"), d.get("ma200"), d.get("ma20")
    vol, vol_ma = d.get("volume"), d.get("vol_ma20")
    macd_l, sig_l = d.get("macd_line"), d.get("sig_line")

    e1 = bool(rsi and rsi_p and rsi > 70 and rsi < rsi_p)
    e2 = bool(hist and hist_p and hist > 0 and hist < hist_p)
    e3 = bool(sk and sd and sk_p is not None and sd_p is not None
              and sk < sd and sk_p >= sd_p and sk_p >= 80)
    e4 = bool(macd_l and sig_l and hist and hist_p and macd_l < sig_l and hist < hist_p)
    e5 = bool(c and ma20 and vol and vol_ma and c < ma20 and vol > vol_ma)
    e6 = bool(obv is not None and obv_p is not None and obv < obv_p)
    e7 = bool(c and ma200 and c < ma200)

    exits = {"E1":("independent",e1),"E2":("independent",e2),"E3":("independent",e3),
             "E4":("pair",e4),"E5":("pair",e5),"E6":("independent",e6),"E7":("independent",e7)}

    ind_fired  = [k for k,(kind,p) in exits.items() if p and kind=="independent"]
    pair_fired = [k for k,(kind,p) in exits.items() if p and kind=="pair"]
    signal     = len(ind_fired) + (1 if pair_fired else 0) >= 2 and len(ind_fired) >= 1

    ok_m  = {"E1":f"RSI {_f(rsi)}>70 falling","E2":f"MACD hist {_f(hist,2)}>0 falling",
              "E3":f"Stoch bear cross","E4":"MACD bearish+worsening","E5":"Price<MA20 on vol",
              "E6":"OBV falling","E7":f"Price<MA200"}
    fail_m = {"E1":f"RSI {_f(rsi)}","E2":f"MACD hist {_f(hist,2)}","E3":f"K={_f(sk)} D={_f(sd)}",
               "E4":"MACD ok","E5":"Price>MA20","E6":"OBV rising","E7":"Price>MA200"}
    rule_details = {k:{"passed":p,"kind":kind,"ok":ok_m[k],"fail":fail_m[k]}
                    for k,(kind,p) in exits.items()}
    return {"exit_signal":signal,"exit_rules":rule_details,
            "exit_ind_fired":ind_fired,"exit_pair_fired":pair_fired}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_mcap(v):
    if v is None: return "n/a"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.0f}B"
    return f"${v/1e6:.0f}M"

def _fmt_pct(v):
    if v is None: return "n/a"
    return f"{v*100:+.1f}%"

ANALYST_LABELS = {
    "strong_buy":  ("⬆ Strong Buy", "green"),
    "buy":         ("↑ Buy",         "green"),
    "hold":        ("— Hold",        "muted"),
    "sell":        ("↓ Sell",        "red"),
    "strong_sell": ("⬇ Strong Sell", "red"),
}

def _f(v, dec=1):
    if v is None: return "n/a"
    return f"{v:.{dec}f}"
def _f1(v):
    if v is None: return "n/a"
    return f"{v:+.1f}"
def _fv(v):
    if v is None: return "n/a"
    if v>=1e9: return f"{v/1e9:.1f}B"
    if v>=1e6: return f"{v/1e6:.1f}M"
    return f"{v:.0f}"

# ── Symbol analysis ────────────────────────────────────────────────────────────
def analyze_symbol(ticker):
    try:
        raw = yf.download(ticker, period="2y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 60:
            return {"symbol": ticker, "error": "insufficient data"}

        df = raw.copy()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        rsi_s          = _rsi(close)
        ml_s, sl_s, hist_s = _macd(close)
        sk_s, sd_s     = _stoch(high, low, close)
        obv_s          = _obv(close, volume)
        vol_ma_s       = volume.rolling(20).mean()

        args = (df, rsi_s, ml_s, sl_s, hist_s, sk_s, sd_s, obv_s, vol_ma_s, ma20, ma50, ma200)

        # Today's snapshot
        today = _day(*args, idx=-1)
        today["symbol"]   = ticker
        today["date"]     = df.index[-1].strftime("%Y-%m-%d")
        today["sparkline"] = [round(float(x),2) for x in close.iloc[-60:]
                               if x is not None and not math.isnan(float(x))]
        today.update(_score_entry(today))
        today.update(_score_exit(today))
        today["error"] = None

        # Daily % change
        if len(close) >= 2:
            today["day_chg"] = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
        else:
            today["day_chg"] = None

        # Fundamentals from ticker.info
        try:
            info = yf.Ticker(ticker).info
            today["mkt_cap"]     = info.get("marketCap")
            today["pe_trailing"] = info.get("trailingPE")
            today["pe_forward"]  = info.get("forwardPE")
            today["eps_growth"]  = info.get("earningsGrowth")
            today["rev_growth"]  = info.get("revenueGrowth")
            today["beta"]        = info.get("beta")
            today["div_yield"]   = info.get("dividendYield")
            today["sector"]      = info.get("sector") or info.get("category", "")
            today["analyst"]     = info.get("recommendationKey")
            today["target_px"]   = info.get("targetMeanPrice")
            today["wk52_chg"]    = info.get("52WeekChange")
        except Exception:
            for k in ("mkt_cap","pe_trailing","pe_forward","eps_growth","rev_growth",
                      "beta","div_yield","sector","analyst","target_px","wk52_chg"):
                today.setdefault(k, None)

        # 90-day signal history
        n = len(df)
        hist_len = min(HISTORY_DAYS, n-2)
        history = []
        for i in range(-hist_len, 0):
            d = _day(*args, idx=i)
            se = _score_entry(d)
            sx = _score_exit(d)
            ind_f = sx["exit_ind_fired"]
            ex_type = ("stoploss" if "E7" in ind_f else
                       "profittake" if any(e in ind_f for e in ["E1","E2","E3"]) else
                       "caution") if sx["exit_signal"] else None
            history.append({
                "date":    df.index[i].strftime("%Y-%m-%d"),
                "close":   round(float(close.iloc[i]),2) if not math.isnan(float(close.iloc[i])) else None,
                "ma50":    round(float(ma50.iloc[i]),2)  if not math.isnan(float(ma50.iloc[i]))  else None,
                "ma200":   round(float(ma200.iloc[i]),2) if not math.isnan(float(ma200.iloc[i])) else None,
                "signal":  se["entry_signal"],
                "score":   se["entry_score"],
                "exit":    sx["exit_signal"],
                "ex_type": ex_type,
            })
        today["history"] = history
        return today

    except Exception as e:
        return {"symbol": ticker, "error": str(e)}

# ── HTML ───────────────────────────────────────────────────────────────────────
def build_html(results):
    ts = datetime.now().strftime("%A, %B %d %Y — %H:%M")
    buy_c   = sum(1 for r in results if r.get("entry_signal")=="BUY")
    watch_c = sum(1 for r in results if r.get("entry_signal")=="WATCH")
    exit_c  = sum(1 for r in results if r.get("exit_signal"))

    order = {"BUY":0,"WATCH":1,"HOLD":2}
    rs = sorted(results, key=lambda r:(3 if r.get("error") else order.get(r.get("entry_signal","HOLD"),2),
                                        -r.get("entry_score",0)))

    cards_html = "\n".join(_card(r) for r in rs)
    tl_html    = "\n".join(_tl_row(r) for r in rs if not r.get("error"))
    hist_json  = json.dumps({r["symbol"]:r.get("history",[])
                              for r in results if not r.get("error")})

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Pivot Scanner</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{
  --bg:#0f1117;--bg2:#1a1d27;--bg3:#242736;--border:#2e3347;
  --text:#e2e4ef;--muted:#7b82a0;--accent:#5b8ef0;
  --green:#34d399;--red:#f87171;--yellow:#fbbf24;--orange:#fb923c;
  color-scheme:dark;
}}
@media(prefers-color-scheme:light){{:root:not([data-theme="dark"]){{
  --bg:#f5f6fa;--bg2:#fff;--bg3:#eef0f8;--border:#d4d8ed;
  --text:#1a1d2e;--muted:#6b7094;--accent:#3b6fe0;
  --green:#059669;--red:#dc2626;--yellow:#d97706;--orange:#ea580c;
  color-scheme:light;
}}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
  font-size:14px;line-height:1.6;padding:24px 16px}}
.page{{max-width:1200px;margin:0 auto;display:flex;flex-direction:column;gap:24px}}
h1{{font-size:1.4rem;font-weight:700}}
.subtitle{{color:var(--muted);font-size:12px;margin-top:4px}}
.kpi-row{{display:flex;gap:12px;flex-wrap:wrap}}
.kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:12px 16px;min-width:110px}}
.kpi-btn{{cursor:pointer;transition:border-color .15s,box-shadow .15s}}
.kpi-btn:hover{{border-color:var(--accent)}}
.kpi-btn.active-filter{{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)44}}
.card.hidden{{display:none}}
.kpi-label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.kpi-val{{font-size:1.5rem;font-weight:700;font-variant-numeric:tabular-nums}}
/* Tabs */
.tabs{{display:flex;gap:0;border-bottom:1px solid var(--border)}}
.tab{{padding:8px 20px;cursor:pointer;font-weight:500;color:var(--muted);
  border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s}}
.tab.active,.tab:hover{{color:var(--text)}}
.tab.active{{border-bottom-color:var(--accent);color:var(--accent)}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}
/* Cards */
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:16px;padding-top:16px}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  padding:16px;display:flex;flex-direction:column;gap:12px}}
.card-header{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.ticker{{font-size:1.1rem;font-weight:700;font-family:'JetBrains Mono',monospace}}
.ticker-link{{cursor:pointer;text-decoration:underline dotted;text-underline-offset:3px}}
.ticker-link:hover{{color:var(--accent)}}
.price{{font-size:.95rem;font-weight:600;color:var(--muted);font-variant-numeric:tabular-nums;margin-right:auto}}
.badge{{display:inline-block;padding:3px 9px;border-radius:4px;font-size:11px;
  font-weight:700;letter-spacing:.05em;text-transform:uppercase}}
.badge-buy{{background:#34d39922;color:var(--green);border:1px solid #34d39944}}
.badge-watch{{background:#fbbf2422;color:var(--yellow);border:1px solid #fbbf2444}}
.badge-hold{{background:#7b82a022;color:var(--muted);border:1px solid #7b82a033}}
.badge-gate{{background:#f8717122;color:var(--red);border:1px solid #f8717133}}
.badge-profittake{{background:#34d39922;color:var(--green);border:1px solid #34d39966}}
.badge-stoploss{{background:#f8717133;color:var(--red);border:1px solid #f87171}}
.badge-caution{{background:#fbbf2422;color:var(--yellow);border:1px solid #fbbf2444}}
.score-bar{{height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}}
.score-fill{{height:100%;border-radius:3px}}
.rules{{display:flex;flex-direction:column;gap:3px}}
.rule{{display:flex;gap:6px;font-size:11.5px;align-items:baseline}}
.ri{{width:13px;flex-shrink:0;text-align:center}}
.rk{{font-weight:600;min-width:20px}}
.rt{{color:var(--muted)}}.rt.p{{color:var(--text)}}
.inds{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}}
.ind{{background:var(--bg3);border-radius:6px;padding:5px 8px}}
.ind-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.ind-val{{font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums}}
.red{{color:var(--red)}}.green{{color:var(--green)}}.yellow{{color:var(--yellow)}}
.note{{font-size:11px;border-radius:4px;padding:4px 8px}}
.note-gate{{color:var(--red);background:#f8717110;border:1px solid #f8717130}}
.note-exit{{color:var(--orange);background:#fb923c10;border:1px solid #fb923c30}}
canvas.spark{{width:100%;height:36px}}
/* Filter bar */
.filter-bar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  padding:12px 0 4px;border-bottom:1px solid var(--border);margin-bottom:4px}}
.flt{{background:var(--bg3);border:1px solid var(--border);border-radius:6px;
  color:var(--text);font-size:12px;padding:5px 10px;cursor:pointer;
  appearance:none;-webkit-appearance:none;min-width:120px;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%237b82a0'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center;padding-right:24px}}
.flt:focus{{outline:none;border-color:var(--accent)}}
.flt.active{{border-color:var(--accent);color:var(--accent)}}
.flt-pe{{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--muted)}}
.flt-inp{{width:54px;background:var(--bg3);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:12px;padding:5px 7px}}
.flt-inp:focus{{outline:none;border-color:var(--accent)}}
.flt-reset{{background:none;border:1px solid var(--border);border-radius:6px;
  color:var(--muted);font-size:12px;padding:5px 10px;cursor:pointer}}
.flt-reset:hover{{color:var(--red);border-color:var(--red)}}
.flt-count{{font-size:12px;color:var(--muted);margin-left:auto}}
.chg-pos{{color:var(--green)}}.chg-neg{{color:var(--red)}}
.sector-tag{{font-size:10px;color:var(--muted);background:var(--bg3);
  border-radius:3px;padding:1px 6px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;max-width:160px}}
.fund-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}}
.fund-item{{background:var(--bg3);border-radius:6px;padding:5px 8px}}
.fund-label{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
.fund-val{{font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}}
.analyst-badge{{display:inline-block;font-size:11px;font-weight:700;padding:2px 7px;
  border-radius:4px;background:var(--bg3)}}
/* Timeline */
.tl-list{{display:flex;flex-direction:column;gap:8px;padding-top:16px}}
.tl-card{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;overflow:hidden;position:relative}}
.tl-header{{display:flex;align-items:center;gap:10px;padding:10px 16px 6px}}
.tl-sym{{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:14px;min-width:56px}}
.tl-price{{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;margin-right:auto}}
canvas.tl{{width:100%;height:130px;display:block;padding:0 16px 10px;cursor:pointer}}
canvas.tl:hover{{opacity:.85}}
.tl-tooltip{{position:absolute;background:var(--bg3);border:1px solid var(--border);
  border-radius:6px;padding:5px 10px;font-size:12px;pointer-events:none;
  opacity:0;transition:opacity .1s;white-space:nowrap;z-index:10}}
/* Modal */
.modal-overlay{{position:fixed;inset:0;background:rgba(0,0,0,.78);display:flex;
  align-items:center;justify-content:center;z-index:200;
  opacity:0;pointer-events:none;transition:opacity .18s}}
.modal-overlay.open{{opacity:1;pointer-events:auto}}
.modal-box{{background:var(--bg2);border:1px solid var(--border);border-radius:14px;
  width:min(96vw,980px);padding:20px;display:flex;flex-direction:column;gap:14px;
  box-shadow:0 24px 80px #0008}}
.modal-header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
.modal-sym{{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:1.3rem}}
.modal-price{{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums;margin-right:auto}}
.modal-close{{margin-left:auto;cursor:pointer;font-size:22px;line-height:1;
  color:var(--muted);padding:2px 6px;border-radius:4px}}
.modal-close:hover{{color:var(--text);background:var(--bg3)}}
canvas.tl-big{{width:100%;height:280px;display:block}}
.modal-tip{{font-size:12px;color:var(--muted);min-height:18px;font-variant-numeric:tabular-nums}}
.modal-legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted)}}
.tl-legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);
  padding:0 16px 10px}}
.leg{{display:flex;align-items:center;gap:5px}}
.leg-sq{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
footer{{color:var(--muted);font-size:11px;text-align:center;padding-top:4px}}
.legend{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:0}}
.legend summary{{padding:12px 16px;cursor:pointer;font-weight:600;font-size:13px;
  list-style:none;display:flex;align-items:center;gap:8px;color:var(--text)}}
.legend summary::-webkit-details-marker{{display:none}}
.legend summary::after{{content:'›';margin-left:auto;transition:transform .2s;font-size:16px}}
.legend[open] summary::after{{transform:rotate(90deg)}}
.legend-body{{padding:0 16px 16px;display:flex;flex-direction:column;gap:20px}}
.leg-section{{display:flex;flex-direction:column;gap:8px}}
.leg-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.leg-grid{{display:grid;grid-template-columns:160px 1fr;gap:6px 12px;align-items:start}}
.leg-item{{display:flex;align-items:center;padding:4px 10px;border-radius:5px;
  border:1px solid var(--border);font-size:12px;font-weight:600}}
.badge-buy-bg{{background:#34d39915;border-color:#34d39966;color:var(--green)}}
.badge-watch-bg{{background:#fbbf2415;border-color:#fbbf2466;color:var(--yellow)}}
.badge-hold-bg{{background:#7b82a015;border-color:#7b82a040;color:var(--muted)}}
.leg-desc{{font-size:12px;color:var(--muted);line-height:1.5;padding-top:4px}}
.leg-rules{{font-size:12px;color:var(--muted);line-height:1.9;background:var(--bg3);
  border-radius:6px;padding:10px 12px}}
</style>
</head>
<body>
<div class="page">
  <div>
    <h1>Pivot Scanner</h1>
    <div class="subtitle">Generated {ts}</div>
  </div>
  <div class="kpi-row">
    <div class="kpi kpi-btn" data-filter="ALL"><div class="kpi-label">All Symbols</div><div class="kpi-val">{len(results)}</div></div>
    <div class="kpi kpi-btn active-filter" data-filter="BUY"><div class="kpi-label">↑ BUY</div><div class="kpi-val green">{buy_c}</div></div>
    <div class="kpi kpi-btn" data-filter="WATCH"><div class="kpi-label">◉ WATCH</div><div class="kpi-val yellow">{watch_c}</div></div>
    <div class="kpi kpi-btn" data-filter="HOLD"><div class="kpi-label">— HOLD</div><div class="kpi-val" style="color:var(--muted)">{len(results)-buy_c-watch_c-exit_c}</div></div>
    <div class="kpi kpi-btn" data-filter="EXIT"><div class="kpi-label">Exit signals</div><div class="kpi-val" style="color:var(--orange)">{exit_c}</div></div>
  </div>

  <details class="legend">
    <summary>📖 Signal Legend — click to expand</summary>
    <div class="legend-body">
      <div class="leg-section">
        <div class="leg-title">Entry Signals (score 0–120, 20 pts each)</div>
        <div class="leg-grid">
          <div class="leg-item badge-buy-bg"><b>↑ BUY</b> ≥60 pts</div>
          <div class="leg-desc">At least 3 of 6 entry conditions are met. This is the trigger to consider opening or adding to a position.</div>
          <div class="leg-item badge-watch-bg"><b>◉ WATCH</b> 40–59 pts</div>
          <div class="leg-desc">2 of 6 conditions met — setup is forming but not ready. Monitor daily; may become a BUY.</div>
          <div class="leg-item badge-hold-bg"><b>— HOLD</b> 0–39 pts</div>
          <div class="leg-desc">Fewer than 2 conditions met. No action — stock is not in an oversold dip. If you own it, keep holding.</div>
        </div>
        <div class="leg-rules">
          <b>The 6 entry rules (20 pts each):</b><br>
          <b>R1</b> RSI(14) &lt; 40 — stock is oversold on momentum<br>
          <b>R2</b> MACD histogram &lt; 0 but rising — selling pressure is easing<br>
          <b>R3</b> Stochastic %K crosses above %D while both &lt; 50 — early bullish flip<br>
          <b>R4</b> Volume above 20-day average — buyers stepping in with conviction<br>
          <b>R5</b> Price ≤ MA50 × 1.01 — stock is near or below its 50-day average (value zone)<br>
          <b>R6</b> OBV (On-Balance Volume) rising — money flowing into the stock
        </div>
      </div>
      <div class="leg-section">
        <div class="leg-title">Exit Signals (shown when you should consider selling)</div>
        <div class="leg-grid">
          <div class="leg-item" style="background:#34d39915;border-color:#34d39966"><b style="color:var(--green)">↑ PROFIT TAKE</b></div>
          <div class="leg-desc">Stock has run up and is now <b>overbought</b>. RSI &gt; 70 and falling, MACD histogram turning negative from above zero, or Stochastic doing a bearish cross from overbought. If you entered on a BUY signal and the stock has gained — <b>this is your sell signal.</b></div>
          <div class="leg-item" style="background:#f8717115;border-color:#f87171"><b style="color:var(--red)">⬇ STOP LOSS</b></div>
          <div class="leg-desc">Price has broken <b>below its 200-day moving average</b> — the long-term trend has reversed. If you're holding a position, this is a hard exit to protect capital.</div>
          <div class="leg-item" style="background:#fbbf2415;border-color:#fbbf2466"><b style="color:var(--yellow)">⚠ CAUTION</b></div>
          <div class="leg-desc">Momentum is weakening (MACD bearish, OBV falling, or price slipping below MA20 on volume) but the stock is not yet overbought. <b>Not a sell signal by itself</b> — watch closely for a PROFIT TAKE to follow.</div>
        </div>
      </div>
      <div class="leg-section">
        <div class="leg-title">Timeline Chart Colors</div>
        <div class="leg-grid">
          <div class="leg-item" style="background:#34d39930;border-color:#34d39966">Green zone</div>
          <div class="leg-desc">BUY signal was active on those days — entry conditions were met.</div>
          <div class="leg-item" style="background:#fbbf2430;border-color:#fbbf2466">Yellow zone</div>
          <div class="leg-desc">WATCH — setup was forming but not complete.</div>
          <div class="leg-item" style="background:#7b82a015;border-color:#7b82a040">Grey zone</div>
          <div class="leg-desc">HOLD — no significant entry conditions active.</div>
          <div class="leg-item" style="color:var(--green)">▼ Green triangle</div>
          <div class="leg-desc">PROFIT TAKE exit signal fired on that day.</div>
          <div class="leg-item" style="color:var(--red)">▼ Red triangle</div>
          <div class="leg-desc">STOP LOSS exit signal fired — price broke below MA200.</div>
          <div class="leg-item" style="color:var(--yellow)">▼ Yellow triangle</div>
          <div class="leg-desc">CAUTION exit signal fired — momentum weakening.</div>
        </div>
      </div>
    </div>
  </details>

  <div class="tabs">
    <div class="tab active" data-panel="p-today">Today's Signals</div>
    <div class="tab" data-panel="p-timeline">Signal Timeline (90 days)</div>
  </div>

  <div id="p-today" class="tab-panel active">
    <div class="filter-bar">
      <select class="flt" id="f-mktcap" title="Market Cap">
        <option value="">Mkt Cap: All</option>
        <option value="mega">Mega ($200B+)</option>
        <option value="large">Large ($10B–$200B)</option>
        <option value="mid">Mid ($2B–$10B)</option>
        <option value="small">Small ($300M–$2B)</option>
        <option value="micro">Micro (&lt;$300M)</option>
      </select>
      <select class="flt" id="f-analyst" title="Analyst Rating">
        <option value="">Analyst: All</option>
        <option value="strong_buy">⬆ Strong Buy</option>
        <option value="buy">↑ Buy</option>
        <option value="hold">— Hold</option>
        <option value="sell">↓ Sell / Strong Sell</option>
      </select>
      <select class="flt" id="f-chg" title="Today's Change">
        <option value="">Today: All</option>
        <option value="up">↑ Up today</option>
        <option value="down">↓ Down today</option>
      </select>
      <select class="flt" id="f-exit" title="Exit Signal">
        <option value="">Exit: All</option>
        <option value="yes">Has exit signal</option>
        <option value="no">No exit signal</option>
      </select>
      <select class="flt" id="f-sector" title="Sector">
        <option value="">Sector: All</option>
      </select>
      <div class="flt-pe">
        Score:&nbsp;<input class="flt-inp" id="f-score-min" placeholder="0" type="number" min="0" max="120" step="20">
        <span>–</span><input class="flt-inp" id="f-score-max" placeholder="120" type="number" min="0" max="120" step="20">
      </div>
      <div class="flt-pe">
        Rev Grw ≥&nbsp;<input class="flt-inp" id="f-revgrow-min" placeholder="%" type="number" step="1">%
      </div>
      <div class="flt-pe">
        EPS Grw ≥&nbsp;<input class="flt-inp" id="f-epsgrow-min" placeholder="%" type="number" step="1">%
      </div>
      <div class="flt-pe">
        P/E:&nbsp;<input class="flt-inp" id="f-pe-min" placeholder="min" type="number" min="0" step="1">
        <span>–</span><input class="flt-inp" id="f-pe-max" placeholder="max" type="number" min="0" step="1">
      </div>
      <div class="flt-pe">
        Fwd P/E:&nbsp;<input class="flt-inp" id="f-fwdpe-min" placeholder="min" type="number" min="0" step="1">
        <span>–</span><input class="flt-inp" id="f-fwdpe-max" placeholder="max" type="number" min="0" step="1">
      </div>
      <div class="flt-pe">
        Beta ≤&nbsp;<input class="flt-inp" id="f-beta-max" placeholder="e.g. 1.5" type="number" step="0.1">
      </div>
      <div class="flt-pe">
        52W Chg ≥&nbsp;<input class="flt-inp" id="f-wk52-min" placeholder="%" type="number" step="1">%
      </div>
      <button class="flt-reset" id="f-reset">✕ Clear</button>
      <span class="flt-count" id="f-count"></span>
    </div>
    <div class="grid">{cards_html}</div>
  </div>

  <div id="p-timeline" class="tab-panel">
    <div class="tl-legend" style="padding-top:12px">
      <span class="leg"><span class="leg-sq" style="background:#34d39930;border:1px solid #34d39966"></span>BUY zone</span>
      <span class="leg"><span class="leg-sq" style="background:#fbbf2430;border:1px solid #fbbf2466"></span>WATCH zone</span>
      <span class="leg"><span class="leg-sq" style="background:#7b82a015;border:1px solid #7b82a040"></span>HOLD</span>
      <span class="leg" style="color:var(--green)">▼ PROFIT TAKE day</span>
      <span class="leg" style="color:var(--red)">▼ STOP LOSS day</span>
      <span class="leg" style="color:var(--yellow)">▼ CAUTION day</span>
    </div>
    <div class="tl-list">{tl_html}</div>
  </div>

  <footer>Entry: 6 rules × 20 pts — score ≥60 = BUY · Exit: ≥2 signals incl. ≥1 independent · Data from Yahoo Finance, updated daily after market close</footer>
</div>

<!-- Enlarged chart modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal-box" id="modal-box">
    <div class="modal-header">
      <span class="modal-sym" id="modal-sym"></span>
      <span class="modal-price" id="modal-price"></span>
      <span id="modal-badges"></span>
      <span class="modal-close" id="modal-close">✕</span>
    </div>
    <div class="modal-legend">
      <span class="leg"><span class="leg-sq" style="background:#34d39930;border:1px solid #34d39966"></span>BUY zone</span>
      <span class="leg"><span class="leg-sq" style="background:#fbbf2430;border:1px solid #fbbf2466"></span>WATCH zone</span>
      <span class="leg"><span class="leg-sq" style="background:#7b82a015;border:1px solid #7b82a040"></span>HOLD</span>
      <span class="leg" style="color:var(--green)">▼ PROFIT TAKE</span>
      <span class="leg" style="color:var(--red)">▼ STOP LOSS</span>
      <span class="leg" style="color:var(--yellow)">▼ CAUTION</span>
      <span style="margin-left:auto;font-size:11px;color:var(--muted)">— MA50 &nbsp;&nbsp; ╌ MA200</span>
    </div>
    <canvas class="tl-big" id="modal-canvas"></canvas>
    <div class="modal-tip" id="modal-tip">Hover over the chart to see daily details</div>
  </div>
</div>

<script>
// Tab switching
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab,.tab-panel').forEach(e => e.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');
    if (tab.dataset.panel === 'p-timeline') drawTimelines();
  }});
}});

// Sparklines (today)
function drawSpark(canvas) {{
  const data = JSON.parse(canvas.dataset.prices);
  if (!data.length) return;
  const dpr = devicePixelRatio || 1;
  const W = canvas.offsetWidth, H = 36;
  canvas.width = W*dpr; canvas.height = H*dpr;
  canvas.style.height = H+'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  const mn = Math.min(...data), mx = Math.max(...data), rng = mx-mn||1;
  const x = i => i/(data.length-1)*W;
  const y = v => H-2-((v-mn)/rng)*(H-6);
  const col = data[data.length-1]>=data[0] ? '#34d399' : '#f87171';
  ctx.strokeStyle=col; ctx.lineWidth=1.5;
  ctx.beginPath();
  data.forEach((v,i) => i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v)));
  ctx.stroke();
  ctx.lineTo(x(data.length-1),H); ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle=col+'18'; ctx.fill();
}}

window.addEventListener('load', () => document.querySelectorAll('canvas.spark').forEach(drawSpark));

// ── Combined filter engine ─────────────────────────────────────────────────────
const TOTAL_CARDS = document.querySelectorAll('.card[data-sig]').length;
const fState = {{ signal:'ALL', mktcap:'', analyst:'', chg:'', exit:'', scoreMin:'', scoreMax:'', revGrowMin:'', epsGrowMin:'', peMin:'', peMax:'', fwdPeMin:'', fwdPeMax:'', betaMax:'', wk52Min:'', sector:'' }};

// Populate sector dropdown from card data
(function() {{
  const sectors = new Set();
  document.querySelectorAll('.card[data-sector]').forEach(c => {{
    if (c.dataset.sector) sectors.add(c.dataset.sector);
  }});
  const sel = document.getElementById('f-sector');
  [...sectors].sort().forEach(s => {{
    const o = document.createElement('option');
    o.value = s; o.textContent = s; sel.appendChild(o);
  }});
}})();

function applyFilters() {{
  let shown = 0;
  document.querySelectorAll('.card[data-sig]').forEach(card => {{
    const sig     = card.dataset.sig;
    const hasExit = card.dataset.exit === 'true';
    const score   = parseFloat(card.dataset.score) || 0;
    const mcap    = parseFloat(card.dataset.mktcap) || 0;
    const analyst = card.dataset.analyst || '';
    const daychg  = parseFloat(card.dataset.daychg) || 0;
    const pe      = parseFloat(card.dataset.pe) || 0;
    const sector  = card.dataset.sector || '';
    const epsgrow = card.dataset.epsgrow !== '' ? parseFloat(card.dataset.epsgrow) : null;
    const revgrow = card.dataset.revgrow !== '' ? parseFloat(card.dataset.revgrow) : null;
    const fwdpe   = parseFloat(card.dataset.fwdpe) || 0;
    const beta    = card.dataset.beta  !== '' ? parseFloat(card.dataset.beta)  : null;
    const wk52    = card.dataset.wk52  !== '' ? parseFloat(card.dataset.wk52)  : null;
    let show = true;

    // Signal (KPI buttons)
    if (fState.signal !== 'ALL') {{
      if (fState.signal === 'BUY'   && sig !== 'BUY')   show = false;
      if (fState.signal === 'WATCH' && sig !== 'WATCH') show = false;
      if (fState.signal === 'HOLD'  && sig !== 'HOLD')  show = false;
      if (fState.signal === 'EXIT'  && !hasExit)        show = false;
    }}
    // Market cap (in $B stored in data-mktcap)
    if (fState.mktcap) {{
      if (fState.mktcap === 'mega'  && mcap < 200)                     show = false;
      if (fState.mktcap === 'large' && (mcap < 10  || mcap >= 200))    show = false;
      if (fState.mktcap === 'mid'   && (mcap < 2   || mcap >= 10))     show = false;
      if (fState.mktcap === 'small' && (mcap < 0.3 || mcap >= 2))      show = false;
      if (fState.mktcap === 'micro' && mcap >= 0.3)                    show = false;
    }}
    // Analyst rating
    if (fState.analyst) {{
      if (fState.analyst === 'sell') {{
        if (analyst !== 'sell' && analyst !== 'strong_sell') show = false;
      }} else if (analyst !== fState.analyst) show = false;
    }}
    // Daily change direction
    if (fState.chg === 'up'   && daychg <= 0) show = false;
    if (fState.chg === 'down' && daychg >= 0) show = false;
    // Exit signal
    if (fState.exit === 'yes' && !hasExit) show = false;
    if (fState.exit === 'no'  && hasExit)  show = false;
    // Entry score range
    if (fState.scoreMin !== '' && score < parseFloat(fState.scoreMin)) show = false;
    if (fState.scoreMax !== '' && score > parseFloat(fState.scoreMax)) show = false;
    // Revenue growth minimum (stored as %, e.g. 10 = 10%)
    if (fState.revGrowMin !== '') {{
      if (revgrow === null || revgrow < parseFloat(fState.revGrowMin)) show = false;
    }}
    // EPS growth minimum
    if (fState.epsGrowMin !== '') {{
      if (epsgrow === null || epsgrow < parseFloat(fState.epsGrowMin)) show = false;
    }}
    // P/E range
    if (fState.peMin !== '' && pe > 0 && pe < parseFloat(fState.peMin)) show = false;
    if (fState.peMax !== '' && pe > 0 && pe > parseFloat(fState.peMax)) show = false;
    // Fwd P/E range
    if (fState.fwdPeMin !== '' && fwdpe > 0 && fwdpe < parseFloat(fState.fwdPeMin)) show = false;
    if (fState.fwdPeMax !== '' && fwdpe > 0 && fwdpe > parseFloat(fState.fwdPeMax)) show = false;
    // Beta max
    if (fState.betaMax !== '') {{
      if (beta === null || beta > parseFloat(fState.betaMax)) show = false;
    }}
    // 52W Change minimum
    if (fState.wk52Min !== '') {{
      if (wk52 === null || wk52 < parseFloat(fState.wk52Min)) show = false;
    }}
    // Sector
    if (fState.sector && sector !== fState.sector) show = false;

    card.classList.toggle('hidden', !show);
    if (show) shown++;
  }});

  document.getElementById('f-count').textContent = `${{shown}} of ${{TOTAL_CARDS}} symbols`;

  // Mark active filter dropdowns
  ['f-mktcap','f-analyst','f-chg','f-exit','f-sector'].forEach(id => {{
    const el = document.getElementById(id);
    el.classList.toggle('active', !!el.value);
  }});
}}

// Wire up filter controls
['f-mktcap','f-analyst','f-chg','f-exit','f-sector'].forEach(id => {{
  document.getElementById(id).addEventListener('change', e => {{
    fState[id.replace('f-','').replace('-','').replace('mktcap','mktcap')
               .replace('analyst','analyst').replace('chg','chg')
               .replace('exit','exit').replace('sector','sector')] = e.target.value;
    // map id to fState key
    const keyMap = {{'f-mktcap':'mktcap','f-analyst':'analyst','f-chg':'chg','f-exit':'exit','f-sector':'sector'}};
    fState[keyMap[id]] = e.target.value;
    applyFilters();
  }});
}});
document.getElementById('f-score-min').addEventListener('input', e => {{ fState.scoreMin = e.target.value; applyFilters(); }});
document.getElementById('f-score-max').addEventListener('input', e => {{ fState.scoreMax = e.target.value; applyFilters(); }});
document.getElementById('f-revgrow-min').addEventListener('input', e => {{ fState.revGrowMin = e.target.value; applyFilters(); }});
document.getElementById('f-epsgrow-min').addEventListener('input', e => {{ fState.epsGrowMin = e.target.value; applyFilters(); }});
document.getElementById('f-pe-min').addEventListener('input', e => {{ fState.peMin = e.target.value; applyFilters(); }});
document.getElementById('f-pe-max').addEventListener('input', e => {{ fState.peMax = e.target.value; applyFilters(); }});
document.getElementById('f-fwdpe-min').addEventListener('input', e => {{ fState.fwdPeMin = e.target.value; applyFilters(); }});
document.getElementById('f-fwdpe-max').addEventListener('input', e => {{ fState.fwdPeMax = e.target.value; applyFilters(); }});
document.getElementById('f-beta-max').addEventListener('input', e => {{ fState.betaMax = e.target.value; applyFilters(); }});
document.getElementById('f-wk52-min').addEventListener('input', e => {{ fState.wk52Min = e.target.value; applyFilters(); }});
document.getElementById('f-reset').addEventListener('click', () => {{
  fState.signal = 'ALL';
  fState.mktcap = fState.analyst = fState.chg = fState.exit = fState.sector = '';
  fState.peMin = fState.peMax = fState.fwdPeMin = fState.fwdPeMax = fState.betaMax = fState.wk52Min = '';
  fState.scoreMin = fState.scoreMax = fState.revGrowMin = fState.epsGrowMin = '';
  ['f-mktcap','f-analyst','f-chg','f-exit','f-sector'].forEach(id => document.getElementById(id).value = '');
  document.getElementById('f-score-min').value = '';
  document.getElementById('f-score-max').value = '';
  document.getElementById('f-revgrow-min').value = '';
  document.getElementById('f-epsgrow-min').value = '';
  document.getElementById('f-pe-min').value = '';
  document.getElementById('f-pe-max').value = '';
  document.getElementById('f-fwdpe-min').value = '';
  document.getElementById('f-fwdpe-max').value = '';
  document.getElementById('f-beta-max').value = '';
  document.getElementById('f-wk52-min').value = '';
  document.querySelectorAll('.kpi-btn').forEach(b => b.classList.remove('active-filter'));
  document.querySelector('.kpi-btn[data-filter="ALL"]').classList.add('active-filter');
  applyFilters();
}});

// KPI buttons → set signal filter
document.querySelectorAll('.kpi-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    fState.signal = btn.dataset.filter;
    document.querySelectorAll('.kpi-btn').forEach(b => b.classList.remove('active-filter'));
    btn.classList.add('active-filter');
    // Switch to Today tab
    document.querySelectorAll('.tab,.tab-panel').forEach(e => e.classList.remove('active'));
    document.querySelector('[data-panel="p-today"]').classList.add('active');
    document.getElementById('p-today').classList.add('active');
    applyFilters();
  }});
}});

// Init count
applyFilters();

// Click ticker name → switch to timeline tab and scroll to that ticker
document.querySelectorAll('.ticker-link').forEach(el => {{
  el.addEventListener('click', () => {{
    document.querySelectorAll('.tab,.tab-panel').forEach(e => e.classList.remove('active'));
    document.querySelector('[data-panel="p-timeline"]').classList.add('active');
    document.getElementById('p-timeline').classList.add('active');
    drawTimelines();
    const card = document.getElementById('tl-' + el.dataset.sym);
    if (card) setTimeout(() => card.scrollIntoView({{behavior:'smooth',block:'start'}}), 80);
  }});
}});

// Timeline charts
const HIST = {hist_json};
const SIG_BG = {{
  BUY:       'rgba(52,211,153,0.15)',
  WATCH:     'rgba(251,191,36,0.12)',
  HOLD:      'rgba(123,130,160,0.05)',
  GATE_FAIL: 'rgba(248,113,113,0.12)',
}};
const SIG_LINE = {{
  BUY:       '#34d399',
  WATCH:     '#fbbf24',
  HOLD:      '#7b82a0',
  GATE_FAIL: '#f87171',
}};

function drawTimelines() {{
  // Double rAF: first fires after tab becomes display:block, second after layout reflow
  requestAnimationFrame(() => requestAnimationFrame(() => {{
    document.querySelectorAll('canvas.tl').forEach(c => {{
      if (c.offsetWidth > 0) drawOneTL(c);
    }});
  }}));
}}

function drawOneTL(canvas) {{
  const sym  = canvas.dataset.sym;
  const bars = HIST[sym] || [];
  if (!bars.length) return;

  const dpr = devicePixelRatio || 1;
  const pad = 16;
  const W   = canvas.offsetWidth - pad*2;
  const H   = 80;
  canvas.width  = (W + pad*2) * dpr;
  canvas.height = H * dpr;
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.translate(pad, 0);

  const prices = bars.map(b => b.close).filter(v=>v!=null);
  const ma50s  = bars.map(b => b.ma50).filter(v=>v!=null);
  const ma200s = bars.map(b => b.ma200).filter(v=>v!=null);
  const allP   = [...prices, ...ma50s, ...ma200s];
  const mn = Math.min(...allP)*0.998, mx = Math.max(...allP)*1.002;
  const rng = mx - mn || 1;
  const n = bars.length;
  const bw = W / n;

  const xc = i => (i + 0.5) * bw;
  const yp = v => 4 + (1-(v-mn)/rng)*(H-16);

  // Signal background bands
  bars.forEach((b,i) => {{
    ctx.fillStyle = SIG_BG[b.signal] || SIG_BG.HOLD;
    ctx.fillRect(i*bw, 0, bw+0.5, H-12);
  }});

  // MA200 line (dashed grey)
  ctx.setLineDash([3,3]);
  ctx.strokeStyle = 'rgba(123,130,160,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  let first200 = true;
  bars.forEach((b,i) => {{
    if (b.ma200==null) return;
    if (first200) {{ ctx.moveTo(xc(i), yp(b.ma200)); first200=false; }}
    else ctx.lineTo(xc(i), yp(b.ma200));
  }});
  ctx.stroke();

  // MA50 line (dashed lighter)
  ctx.strokeStyle = 'rgba(91,142,240,0.45)';
  ctx.beginPath();
  let first50 = true;
  bars.forEach((b,i) => {{
    if (b.ma50==null) return;
    if (first50) {{ ctx.moveTo(xc(i), yp(b.ma50)); first50=false; }}
    else ctx.lineTo(xc(i), yp(b.ma50));
  }});
  ctx.stroke();
  ctx.setLineDash([]);

  // Price line — colored by signal segment
  ctx.lineWidth = 1.8;
  let prevSig = null, prevX = 0, prevY = 0;
  bars.forEach((b,i) => {{
    if (b.close==null) return;
    const px = xc(i), py = yp(b.close);
    if (b.signal !== prevSig && i > 0) {{
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(prevX, prevY);
      ctx.lineTo(px, py);
      ctx.strokeStyle = SIG_LINE[b.signal]||'#7b82a0';
      ctx.stroke();
      ctx.beginPath();
    }}
    if (i===0||b.signal!==prevSig) {{
      ctx.strokeStyle = SIG_LINE[b.signal]||'#7b82a0';
      ctx.beginPath();
      ctx.moveTo(px, py);
    }} else {{
      ctx.lineTo(px, py);
    }}
    prevSig=b.signal; prevX=px; prevY=py;
    if (i===bars.length-1) ctx.stroke();
  }});

  // EXIT markers — colored by type, drawn above the price point
  const EXIT_COL = {{profittake:'#34d399', stoploss:'#f87171', caution:'#fbbf24'}};
  bars.forEach((b,i) => {{
    if (!b.exit||b.close==null) return;
    const px=xc(i), py=yp(b.close)-4;
    ctx.fillStyle = EXIT_COL[b.ex_type] || '#fb923c';
    ctx.beginPath();
    ctx.moveTo(px,     py+7);
    ctx.lineTo(px-4.5, py);
    ctx.lineTo(px+4.5, py);
    ctx.closePath(); ctx.fill();
  }});

  // Month labels on x-axis
  ctx.fillStyle='rgba(123,130,160,0.65)';
  ctx.font=`9px Inter,system-ui`;
  ctx.textAlign='center';
  let lastMo='';
  bars.forEach((b,i) => {{
    const mo=b.date.slice(0,7);
    if (mo!==lastMo) {{
      lastMo=mo;
      const lbl=new Date(b.date+'T12:00').toLocaleDateString('en-US',{{month:'short'}});
      ctx.fillText(lbl, xc(i), H-1);
    }}
  }});

  // Today's price label on right
  const last = bars[bars.length-1];
  if (last&&last.close) {{
    const py = yp(last.close);
    ctx.fillStyle = SIG_LINE[last.signal]||'#7b82a0';
    ctx.textAlign='right';
    ctx.font='bold 9px Inter,system-ui';
    ctx.fillText('$'+last.close.toFixed(2), W-2, py-2);
  }}

  // Hover tooltip — show date, price, signal
  const tip = document.getElementById('tt-'+sym);
  canvas.addEventListener('mousemove', e => {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left - pad;
    const idx = Math.round(mx / bw - 0.5);
    if (idx < 0 || idx >= bars.length) {{ tip.style.opacity=0; return; }}
    const b = bars[idx];
    if (!b.close) {{ tip.style.opacity=0; return; }}
    const sigLabel = {{BUY:'↑ BUY',WATCH:'◉ WATCH',HOLD:'— HOLD'}}[b.signal]||b.signal;
    tip.innerHTML = `<b>${{b.date}}</b>&nbsp; ${{b.close.toFixed(2)}} &nbsp;<span style="color:${{SIG_LINE[b.signal]||'#7b82a0'}}">${{sigLabel}}</span>${{b.exit?' &nbsp;<span style="color:#fb923c">EXIT</span>':''}}`;
    const tx = Math.min(e.clientX - rect.left + 10, rect.width - tip.offsetWidth - 4);
    tip.style.left = tx + 'px';
    tip.style.top  = (e.clientY - rect.top - 36) + 'px';
    tip.style.opacity = 1;
  }});
  canvas.addEventListener('mouseleave', () => {{ tip.style.opacity=0; }});
}}

// ── Enlarged modal ─────────────────────────────────────────────────────────────
function openModal(sym) {{
  const bars = HIST[sym] || [];
  if (!bars.length) return;
  const last = bars[bars.length-1];
  document.getElementById('modal-sym').textContent = sym;
  document.getElementById('modal-price').textContent = last && last.close ? '$'+last.close.toFixed(2) : '';
  // copy badges from the small card header
  const smallHeader = document.querySelector('.tl-card[id="tl-'+sym+'"] .tl-header');
  const badgesEl = document.getElementById('modal-badges');
  badgesEl.innerHTML = smallHeader ? Array.from(smallHeader.querySelectorAll('.badge')).map(b=>b.outerHTML).join(' ') : '';
  document.getElementById('modal-overlay').classList.add('open');
  requestAnimationFrame(() => drawBigTL(sym));
}}

function closeModal() {{
  document.getElementById('modal-overlay').classList.remove('open');
}}

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-overlay').addEventListener('click', e => {{
  if (e.target === document.getElementById('modal-overlay')) closeModal();
}});
document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeModal(); }});

// Click small canvas → open modal
document.querySelectorAll('canvas.tl').forEach(c => {{
  c.addEventListener('click', () => openModal(c.dataset.sym));
}});

function drawBigTL(sym) {{
  const bars = HIST[sym] || [];
  if (!bars.length) return;
  const canvas = document.getElementById('modal-canvas');
  const dpr = devicePixelRatio || 1;
  const padL = 62, padR = 12, padB = 22;
  const W = canvas.offsetWidth;
  const H = 240;
  canvas.width  = W * dpr;
  canvas.height = (H + padB) * dpr;
  canvas.style.height = (H + padB) + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const prices = bars.map(b=>b.close).filter(v=>v!=null);
  const ma50s  = bars.map(b=>b.ma50).filter(v=>v!=null);
  const ma200s = bars.map(b=>b.ma200).filter(v=>v!=null);
  const allP   = [...prices,...ma50s,...ma200s];
  const mn = Math.min(...allP)*0.997, mx = Math.max(...allP)*1.003;
  const rng = mx-mn||1;
  const n = bars.length;
  const plotW = W - padL - padR;
  const bw = plotW / n;

  const xc = i => padL + (i+0.5)*bw;
  const yp = v => 4 + (1-(v-mn)/rng)*(H-8);

  // Y-axis grid + labels
  const ticks = 5;
  ctx.textAlign='right'; ctx.textBaseline='middle';
  ctx.font='10px Inter,system-ui';
  for (let t=0; t<=ticks; t++) {{
    const v = mn + (rng*t/ticks);
    const y = yp(v);
    ctx.fillStyle='rgba(123,130,160,0.5)';
    ctx.fillText('$'+v.toFixed(v>=100?0:1), padL-6, y);
    ctx.strokeStyle='rgba(123,130,160,0.12)';
    ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
  }}

  // Signal background bands
  bars.forEach((b,i) => {{
    ctx.fillStyle = SIG_BG[b.signal]||SIG_BG.HOLD;
    ctx.fillRect(xc(i)-bw/2, 0, bw+0.5, H);
  }});

  // MA200 dashed grey
  ctx.setLineDash([4,4]); ctx.strokeStyle='rgba(123,130,160,0.45)'; ctx.lineWidth=1.2;
  ctx.beginPath(); let f200=true;
  bars.forEach((b,i)=>{{ if(b.ma200==null)return; if(f200){{ctx.moveTo(xc(i),yp(b.ma200));f200=false;}}else ctx.lineTo(xc(i),yp(b.ma200)); }});
  ctx.stroke();

  // MA50 dashed blue
  ctx.strokeStyle='rgba(91,142,240,0.5)';
  ctx.beginPath(); let f50=true;
  bars.forEach((b,i)=>{{ if(b.ma50==null)return; if(f50){{ctx.moveTo(xc(i),yp(b.ma50));f50=false;}}else ctx.lineTo(xc(i),yp(b.ma50)); }});
  ctx.stroke();
  ctx.setLineDash([]);

  // Price line — colored by signal segment
  ctx.lineWidth=2.2;
  let prevSig=null,prevX=0,prevY=0;
  bars.forEach((b,i) => {{
    if (b.close==null) return;
    const px=xc(i), py=yp(b.close);
    if (b.signal!==prevSig && i>0) {{
      ctx.stroke(); ctx.beginPath(); ctx.moveTo(prevX,prevY); ctx.lineTo(px,py);
      ctx.strokeStyle=SIG_LINE[b.signal]||'#7b82a0'; ctx.stroke(); ctx.beginPath();
    }}
    if (i===0||b.signal!==prevSig) {{
      ctx.strokeStyle=SIG_LINE[b.signal]||'#7b82a0'; ctx.beginPath(); ctx.moveTo(px,py);
    }} else {{ ctx.lineTo(px,py); }}
    prevSig=b.signal; prevX=px; prevY=py;
    if (i===bars.length-1) ctx.stroke();
  }});

  // Exit triangles (larger)
  const EXIT_COL={{profittake:'#34d399',stoploss:'#f87171',caution:'#fbbf24'}};
  bars.forEach((b,i) => {{
    if (!b.exit||b.close==null) return;
    const px=xc(i), py=yp(b.close)-6;
    ctx.fillStyle=EXIT_COL[b.ex_type]||'#fb923c';
    ctx.beginPath(); ctx.moveTo(px,py+10); ctx.lineTo(px-6,py); ctx.lineTo(px+6,py);
    ctx.closePath(); ctx.fill();
  }});

  // X-axis date labels — every ~2 weeks
  ctx.fillStyle='rgba(123,130,160,0.7)'; ctx.font='10px Inter,system-ui';
  ctx.textAlign='center'; ctx.textBaseline='top';
  let lastLabel='';
  bars.forEach((b,i) => {{
    const d=new Date(b.date+'T12:00');
    const day=d.getDate(), mo=d.getMonth();
    const key=mo+'-'+(day<15?'a':'b');
    if (key!==lastLabel) {{
      lastLabel=key;
      const lbl=d.toLocaleDateString('en-US',{{month:'short',day:'numeric'}});
      ctx.fillText(lbl, xc(i), H+4);
    }}
  }});

  // Hover interaction
  const tip = document.getElementById('modal-tip');
  canvas.onmousemove = e => {{
    const rect=canvas.getBoundingClientRect();
    const mx=e.clientX-rect.left;
    const idx=Math.round((mx-padL)/bw-0.5);
    if (idx<0||idx>=bars.length) {{ tip.textContent='Hover over the chart to see daily details'; return; }}
    const b=bars[idx];
    if (!b.close) return;
    const sigLabel={{BUY:'↑ BUY',WATCH:'◉ WATCH',HOLD:'— HOLD'}}[b.signal]||b.signal;
    const col=SIG_LINE[b.signal]||'#7b82a0';
    const exitPart=b.exit?' · <span style="color:'+((EXIT_COL[b.ex_type])||'#fb923c')+'">'+
      (b.ex_type==='stoploss'?'⬇ STOP LOSS':b.ex_type==='profittake'?'↑ PROFIT TAKE':'⚠ CAUTION')+'</span>':'';
    tip.innerHTML=`<b>${{b.date}}</b>&nbsp; ${{b.close.toFixed(2)}} &nbsp;<span style="color:${{col}}">${{sigLabel}}</span>${{exitPart}}`;
  }};
  canvas.onmouseleave = () => {{ tip.textContent='Hover over the chart to see daily details'; }};
}}
</script>
</body>
</html>"""

def _badge(sig, score):
    cls = {"BUY":"badge-buy","WATCH":"badge-watch","HOLD":"badge-hold"}.get(sig,"badge-hold")
    label = {"BUY":"↑ BUY","WATCH":"◉ WATCH","HOLD":"— HOLD"}.get(sig, sig)
    return f'<span class="badge {cls}">{label} {score}/120</span>'

def _exit_badge(r):
    if not r.get("exit_signal"): return ""
    ind = r.get("exit_ind_fired", [])
    if "E7" in ind:
        return '<span class="badge badge-stoploss" title="Price broke below MA200 — trend reversal">⬇ STOP LOSS</span>'
    if any(e in ind for e in ["E1","E2","E3"]):
        fired = " + ".join(e for e in ["E1","E2","E3"] if e in ind)
        return f'<span class="badge badge-profittake" title="Overbought — {fired} fired">↑ PROFIT TAKE</span>'
    pair = r.get("exit_pair_fired",[])
    fired = " + ".join(ind + pair)
    return f'<span class="badge badge-caution" title="Momentum weakening — {fired}">⚠ CAUTION</span>'

def _card(r):
    if r.get("error"):
        return (f'<div class="card"><div class="card-header">'
                f'<span class="ticker">{r["symbol"]}</span></div>'
                f'<p class="red" style="font-size:12px">Error: {r["error"]}</p></div>')

    sig, score = r.get("entry_signal","HOLD"), r.get("entry_score",0)
    close = r.get("close",0)
    spark = json.dumps(r.get("sparkline",[]))
    sc = "#34d399" if score>=60 else "#fbbf24" if score>=40 else "#7b82a0"

    rules_html = ""
    for k, info in r.get("entry_rules",{}).items():
        col = "var(--green)" if info["passed"] else "var(--red)"
        icon = "✓" if info["passed"] else "✗"
        txt = info["ok"] if info["passed"] else info["fail"]
        rules_html += (f'<div class="rule">'
                       f'<span class="ri" style="color:{col}">{icon}</span>'
                       f'<span class="rk" style="color:{col}">{k}</span>'
                       f'<span class="rt {"p" if info["passed"] else ""}">{txt}</span></div>')

    rsi, hist, sk, sd = r.get("rsi"), r.get("macd_hist"), r.get("stoch_k"), r.get("stoch_d")
    ma50, ma200 = r.get("ma50"), r.get("ma200")
    vol, vol_ma = r.get("volume"), r.get("vol_ma20")
    vol_ratio   = (vol/vol_ma) if (vol and vol_ma and vol_ma>0) else None
    rsi_cls  = "red" if (rsi and rsi>70) else "green" if (rsi and rsi<40) else ""
    hist_cls = "green" if (hist and hist>0) else "red" if (hist and hist<0) else ""
    ma_cls   = "green" if (close and ma200 and close>ma200) else "red"

    def ind(lbl, val, cls=""):
        return (f'<div class="ind"><div class="ind-label">{lbl}</div>'
                f'<div class="ind-val {cls}">{val}</div></div>')

    inds = (ind("RSI(14)", _f(rsi), rsi_cls) +
            ind("MACD Hist", _f(hist,2), hist_cls) +
            ind("Stoch K/D", f"{_f(sk)}/{_f(sd)}") +
            ind("vs MA50", f"{((close/ma50-1)*100):+.1f}%" if (close and ma50) else "n/a",
                "green" if (close and ma50 and close<=ma50*1.01) else "") +
            ind("vs MA200", f"{((close/ma200-1)*100):+.1f}%" if (close and ma200) else "n/a", ma_cls) +
            ind("Vol ratio", f"{vol_ratio:.1f}×" if vol_ratio else "n/a",
                "green" if (vol_ratio and vol_ratio>1) else ""))

    exit_b = _exit_badge(r)
    exit_n = ""
    if r.get("exit_signal"):
        ind = r.get("exit_ind_fired",[])
        pair = r.get("exit_pair_fired",[])
        descriptions = {"E1":"RSI &gt;70 falling","E2":"MACD hist rolling over","E3":"Stoch bear cross",
                        "E4":"MACD bearish","E5":"Price &lt; MA20 on vol","E6":"OBV falling","E7":"Price &lt; MA200"}
        fired_desc = " · ".join(descriptions.get(e,e) for e in ind+pair)
        exit_n = f'<div class="note note-exit">{fired_desc}</div>'

    sym = r["symbol"]
    exit_attr  = 'true' if r.get("exit_signal") else 'false'

    # Daily change
    day_chg = r.get("day_chg")
    if day_chg is not None:
        chg_cls = "chg-pos" if day_chg >= 0 else "chg-neg"
        chg_html = f'<span class="{chg_cls}" style="font-size:12px;font-weight:600">{day_chg:+.2f}%</span>'
    else:
        chg_html = ""

    # Sector tag
    sector = r.get("sector") or ""
    sector_html = f'<span class="sector-tag">{sector}</span>' if sector else ""

    # Fundamentals row
    pe   = r.get("pe_trailing")
    fpe  = r.get("pe_forward")
    mcap = r.get("mkt_cap")
    beta = r.get("beta")
    epsg = r.get("eps_growth")
    revg = r.get("rev_growth")
    divy = r.get("div_yield")
    anlst= r.get("analyst")
    tgt  = r.get("target_px")
    wk52 = r.get("wk52_chg")

    tgt_upside = f"{((tgt/close-1)*100):+.0f}%" if (tgt and close and close > 0) else "n/a"
    wk52_html  = f"{wk52*100:+.0f}%" if wk52 is not None else "n/a"
    wk52_cls   = "chg-pos" if (wk52 and wk52 > 0) else "chg-neg" if wk52 else ""

    anlst_lbl, anlst_col = ANALYST_LABELS.get(anlst or "", ("n/a", "muted"))

    # Data attributes for JS filtering
    mktcap_b    = f"{mcap/1e9:.2f}" if mcap else "0"
    pe_attr     = f"{pe:.1f}" if pe else "0"
    fpe_attr    = f"{fpe:.1f}" if fpe else "0"
    chg_attr    = f"{day_chg:.2f}" if day_chg is not None else "0"
    anlst_attr  = anlst or ""
    sec_attr    = (sector or "").replace('"', "")
    epsg_attr   = f"{epsg*100:.1f}" if epsg is not None else ""
    revg_attr   = f"{revg*100:.1f}" if revg is not None else ""
    beta_attr   = f"{beta:.2f}" if beta is not None else ""
    wk52_attr   = f"{wk52*100:.1f}" if wk52 is not None else ""

    def fi(lbl, val, cls=""):
        return (f'<div class="fund-item"><div class="fund-label">{lbl}</div>'
                f'<div class="fund-val {cls}">{val}</div></div>')

    funds = (fi("Mkt Cap",   _fmt_mcap(mcap)) +
             fi("P/E",       _f(pe,1) if pe else "n/a") +
             fi("Fwd P/E",   _f(fpe,1) if fpe else "n/a") +
             fi("Beta",      _f(beta,2) if beta else "n/a") +
             fi("EPS Grw",   _fmt_pct(epsg), "chg-pos" if (epsg and epsg>0) else "chg-neg" if epsg else "") +
             fi("Rev Grw",   _fmt_pct(revg), "chg-pos" if (revg and revg>0) else "chg-neg" if revg else "") +
             fi("Div Yield", _fmt_pct(divy) if divy else "—") +
             fi("52W Chg",   wk52_html, wk52_cls))

    analyst_html = f'<span class="analyst-badge" style="color:var(--{anlst_col})">{anlst_lbl}</span>'
    target_html  = (f'<span style="font-size:11px;color:var(--muted)">Target '
                    f'<b>${_f(tgt,0)}</b> ({tgt_upside})</span>') if tgt else ""

    return f"""<div class="card" data-sig="{sig}" data-exit="{exit_attr}" data-score="{score}" data-mktcap="{mktcap_b}" data-pe="{pe_attr}" data-fwdpe="{fpe_attr}" data-daychg="{chg_attr}" data-analyst="{anlst_attr}" data-sector="{sec_attr}" data-epsgrow="{epsg_attr}" data-revgrow="{revg_attr}" data-beta="{beta_attr}" data-wk52="{wk52_attr}">
  <div class="card-header">
    <span class="ticker ticker-link" data-sym="{sym}" title="Click to see {sym} timeline">{sym}</span>
    <span class="price">${_f(close,2)}</span>
    {chg_html}
    {_badge(sig,score)} {exit_b}
  </div>
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
    {sector_html} {analyst_html} {target_html}
  </div>
  <canvas class="spark" data-prices='{spark}'></canvas>
  <div class="score-bar"><div class="score-fill" style="width:{min(score/120*100,100):.0f}%;background:{sc}"></div></div>
  <div class="rules">{rules_html}</div>
  <div class="inds">{inds}</div>
  <div class="fund-row">{funds}</div>
  {exit_n}
</div>"""

def _tl_row(r):
    if r.get("error"): return ""
    sig, score = r.get("entry_signal","HOLD"), r.get("entry_score",0)
    close = r.get("close",0)
    sym = r["symbol"]
    day_chg = r.get("day_chg")
    chg_html = ""
    if day_chg is not None:
        cls = "chg-pos" if day_chg >= 0 else "chg-neg"
        chg_html = f'<span class="{cls}" style="font-size:11px;font-weight:600">{day_chg:+.2f}%</span>'
    sector = r.get("sector") or ""
    sec_html = f'<span class="sector-tag">{sector}</span>' if sector else ""
    mcap = _fmt_mcap(r.get("mkt_cap"))
    pe   = _f(r.get("pe_trailing"),1) if r.get("pe_trailing") else "—"
    return f"""<div class="tl-card" id="tl-{sym}">
  <div class="tl-header">
    <span class="tl-sym">{sym}</span>
    <span class="tl-price">${_f(close,2)}</span>
    {chg_html}
    <span style="font-size:11px;color:var(--muted)">Cap {mcap}</span>
    <span style="font-size:11px;color:var(--muted)">P/E {pe}</span>
    {sec_html}
    {_badge(sig,score)} {_exit_badge(r)}
  </div>
  <canvas class="tl" data-sym="{sym}"></canvas>
  <div class="tl-tooltip" id="tt-{sym}"></div>
</div>"""

# ── Run ────────────────────────────────────────────────────────────────────────
def main():
    print(f"Pivot Scanner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Scanning {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}\n")
    results = []
    for sym in SYMBOLS:
        print(f"  {sym:<8}", end="", flush=True)
        r = analyze_symbol(sym)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            sig, score = r.get("entry_signal","?"), r.get("entry_score",0)
            exit_w = " ⚠ EXIT" if r.get("exit_signal") else ""
            print(f"{sig:<10} score={score:3d}/120  RSI={_f(r.get('rsi'))}{exit_w}")
        results.append(r)

    html = build_html(results)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved: {REPORT_PATH}")
    if OPEN_BROWSER:
        webbrowser.open(f"file:///{REPORT_PATH.replace(os.sep, '/')}")

if __name__ == "__main__":
    main()
