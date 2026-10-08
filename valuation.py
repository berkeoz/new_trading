"""
Intrinsic value estimates — compare a company's estimated value with its price.

Stocks:
    * DCF: trailing-12-month free cash flow grown at g1 for years 1-5 and g2
      (half-way between g1 and the terminal rate) for years 6-10, a Gordon
      terminal value, discounted at r = 10Y yield + beta x equity risk premium
      (clamped to 8-14%); plus cash minus debt, divided by shares.
    * Bear / base / bull: base g1 from analysts' growth estimate (capped),
      bear and bull = base -/+ 10 percentage points.
    * Reverse DCF: the g1 that makes the DCF value equal to today's price.
    * Multiples: P/E and P/FCF at each fiscal year end vs today.
ETFs:
    * Earnings yield (1 / P/E) vs the 10Y Treasury yield (equity risk premium).

Every figure is an estimate that depends on the assumptions; data-quality
warnings are returned in "flags". Not investment advice.

Usage:
    python valuation.py NVDA            # prints a summary
"""

import sys, math
import numpy as np
import pandas as pd
import yfinance as yf

ERP = 0.05            # equity risk premium used for the discount rate
TERMINAL = 0.03       # long-run growth after year 10
R_MIN, R_MAX = 0.08, 0.14
G_MIN, G_BASE_MAX, G_BULL_MAX = -0.05, 0.25, 0.35


def dcf_value(fcf0, g1, g2, r, tg, cash, debt, shares):
    """Per-share value from a two-stage DCF (5 + 5 years) plus terminal value."""
    if r <= tg or shares <= 0:
        return None
    pv, f = 0.0, fcf0
    for y in range(1, 11):
        f *= 1 + (g1 if y <= 5 else g2)
        pv += f / (1 + r) ** y
    tv = f * (1 + tg) / (r - tg) / (1 + r) ** 10
    return (pv + tv + cash - debt) / shares


def implied_growth(price, fcf0, r, tg, cash, debt, shares):
    """Year 1-5 growth (with g2 = (g1 + tg) / 2) that makes the DCF equal the price."""
    f = lambda g: dcf_value(fcf0, g, (g + tg) / 2, r, tg, cash, debt, shares) - price
    lo, hi = -0.5, 1.5
    if fcf0 <= 0 or f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def _row(df, *names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def valuation(sym):
    sym = sym.upper()
    t = yf.Ticker(sym)
    info = t.info or {}
    hist = t.history(period="6y", interval="1d", auto_adjust=False)["Close"].dropna()
    if hist.empty:
        raise ValueError(f"no price data for {sym}")
    hist.index = hist.index.tz_localize(None)
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice")) or float(hist.iloc[-1])
    tnx = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
    rf = float(tnx.iloc[-1]) / 100 if not tnx.empty else 0.045
    out = {"symbol": sym, "name": info.get("shortName") or info.get("longName") or sym,
           "quote_type": info.get("quoteType"), "price": round(price, 2),
           "currency": info.get("currency"), "ten_year": round(rf, 4), "flags": []}

    # ── ETFs: earnings yield vs bonds ──
    if info.get("quoteType") == "ETF":
        pe = _num(info.get("trailingPE"))
        out["etf"] = {"pe": None if pe is None else round(pe, 2)}
        if pe and pe > 0:
            ey = 1 / pe
            out["etf"].update({"earnings_yield": round(ey, 4), "equity_risk_premium": round(ey - rf, 4)})
        else:
            out["flags"].append("Yahoo has no P/E for this ETF, so its earnings yield cannot be computed.")
        return out

    # ── Stocks ──
    fin_cur = info.get("financialCurrency")
    mixed = bool(fin_cur and out["currency"] and fin_cur != out["currency"])
    if mixed:
        out["flags"].append(f"Financials are reported in {fin_cur} but the stock trades in {out['currency']} "
                            "(often an ADR with its own share ratio), so the DCF and the P/E and P/FCF history are "
                            "skipped. Yahoo's own P/E figures are still shown.")
    cf, qcf, inc, bs = t.cashflow, t.quarterly_cashflow, t.income_stmt, t.balance_sheet
    fcf_a = _row(cf, "Free Cash Flow")
    fcf_hist = {} if fcf_a is None else {str(k.date()): round(float(v), 0) for k, v in fcf_a.dropna().items()}
    fcf_q = _row(qcf, "Free Cash Flow")
    fcf_q = None if fcf_q is None else fcf_q.dropna()
    if fcf_q is not None and len(fcf_q) >= 4:
        fcf0, fcf_src = float(fcf_q.iloc[:4].sum()), "trailing 12 months (last 4 quarters)"
    elif fcf_hist:
        fcf0, fcf_src = float(list(fcf_hist.values())[0]), f"fiscal year {list(fcf_hist)[0]}"
    else:
        fcf0, fcf_src = None, None
    shares_row = _row(bs, "Ordinary Shares Number", "Share Issued")
    shares = _num(info.get("sharesOutstanding")) or (float(shares_row.dropna().iloc[0]) if shares_row is not None else None)
    cash = _num(info.get("totalCash"))
    if cash is None:
        c = _row(bs, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
        cash = float(c.dropna().iloc[0]) if c is not None else 0.0
    debt = _num(info.get("totalDebt"))
    if debt is None:
        d = _row(bs, "Total Debt")
        debt = float(d.dropna().iloc[0]) if d is not None else 0.0
    beta = _num(info.get("beta")) or 1.0
    r = _clamp(rf + beta * ERP, R_MIN, R_MAX)

    # growth inputs: analysts' long-term or next-year estimate, else FCF history
    ge = {}
    try:
        g = t.growth_estimates
        if g is not None and "stockTrend" in g.columns:
            ge = {k: _num(g.loc[k, "stockTrend"]) for k in ("0y", "+1y", "+5y", "LTG") if k in g.index}
    except Exception:
        pass
    analyst = ge.get("LTG") or ge.get("+5y") or ge.get("+1y")
    vals = [v for v in fcf_hist.values()][::-1]            # oldest -> newest
    cagr = None
    if len(vals) >= 3 and vals[0] > 0 and vals[-1] > 0:
        cagr = (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1
    if analyst is not None:
        base_g, g_src = analyst, ("analysts' long-term growth" if (ge.get("LTG") or ge.get("+5y")) else "analysts' next-year growth")
    elif cagr is not None:
        base_g, g_src = cagr, f"{len(vals) - 1}-year FCF growth"
    else:
        base_g, g_src = 0.08, "default 8% (no estimate available)"
    raw_g = base_g
    base_g = _clamp(base_g, G_MIN, G_BASE_MAX)
    if raw_g != base_g and not mixed:
        out["flags"].append(f"Growth input ({g_src}) of {raw_g * 100:.0f}% was capped at {base_g * 100:.0f}% for 5 years; "
                            "very high growth rarely lasts that long.")

    # multiples at each fiscal year end vs today
    eps = _row(inc, "Diluted EPS")
    sh_hist = shares_row
    mult = []
    for col in (fcf_a.index if fcf_a is not None and not mixed else []):
        p = hist[hist.index <= pd.Timestamp(col)]
        if p.empty:
            continue
        px = float(p.iloc[-1])
        e = _num(eps.get(col)) if eps is not None else None
        s_ = _num(sh_hist.get(col)) if sh_hist is not None else None
        f_ = _num(fcf_a.get(col))
        mult.append({"fiscal_year_end": str(col.date()), "price": round(px, 2),
                     "pe": round(px / e, 1) if e and e > 0 else None,
                     "p_fcf": round(px / (f_ / s_), 1) if f_ and s_ and f_ > 0 else None})
    out["multiples"] = {"history": mult,
                        "pe_now": _num(info.get("trailingPE")), "forward_pe": _num(info.get("forwardPE")),
                        "p_fcf_now": round(price / (fcf0 / shares), 1) if fcf0 and shares and fcf0 > 0 and not mixed else None}
    tp, fp = out["multiples"]["pe_now"], out["multiples"]["forward_pe"]
    if tp and fp and (fp < tp * 0.4 or fp > tp * 1.5):
        out["flags"].append(f"Trailing P/E {tp:.1f} vs forward P/E {fp:.1f} differ a lot; check the forward estimate.")

    out["inputs"] = {"fcf": fcf0, "fcf_source": fcf_src, "fcf_history": fcf_hist, "shares": shares,
                     "cash": cash, "debt": debt, "beta": round(beta, 2), "risk_free": round(rf, 4), "erp": ERP,
                     "discount_rate": round(r, 4), "terminal_growth": TERMINAL,
                     "growth_base": round(base_g, 4), "growth_source": g_src,
                     "analyst_growth": ge, "fcf_cagr": None if cagr is None else round(cagr, 4)}
    if fcf0 is None or not shares:
        out["flags"].append("Free cash flow or share count is missing, so no DCF.")
        return out
    if fcf0 <= 0:
        out["flags"].append("Trailing free cash flow is negative; a cash-flow DCF is not meaningful. Use the multiples.")
        return out
    if mixed:
        return out
    if len(vals) >= 2 and vals[-2] > 0 and vals[-1] / vals[-2] > 3:
        out["flags"].append("Free cash flow jumped more than 3x in the last year; the base may not be repeatable.")

    sc = {}
    for name, g1 in (("bear", max(base_g - 0.10, G_MIN)), ("base", base_g), ("bull", min(base_g + 0.10, G_BULL_MAX))):
        g2 = (g1 + TERMINAL) / 2
        v = dcf_value(fcf0, g1, g2, r, TERMINAL, cash, debt, shares)
        sc[name] = {"g1": round(g1, 4), "g2": round(g2, 4), "value": round(v, 2),
                    "vs_price_pct": round((v / price - 1) * 100, 1)}
    out["scenarios"] = sc
    ig = implied_growth(price, fcf0, r, TERMINAL, cash, debt, shares)
    out["implied_growth"] = None if ig is None else round(ig, 4)
    return out


def summary(v):
    L = [f"{v['symbol']} {v['name']} — price {v['price']} {v['currency'] or ''}, 10Y {v['ten_year'] * 100:.2f}%"]
    if "etf" in v:
        e = v["etf"]
        if e.get("earnings_yield") is not None:
            L.append(f"  ETF P/E {e['pe']}, earnings yield {e['earnings_yield'] * 100:.2f}% vs 10Y "
                     f"{v['ten_year'] * 100:.2f}% → equity risk premium {e['equity_risk_premium'] * 100:+.2f}pp")
    if v.get("scenarios"):
        i = v["inputs"]
        L.append(f"  DCF: FCF {i['fcf'] / 1e9:.1f}B ({i['fcf_source']}), discount {i['discount_rate'] * 100:.1f}% "
                 f"(beta {i['beta']}), terminal {i['terminal_growth'] * 100:.0f}%, base growth {i['growth_base'] * 100:.0f}% ({i['growth_source']})")
        for k, s in v["scenarios"].items():
            L.append(f"  {k:<4} g1 {s['g1'] * 100:>5.1f}%  value {s['value']:>9.2f}  ({s['vs_price_pct']:+.1f}% vs price)")
        if v.get("implied_growth") is not None:
            L.append(f"  reverse DCF: today's price implies ~{v['implied_growth'] * 100:.1f}%/yr FCF growth for 5 years, then about half that")
    m = v.get("multiples")
    if m:
        L.append(f"  P/E now {m['pe_now']}, forward {m['forward_pe']}, P/FCF now {m['p_fcf_now']}; history: "
                 + ", ".join(f"{h['fiscal_year_end'][:4]} P/E {h['pe']} P/FCF {h['p_fcf']}" for h in m["history"]))
    for f in v["flags"]:
        L.append(f"  ! {f}")
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for s in sys.argv[1:] or ["NVDA"]:
        try:
            print(summary(valuation(s)))
        except Exception as e:
            print(f"{s}: ERROR {e}")
