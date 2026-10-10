"""
Sector rotation map (RRG-style) — which sectors and themes are leading or lagging
the S&P 500, and whether that is improving or fading.

For each ETF, using price returns (dividends excluded) relative to SPY:
    relative strength  RS = 0.5 x 1-month + 0.3 x 3-month + 0.2 x 1-week excess return
    relative momentum  RM = change of the 1-month excess return over the last 21 days
Quadrants: Leading (RS > 0, RM > 0), Weakening (RS > 0, RM < 0),
Lagging (RS < 0, RM < 0), Improving (RS < 0, RM > 0). Rotation usually runs
clockwise: Improving -> Leading -> Weakening -> Lagging -> Improving.
"Early turn" = moved from Lagging to Improving within the last two weeks.

Usage: python rotation.py
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf

ETFS = [  # symbol, label, group
    ("XLK", "Technology", "sector"), ("XLC", "Communication services", "sector"),
    ("XLY", "Consumer discretionary", "sector"), ("XLP", "Consumer staples", "sector"),
    ("XLE", "Energy", "sector"), ("XLF", "Financials", "sector"), ("XLV", "Health care", "sector"),
    ("XLI", "Industrials", "sector"), ("XLB", "Materials", "sector"), ("XLRE", "Real estate", "sector"),
    ("XLU", "Utilities", "sector"),
    ("SMH", "Semiconductors", "theme"), ("IGV", "Software", "theme"), ("HACK", "Cybersecurity", "theme"),
    ("SKYY", "Cloud computing", "theme"), ("KRE", "Regional banks", "theme"), ("ITB", "Homebuilders", "theme"),
    ("IYT", "Transportation", "theme"), ("XBI", "Biotech", "theme"), ("XOP", "Oil & gas producers", "theme"),
    ("ITA", "Aerospace & defense", "theme"), ("URA", "Uranium & nuclear", "theme"), ("TAN", "Solar", "theme"),
    ("ARKK", "Innovation (ARKK)", "theme"), ("IWM", "Small caps", "theme"),
    ("EEM", "Emerging markets", "macro"), ("KWEB", "China internet", "macro"), ("GLD", "Gold", "macro"),
    ("TLT", "Long Treasuries", "macro"), ("UUP", "US dollar", "macro"),
]
TAIL_WEEKS = 6


def _quadrant(rs, rm):
    if rs >= 0:
        return "Leading" if rm >= 0 else "Weakening"
    return "Improving" if rm >= 0 else "Lagging"


def rrg():
    syms = [e[0] for e in ETFS] + ["SPY"]
    px = yf.download(syms, period="1y", interval="1d", auto_adjust=False, progress=False)["Close"]
    px = px.dropna(how="all").ffill()
    spy = px["SPY"]
    ex = lambda s, n: ((s / s.shift(n) - 1) - (spy / spy.shift(n) - 1)) * 100
    out, errors = [], {}
    for sym, label, group in ETFS:
        try:
            c = px[sym].dropna()
            if len(c) < 90:
                raise ValueError("not enough history")
            e1w, e1m, e3m = ex(c, 5), ex(c, 21), ex(c, 63)
            rs = 0.5 * e1m + 0.3 * e3m + 0.2 * e1w
            rm = e1m - e1m.shift(21)
            pts = [(-1 - 5 * k) for k in range(TAIL_WEEKS)][::-1]       # weekly points, oldest first
            tail = [{"date": str(rs.index[i].date()), "rs": round(float(rs.iloc[i]), 2), "rm": round(float(rm.iloc[i]), 2)}
                    for i in pts if not (np.isnan(rs.iloc[i]) or np.isnan(rm.iloc[i]))]
            q_now = _quadrant(tail[-1]["rs"], tail[-1]["rm"])
            q_1w, q_2w = (_quadrant(tail[-2]["rs"], tail[-2]["rm"]) if len(tail) > 1 else q_now,
                          _quadrant(tail[-3]["rs"], tail[-3]["rm"]) if len(tail) > 2 else q_now)
            out.append({"symbol": sym, "label": label, "group": group, "rs": tail[-1]["rs"], "rm": tail[-1]["rm"],
                        "quadrant": q_now, "quadrant_1w": q_1w, "quadrant_2w": q_2w,
                        "early_turn": q_now == "Improving" and "Lagging" in (q_1w, q_2w),
                        "ex_1w": round(float(e1w.iloc[-1]), 2), "ex_1m": round(float(e1m.iloc[-1]), 2), "ex_3m": round(float(e3m.iloc[-1]), 2),
                        "price": round(float(c.iloc[-1]), 2), "tail": tail})
        except Exception as e:
            errors[sym] = str(e)
    out.sort(key=lambda x: -x["rs"])
    return {"asof": str(px.index[-1].date()), "etfs": out, "errors": errors,
            "spy": {"r1w": round(float((spy.iloc[-1] / spy.iloc[-6] - 1) * 100), 2),
                    "r1m": round(float((spy.iloc[-1] / spy.iloc[-22] - 1) * 100), 2),
                    "r3m": round(float((spy.iloc[-1] / spy.iloc[-64] - 1) * 100), 2)}}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    r = rrg()
    print(f"RRG as of {r['asof']}, SPY 1w {r['spy']['r1w']:+}% 1m {r['spy']['r1m']:+}% 3m {r['spy']['r3m']:+}%; errors {r['errors']}")
    for x in r["etfs"]:
        flag = "  EARLY TURN" if x["early_turn"] else ("" if x["quadrant"] == x["quadrant_1w"] else f"  (was {x['quadrant_1w']} a week ago)")
        print(f"  {x['symbol']:<5} {x['label']:<24} {x['quadrant']:<10} RS {x['rs']:+6.1f}  RM {x['rm']:+6.1f}{flag}")
