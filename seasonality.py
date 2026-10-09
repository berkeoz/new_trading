"""
Sector seasonality — how each sector usually moves through the calendar year,
and how it is doing this year.

For each sector ETF (and SPY as the reference):
    * average path: for each of the last full years (up to 15), the year-to-date
      return by day of the year, averaged across years;
    * this year: the actual year-to-date return so far;
    * month table: average return and share of years that were positive for each
      calendar month.

Used at the bottom of the formations page. Usage: python seasonality.py
"""

import sys
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf

SECTORS = [  # symbol, label
    ("SPY", "S&P 500 (reference)"),
    ("XLK", "Technology"), ("SMH", "Semiconductors"), ("XLC", "Communication services (telecom, media)"),
    ("XLY", "Consumer discretionary"), ("XLP", "Consumer staples"), ("XLF", "Financials"),
    ("KRE", "Regional banks"), ("XLV", "Health care"), ("XLI", "Industrials"), ("IYT", "Transportation"),
    ("XLE", "Energy"), ("XLB", "Materials"), ("XLRE", "Real estate"), ("XLU", "Utilities"),
    ("ITB", "Homebuilders"), ("IWM", "Small caps (Russell 2000)"),
]
YEARS = 15
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _ytd_by_doy(s, year):
    """Year-to-date return (%) indexed 1..366 by day of year, forward-filled."""
    prev = s[s.index.year == year - 1]
    cur = s[s.index.year == year]
    if prev.empty or cur.empty:
        return None
    r = (cur / float(prev.iloc[-1]) - 1) * 100
    r.index = r.index.dayofyear
    r = r.groupby(level=0).last()
    return r.reindex(range(1, 367)).ffill()


def sector_seasonality(sym, label, this_year):
    h = yf.Ticker(sym).history(period="max", interval="1d", auto_adjust=True)["Close"].dropna()
    if h.empty:
        raise ValueError("no data")
    h.index = h.index.tz_localize(None)
    years = [y for y in range(this_year - YEARS, this_year) if (h.index.year == y - 1).any() and (h.index.year == y).any()
             and h[h.index.year == y].index[-1].month == 12]
    paths = [p for p in (_ytd_by_doy(h, y) for y in years) if p is not None]
    if len(paths) < 3:
        raise ValueError("fewer than 3 full years of history")
    avg = pd.concat(paths, axis=1).mean(axis=1)
    cur = _ytd_by_doy(h, this_year)
    today_doy = h.index[-1].dayofyear
    if cur is not None:
        cur = cur.loc[:today_doy]
    m = h.resample("ME").last().pct_change() * 100
    m = m[(m.index.year >= this_year - YEARS) & (m.index.year < this_year)].dropna()
    month_avg = [round(float(m[m.index.month == k].mean()), 2) for k in range(1, 13)]
    month_up = [round(float((m[m.index.month == k] > 0).mean() * 100), 0) for k in range(1, 13)]
    cur_month = h[(h.index.year == this_year) & (h.index.month == h.index[-1].month)]
    prev_close = h[h.index < cur_month.index[0]].iloc[-1] if not cur_month.empty else None
    return {
        "symbol": sym, "label": label, "years": len(paths), "first_year": years[0] if years else None,
        "avg": [round(float(x), 2) for x in avg.values],
        "cur": [] if cur is None else [None if np.isnan(x) else round(float(x), 2) for x in cur.values],
        "ytd": None if cur is None or cur.empty else round(float(cur.iloc[-1]), 2),
        "norm_today": round(float(avg.loc[today_doy]), 2),
        "month_avg": month_avg, "month_up": month_up,
        "month_so_far": None if prev_close is None else round(float((cur_month.iloc[-1] / prev_close - 1) * 100), 2),
    }


def seasonality():
    this_year = date.today().year
    out, errors = [], {}
    for sym, label in SECTORS:
        try:
            out.append(sector_seasonality(sym, label, this_year))
        except Exception as e:
            errors[sym] = str(e)
    d0 = pd.Timestamp(this_year, 1, 1)
    return {"year": this_year, "x": [str((d0 + pd.Timedelta(days=k)).date()) for k in range(366)],
            "today": str(date.today()), "month": date.today().month, "sectors": out, "errors": errors}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    s = seasonality()
    mi = s["month"] - 1
    print(f"Seasonality {s['year']}, month {MONTHS[mi]}; errors: {s['errors']}")
    for x in sorted(s["sectors"], key=lambda x: -x["month_avg"][mi]):
        print(f"  {x['symbol']:<5} {x['label'][:28]:<28} {x['years']}y  {MONTHS[mi]} avg {x['month_avg'][mi]:+.2f}% "
              f"(up {x['month_up'][mi]:.0f}%)  next {x['month_avg'][(mi + 1) % 12]:+.2f}%  "
              f"YTD {x['ytd']:+.1f}% vs norm {x['norm_today']:+.1f}%  month so far {x['month_so_far']:+.2f}%")
