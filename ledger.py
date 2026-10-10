"""
Hit ledger — freezes the Market Breakdown value lists every day and tracks how
they did afterwards, so the lists can be judged on results.

Each daily pick (date, list, symbol, score, close) is appended to
ledger/value_lists.jsonl once per date and list (forward only; history is never
rewritten). Evaluation compares each pick's return with SPY's over the same
window, 5 / 21 / 63 trading days after the pick date (and to date for younger
picks). A "hit" is a pick that beat SPY.
"""

import os, json
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(ROOT, "ledger", "value_lists.jsonl")
SUMMARY = os.path.join(ROOT, "ledger", "summary.json")
HORIZONS = {"1 week": 5, "1 month": 21, "3 months": 63}
LIST_NAMES = {"rebound": "High-Beta Rebound Candidates", "quality": "Quality Growth Below Value"}


def _entries():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]


def record(picks, day=None):
    """picks: {"rebound": [{"symbol", "score", "close"}], "quality": [...]}"""
    day = day or str(date.today())
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    have = {(e["date"], e["list"]) for e in _entries()}
    n = 0
    with open(LEDGER, "a", encoding="utf-8") as f:
        for lst, items in picks.items():
            if (day, lst) in have:
                continue
            for rank, it in enumerate(items, 1):
                f.write(json.dumps({"date": day, "list": lst, "rank": rank, "symbol": it["symbol"],
                                    "score": it["score"], "close": round(float(it["close"]), 4)}) + "\n")
                n += 1
    return n


def evaluate():
    es = _entries()
    if not es:
        out = {"lists": {}, "recent": [], "first_date": None}
        _save(out)
        return out
    first = min(e["date"] for e in es)
    syms = sorted({e["symbol"] for e in es} | {"SPY"})
    px = yf.download(syms, start=str(date.fromisoformat(first) - timedelta(days=7)), auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])
    px.index = px.index.tz_localize(None) if px.index.tz is not None else px.index
    px = px.ffill()
    rows = []
    for e in es:
        if e["symbol"] not in px.columns:
            continue
        pos = px.index.searchsorted(pd.Timestamp(e["date"]))          # first trading day on/after the pick date
        if pos >= len(px):
            continue
        p0, s0 = px[e["symbol"]].iloc[pos], px["SPY"].iloc[pos]
        if np.isnan(p0) or np.isnan(s0):
            continue
        r = dict(e)
        for name, h in HORIZONS.items():
            if pos + h < len(px):
                ret = px[e["symbol"]].iloc[pos + h] / p0 - 1
                spy = px["SPY"].iloc[pos + h] / s0 - 1
                r[name] = {"ret": round(float(ret) * 100, 2), "excess": round(float(ret - spy) * 100, 2)}
        ret, spy = px[e["symbol"]].iloc[-1] / p0 - 1, px["SPY"].iloc[-1] / s0 - 1
        r["to_date"] = {"ret": round(float(ret) * 100, 2), "excess": round(float(ret - spy) * 100, 2), "days": int(len(px) - 1 - pos)}
        rows.append(r)
    lists = {}
    for lst in LIST_NAMES:
        rs = [r for r in rows if r["list"] == lst]
        stats = {}
        for name in list(HORIZONS) + ["to_date"]:
            xs = [r[name]["excess"] for r in rs if name in r]
            if xs:
                stats[name] = {"n": len(xs), "hit_rate": round(100 * sum(x > 0 for x in xs) / len(xs)),
                               "avg_excess": round(float(np.mean(xs)), 2), "median_excess": round(float(np.median(xs)), 2)}
        lists[lst] = {"name": LIST_NAMES[lst], "picks": len(rs), "days": len({r["date"] for r in rs}), "stats": stats}
    recent = sorted(rows, key=lambda r: (r["date"], r["list"], r["rank"]), reverse=True)[:40]
    out = {"first_date": first, "asof": str(px.index[-1].date()), "lists": lists, "recent": recent}
    _save(out)
    return out


def _save(out):
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    json.dump(out, open(SUMMARY, "w", encoding="utf-8"), indent=1)


def render(s):
    """HTML block for the Market Breakdown tab (scanner page styles)."""
    if not s or not s.get("lists"):
        return ("<div class='bd-note'><b>Track record</b>: the lists above are frozen in a ledger from today; "
                "results vs SPY appear here after the first week.</div>")
    cols = list(HORIZONS) + ["to_date"]
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    def cell(st):
        if not st:
            return "<td style='color:var(--muted)'>—</td>"
        col = "var(--green)" if st["avg_excess"] > 0 else "var(--red)"
        return (f"<td><b style='color:{col}'>{st['avg_excess']:+.1f}%</b> avg vs SPY<br>"
                f"<span style='color:var(--muted)'>beat SPY {st['hit_rate']}% of {st['n']}</span></td>")
    body = "".join(f"<tr><td><b>{v['name']}</b><br><span style='color:var(--muted)'>{v['picks']} picks over {v['days']} days</span></td>"
                   + "".join(cell(v["stats"].get(c)) for c in cols) + "</tr>" for v in s["lists"].values())
    rec = "".join(f"<tr><td>{r['date']}</td><td>{'Rebound' if r['list'] == 'rebound' else 'Quality'}</td><td><b>{r['symbol']}</b></td>"
                  f"<td>{r['score']}</td><td>{r['close']:.2f}</td>"
                  f"<td style='color:{'var(--green)' if r['to_date']['excess'] > 0 else 'var(--red)'}'>{r['to_date']['ret']:+.1f}% "
                  f"({r['to_date']['excess']:+.1f}% vs SPY, {r['to_date']['days']}d)</td></tr>" for r in s["recent"][:20])
    return f"""<div class="bd-section"><div class="bd-sec-hdr" style="border-bottom-color:#94a3b8"><div class="bd-icon" style="background:#94a3b822">📒</div>
  <div><div class="bd-sec-title">Track record (hit ledger)</div></div>
  <div class="bd-sec-sub">Lists frozen daily since {s['first_date']}; returns vs SPY after each pick · as of {s['asof']}</div></div>
  <div style="overflow-x:auto"><table class="bd-ledger"><tr><th>List</th>{head}</tr>{body}</table></div>
  <div style="overflow-x:auto;margin-top:10px"><table class="bd-ledger"><tr><th>Picked</th><th>List</th><th>Symbol</th><th>Score</th><th>Price then</th><th>Since then</th></tr>{rec}</table></div>
  <p class="bd-note">Forward-only: each day's lists are appended once and never edited. "Beat SPY" = the pick's price return exceeded SPY's over the same window. Past results do not predict future ones; not investment advice.</p></div>"""
