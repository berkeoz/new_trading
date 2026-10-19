"""
6-12 month outlook portfolio — a paper $100,000 portfolio of companies expected to benefit
from the economy 6-12 months out, shown on the Market Breakdown tab.

Who does what:
  * Weekly (the Friday closing brief, or whenever THESIS_REVIEW_DUE=yes), the market-brief
    routine writes thesis/outlook.json: the expected regime, the themes, and candidates, each
    with an entry plan (buy zone or buy now), a stop, a target and a conviction (1-3).
    `python thesis.py --apply` then merges it: new candidates join the watch list, removed
    ones are sold or dropped, and held positions get the new stop / target.
  * Every brief run (briefing.py calls run()), the rules act on current prices, during
    regular hours only (9:30-16:30 ET on weekdays, so the 16:15 run trades at the close):
      - a watch-list candidate is bought when the price is inside its buy zone (or at once
        for "now"), sized 5 / 8 / 12% of equity for conviction 1 / 2 / 3, capped by cash;
      - it is dropped if the price falls below its stop before the entry;
      - a position is sold when the price is at or below its stop or at or above its target.
    The run also marks the portfolio and SPY once per day for the equity curve.
  * `python thesis.py --inspect SYM ...` prints the data the routine uses to set entry zones,
    stops and targets (levels, moving averages, RSI, ATR, templates, valuation).

Fills are at the last price seen by the run (Yahoo, can lag ~15 min); no commissions or
slippage. A model, not investment advice.

Files in thesis/ (committed by the routine; the page reads them from GitHub):
  outlook.json        current outlook, themes and candidates (written by the routine)
  outlook-<date>.md   the narrative for that week (written by the routine)
  state.json          cash, positions, watch list, closed trades, equity curve, events
"""

import os, sys, json, math, argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "thesis")
OUTLOOK = os.path.join(TDIR, "outlook.json")
STATE = os.path.join(TDIR, "state.json")
ET = ZoneInfo("America/New_York")
START_CASH = 100_000.0
SIZE = {1: 0.05, 2: 0.08, 3: 0.12}          # share of equity per new position by conviction
REVIEW_DAYS = 7                              # weekly review; due on Friday's closing run or if older


# ── state ─────────────────────────────────────────────────────────────────────
def _now():
    return datetime.now(ET)


def load():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {"start": None, "cash": START_CASH, "positions": {}, "watch": {}, "closed": [],
                "equity": [], "events": [], "outlook_id": None, "outlook_applied": None}


def save(st):
    os.makedirs(TDIR, exist_ok=True)
    st["events"] = st["events"][-200:]
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)


def _event(st, kind, sym, text, now):
    st["events"].append({"time": f"{now:%Y-%m-%d %H:%M}", "kind": kind, "symbol": sym, "text": text})


def _prices(syms):
    """Latest regular-hours price per symbol (5-minute bars of the last few days)."""
    syms = sorted(set(syms))
    if not syms:
        return {}
    df = yf.download(syms, period="5d", interval="5m", auto_adjust=True, progress=False, prepost=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(syms[0])
    last = df.ffill().iloc[-1]
    return {s: float(last[s]) for s in syms if s in last and not math.isnan(last[s])}


def equity(st, px):
    return st["cash"] + sum(p["shares"] * px.get(s, p.get("last", p["entry_price"])) for s, p in st["positions"].items())


def trading_hours(now):
    return now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and (now.hour, now.minute) <= (16, 30)


# ── the weekly outlook ───────────────────────────────────────────────────────
def review_due(st, now, session=None):
    """(yes/no, reason). Due when there is no outlook, on the Friday closing run if this
    week's outlook is not done yet, or when the outlook is older than REVIEW_DAYS + 1."""
    if not os.path.exists(OUTLOOK) or not st.get("outlook_applied"):
        return True, "no 6-12 month outlook yet"
    applied = datetime.fromisoformat(st["outlook_applied"]).replace(tzinfo=ET)
    age = (now - applied).days
    if session == "close" and now.weekday() == 4 and age >= 3:
        return True, "weekly review (Friday close)"
    if age > REVIEW_DAYS + 1:
        return True, f"outlook is {age} days old"
    return False, f"outlook {st.get('outlook_id')} applied {applied:%Y-%m-%d} ({age} days ago); next review Friday's closing brief"


def apply_outlook(now=None):
    """Merge thesis/outlook.json into the state."""
    now = now or _now()
    st = load()
    o = json.load(open(OUTLOOK, encoding="utf-8"))
    cands = {c["symbol"].upper(): c for c in o.get("candidates", [])}
    removes = {r["symbol"].upper(): r.get("reason", "") for r in o.get("remove", [])}
    bad = [s for s, c in cands.items() if not _valid(c)]
    if bad:
        raise SystemExit(f"outlook.json: candidates with missing/invalid fields: {', '.join(bad)} "
                         "(need symbol, theme, thesis, conviction 1-3, entry {type: zone low high | now}, stop, target)")
    px = _prices(list(cands) + list(removes) + list(st["positions"]) + ["SPY"]) if trading_hours(now) else {}
    if st["start"] is None:
        st["start"] = f"{now:%Y-%m-%d}"
    for s, why in removes.items():
        if s in st["positions"]:
            _sell(st, s, px.get(s), f"thesis removed: {why}", now) if trading_hours(now) else \
                st["positions"][s].update({"exit_pending": f"thesis removed: {why}"})
        if s in st["watch"]:
            del st["watch"][s]
            _event(st, "drop", s, f"removed from the watch list: {why}", now)
    for s, c in cands.items():
        keep = {k: c[k] for k in ("theme", "thesis", "conviction", "entry", "stop", "target", "invalidation", "catalysts") if k in c}
        if s in st["positions"]:
            st["positions"][s].update({k: keep[k] for k in ("theme", "thesis", "stop", "target", "invalidation", "catalysts") if k in keep})
        else:
            prev = st["watch"].get(s, {})
            st["watch"][s] = {**keep, "added": prev.get("added", f"{now:%Y-%m-%d}")}
            if not prev:
                _event(st, "watch", s, f"added to the watch list ({c['theme']}): {_entry_txt(c)}", now)
    for s in [s for s in st["watch"] if s not in cands]:
        del st["watch"][s]
        _event(st, "drop", s, "not in this week's outlook", now)
    st["outlook_id"], st["outlook_applied"] = o.get("id"), now.isoformat(timespec="minutes")
    st["outlook"] = {k: o.get(k) for k in ("id", "horizon", "regime", "summary", "drivers", "themes", "risks", "narrative_file")}
    save(st)
    print(f"outlook {o.get('id')} applied: {len(st['watch'])} watching, {len(st['positions'])} held")
    return run(now)


def _valid(c):
    try:
        e = c["entry"]
        ok = e["type"] == "now" or (e["type"] == "zone" and 0 < float(e["low"]) <= float(e["high"]))
        return ok and int(c["conviction"]) in SIZE and float(c["stop"]) > 0 and float(c["target"]) > 0 and c.get("theme") and c.get("thesis")
    except Exception:
        return False


def _entry_txt(c):
    e = c["entry"]
    z = "buy now" if e["type"] == "now" else f"buy zone {e['low']}-{e['high']}"
    return f"{z}, stop {c['stop']}, target {c['target']}, conviction {c['conviction']}"


# ── rules ──────────────────────────────────────────────────────────────────────
def _buy(st, s, c, price, eq, spy, now):
    budget = min(eq * SIZE[int(c["conviction"])], st["cash"])
    sh = math.floor(budget / price)
    if sh <= 0:
        _event(st, "skip", s, f"in its buy zone at {price:.2f} but not enough cash", now)
        return
    st["cash"] -= sh * price
    st["positions"][s] = {"shares": sh, "entry_price": round(price, 4), "entry_date": f"{now:%Y-%m-%d %H:%M}", "spy_entry": spy,
                          "last": round(price, 4), **{k: c[k] for k in ("theme", "thesis", "conviction", "stop", "target", "invalidation", "catalysts", "entry") if k in c}}
    del st["watch"][s]
    _event(st, "buy", s, f"bought {sh} at {price:.2f} ({sh * price:,.0f}, {SIZE[int(c['conviction'])]:.0%} target weight)", now)


def _sell(st, s, price, why, now):
    p = st["positions"].pop(s)
    price = price or p.get("last") or p["entry_price"]
    st["cash"] += p["shares"] * price
    pnl = p["shares"] * (price - p["entry_price"])
    st["closed"].append({"symbol": s, "theme": p.get("theme"), "shares": p["shares"], "entry_price": p["entry_price"],
                         "entry_date": p["entry_date"], "exit_price": round(price, 4), "exit_date": f"{now:%Y-%m-%d %H:%M}",
                         "pnl": round(pnl, 2), "ret_pct": round((price / p["entry_price"] - 1) * 100, 2), "reason": why,
                         "spy_entry": p.get("spy_entry")})
    _event(st, "sell", s, f"sold {p['shares']} at {price:.2f} ({why}); P&L {pnl:+,.0f} ({(price / p['entry_price'] - 1) * 100:+.1f}%)", now)


def run(now=None, session=None):
    """Act on current prices and mark the portfolio. Returns a text summary for the brief."""
    now = now or _now()
    st = load()
    acted = []
    n0 = len(st["events"])
    px = {}
    if st["positions"] or st["watch"]:
        try:
            px = _prices(list(st["positions"]) + list(st["watch"]) + ["SPY"])
        except Exception as e:
            return f"THESIS PORTFOLIO: price download failed ({e})"
    spy = px.get("SPY")
    if trading_hours(now) and px:
        for s, p in list(st["positions"].items()):
            pr = px.get(s)
            if pr is None:
                continue
            p["last"] = round(pr, 4)
            if p.get("exit_pending"):
                _sell(st, s, pr, p["exit_pending"], now)
            elif pr <= float(p["stop"]):
                _sell(st, s, pr, f"stop {p['stop']} hit", now)
            elif pr >= float(p["target"]):
                _sell(st, s, pr, f"target {p['target']} reached", now)
        eq = equity(st, px)
        for s, c in sorted(st["watch"].items(), key=lambda kv: -int(kv[1]["conviction"])):
            pr = px.get(s)
            if pr is None:
                continue
            c["last"] = round(pr, 4)
            e = c["entry"]
            if pr <= float(c["stop"]):
                del st["watch"][s]
                _event(st, "drop", s, f"fell to {pr:.2f}, below its stop {c['stop']}, before an entry", now)
            elif e["type"] == "now" or float(e["low"]) <= pr <= float(e["high"]):
                _buy(st, s, c, pr, eq, spy, now)
    else:
        for s, p in st["positions"].items():
            if px.get(s):
                p["last"] = round(px[s], 4)
        for s, c in st["watch"].items():
            if px.get(s):
                c["last"] = round(px[s], 4)
    if st["start"] and spy:
        day = f"{now:%Y-%m-%d}"
        pt = {"date": day, "equity": round(equity(st, px), 2), "spy": round(spy, 4)}
        if st["equity"] and st["equity"][-1]["date"] == day:
            st["equity"][-1] = pt
        else:
            st["equity"].append(pt)
    st["updated"] = now.isoformat(timespec="minutes")
    st["spy_last"] = spy
    save(st)
    acted = st["events"][n0:]
    return summary(st, px, acted, now, session)


def summary(st, px, acted, now, session=None):
    due, why = review_due(st, now, session)
    eq = equity(st, px)
    L = ["THESIS PORTFOLIO (paper $100k, 6-12 month outlook; Market Breakdown tab)",
         f"  equity {eq:,.0f} ({(eq / START_CASH - 1) * 100:+.2f}%), cash {st['cash']:,.0f}, "
         f"{len(st['positions'])} positions, {len(st['watch'])} on the watch list" +
         ("" if trading_hours(now) else " · outside regular hours: no trades this run")]
    if st["equity"] and st["equity"][0].get("spy") and px.get("SPY"):
        L.append(f"  SPY since start {(px['SPY'] / st['equity'][0]['spy'] - 1) * 100:+.2f}%")
    for s, p in st["positions"].items():
        pr = px.get(s, p["last"])
        L.append(f"  HELD {s} ({p.get('theme')}): {p['shares']} @ {p['entry_price']:.2f}, last {pr:.2f} "
                 f"({(pr / p['entry_price'] - 1) * 100:+.1f}%), stop {p['stop']}, target {p['target']}")
    for s, c in st["watch"].items():
        e, pr = c["entry"], px.get(s, c.get("last"))
        z = "now" if e["type"] == "now" else f"{e['low']}-{e['high']}"
        L.append(f"  WATCH {s} ({c.get('theme')}): last {round(pr, 2) if pr else '?'}, buy zone {z}, stop {c['stop']}, target {c['target']}")
    for e in acted:
        L.append(f"  THIS RUN: {e['kind'].upper()} {e['symbol']}: {e['text']}")
    L.append(f"THESIS_REVIEW_DUE={'yes' if due else 'no'}")
    L.append(f"THESIS_REVIEW_REASON={why}")
    if due:
        L.append(f"WRITE_THESIS_TO=thesis/outlook.json and thesis/outlook-{now:%Y-%m-%d}.md")
    return "\n".join(L)


# ── data for choosing entries ─────────────────────────────────────────────────
def inspect(syms):
    import patterns as P
    try:
        import valuation as V
    except Exception:
        V = None
    out = []
    for s in syms:
        s = s.upper()
        try:
            df = P.load_prices(s, "2y", fill_today=True)
            c = df["Close"]
            r = P.analyze(s, "2y", fill_today=True)
            info = yf.Ticker(s).info or {}
            sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
            d = c.diff()
            g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            lo = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            rsi = 100 - 100 / (1 + g / lo) if lo else 100
            hi52, lo52 = df["High"].iloc[-252:].max(), df["Low"].iloc[-252:].min()
            t = r.get("templates") or {}
            line = [f"{s} — {info.get('shortName', '')} · {info.get('sector') or info.get('category') or ''} / {info.get('industry', '')}",
                    f"  price {c.iloc[-1]:.2f} · 52w {lo52:.2f}-{hi52:.2f} ({(c.iloc[-1] / hi52 - 1) * 100:+.1f}% from high) · "
                    f"SMA50 {sma50:.2f} · SMA200 {sma200:.2f} · RSI14 {rsi:.0f} · ATR {r['atr_pct']}%/day",
                    f"  mkt cap {info.get('marketCap', 0) / 1e9:,.1f}B · avg $ volume {info.get('averageVolume', 0) * c.iloc[-1] / 1e6:,.0f}M · "
                    f"fwd P/E {round(info['forwardPE'], 1) if info.get('forwardPE') else '—'} · analyst target {info.get('targetMeanPrice') or '—'}",
                    "  levels: " + ", ".join(f"{l['role']} {l['price']} x{l['touches']}" for l in r["levels"])]
            if t:
                line.append(f"  Minervini {t['minervini']['passed']}/{t['minervini']['of']} · Weinstein stage {t['weinstein']['stage']} ({t['weinstein']['text']})")
            if V and info.get("quoteType") == "EQUITY":
                try:
                    v = V.valuation(s)
                    sc = v.get("scenarios") or {}
                    if sc:
                        line.append("  DCF value bear/base/bull " + " / ".join(f"{sc[k]['value']:.0f}" for k in ("bear", "base", "bull") if k in sc) +
                                    (f" · price implies {v['implied_growth'] * 100:.0f}%/yr growth" if v.get("implied_growth") is not None else ""))
                except Exception:
                    pass
            out.append("\n".join(line))
        except Exception as e:
            out.append(f"{s}: ERROR {e}")
    return "\n".join(out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="merge thesis/outlook.json, then act on prices")
    ap.add_argument("--run", action="store_true", help="act on prices and mark the portfolio")
    ap.add_argument("--inspect", nargs="+", metavar="SYM", help="data for setting entry zones / stops / targets")
    a = ap.parse_args()
    if a.apply:
        print(apply_outlook())
    elif a.inspect:
        print(inspect(a.inspect))
    elif a.run:
        print(run())
    else:
        print(summary(load(), {}, [], _now()))
