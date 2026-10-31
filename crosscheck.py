"""
Daily second opinion from the local Qwen model (LM Studio, on this PC's GPU).

The closing brief runs this once a day (THE data output says QWEN_CROSSCHECK_DUE=yes):
  1. builds a pack of the day's analysis: the latest brief, the verdicts and trends, the checklist
     levels, sentiment & leadership, the bear market watch, the political risk register and the
     6-12 month outlook;
  2. starts LM Studio's server and loads the model if needed (and unloads it afterwards, so the GPU
     is free again);
  3. asks for an independent read (agree / disagree per ticker, inconsistencies, what is missing,
     the key level to watch) and writes it to briefs/qwen/<brief id>.md.
The routine then checks each point against the data and adds a verified "Second opinion" section
to the brief. The model is weaker than the routine and can be confidently wrong: never publish its
claims unchecked.

    python crosscheck.py            # uses the newest brief
    python crosscheck.py --pack     # only print the pack
"""

import os, sys, json, re, time, subprocess, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
LMS = os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms.exe")
MODEL = "qwen/qwen3.8-27b"
URL = "http://localhost:1234/v1"
SYSTEM = ("You are a skeptical senior market strategist giving an independent second opinion on another analyst's "
          "market analysis. Use only the data provided; say 'not in the data' rather than inventing numbers. Be specific and concise.")
ASK = """Below is today's full analysis from a trading site (brief, verdicts, levels, sentiment, bear-market watch, political risk register, 6-12 month outlook). Give an independent second opinion in this structure:
1. Overall read (3-4 sentences): do you agree with the site's overall stance? Why or why not?
2. Per ticker: agree or disagree with each verdict, and why, citing the data.
3. Inconsistencies: places where parts of the analysis contradict each other (verdict vs levels vs bear watch vs sentiment vs outlook), quoting both numbers.
4. What may be missing or underweighted (risks, signals, scenarios).
5. The single most important level or event to watch next, and why.
Keep it under 900 words.
"""


def _j(path):
    try:
        return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    except Exception:
        return None


def pack():
    L = []
    ix = (_j("briefs/index.json") or {}).get("briefs", [])
    if not ix:
        raise SystemExit("no brief yet")
    bid = ix[0]["id"]
    L.append(f"=== LATEST MARKET BRIEF {bid} ===")
    L.append(open(os.path.join(ROOT, "briefs", bid + ".md"), encoding="utf-8").read())
    v = _j("briefs/verdicts.json") or {}
    L.append("\n=== VERDICTS (brief, combined with the technical checklist) ===")
    for s, x in (v.get("tickers") or {}).items():
        L.append(f"{s}: {x.get('verdict')} (technical {x.get('technical')}, confidence {x.get('confidence')}) — {x.get('summary')} | "
                 f"risks to the verdict: {'; '.join(x.get('risks', []))} | levels {x.get('levels')} | changes if: {x.get('changes_if')}")
    L.append("\n=== CURRENT TRENDS ===")
    for t in v.get("trends", []):
        L.append(f"- {t.get('area')}: {t.get('now')} -> {t.get('implication')}")
    b = _j(f"briefs/{bid}.json") or {}
    L.append("\n=== CHECKLIST LEVELS ===")
    for t in b.get("tickers", []):
        a = t.get("assessment") or {}
        if "levels" in a:
            L.append(f"{t['symbol']} {a['verdict']} {a['levels']} trailing stop {a.get('trailing_stop')} sell alerts {a.get('sell_alerts')}")
    try:
        import sentiment, bearwatch
        d = _j("context/sentiment.json")
        if d:
            L.append("\n=== " + sentiment.summary(d))
        bw = _j("context/bear.json")
        if bw:
            L.append("\n=== " + bearwatch.summary(bw))
    except Exception as e:
        L.append(f"(sentiment / bear watch unavailable: {e})")
    r = _j("context/risks.json")
    if r:
        L.append(f"\n=== POLITICAL & GEOPOLITICAL RISK REGISTER {r.get('updated')} ===\n{r.get('summary', '')}")
        for x in r.get("risks", []):
            L.append(f"- {x.get('title')} [{x.get('region')}] status: {x.get('status')} | next {x.get('next_date')} | odds {x.get('odds')} | trend {x.get('trend')} | impact {x.get('impact')}")
    th = _j("thesis/state.json")
    if th and th.get("outlook"):
        o = th["outlook"]
        L.append(f"\n=== 6-12 MONTH OUTLOOK {o.get('id')} ===\n{o.get('regime')}\n{o.get('summary')}\nThemes: " + ", ".join(t["name"] for t in o.get("themes", [])))
    return bid, "\n".join(L)


def _lms(*args, timeout=600):
    return subprocess.run([LMS, *args], capture_output=True, text=True, timeout=timeout)


def _loaded():
    try:
        with urllib.request.urlopen(URL + "/models", timeout=5) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        ps = _lms("ps", timeout=60).stdout
        return MODEL in ps
    except Exception:
        return False


def ask(text):
    body = {"model": MODEL, "max_tokens": 8000, "temperature": 0.4,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK + "\n" + text + "\n\n/no_think"}]}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.load(r)
    msg = res["choices"][0]["message"]
    return re.sub(r"<think>[\s\S]*?</think>", "", msg.get("content") or "").strip(), res.get("usage", {})


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bid, text = pack()
    if "--pack" in sys.argv:
        print(text)
        return
    if not os.path.exists(LMS):
        print(f"QWEN: LM Studio not found at {LMS}; skipped")
        return
    t0 = time.time()
    _lms("server", "start", timeout=120)
    loaded_here = False
    if not _loaded():
        out = _lms("load", MODEL, "--context-length", "32768", "-y", timeout=900)
        if out.returncode != 0:
            print("QWEN: could not load the model; skipped\n" + (out.stdout + out.stderr)[-500:])
            return
        loaded_here = True
    try:
        answer, usage = ask(text)
    finally:
        if loaded_here:
            _lms("unload", "--all", timeout=120)          # free the GPU again
    if not answer:
        print("QWEN: empty answer (the model spent its budget reasoning); skipped")
        return
    os.makedirs(os.path.join(ROOT, "briefs", "qwen"), exist_ok=True)
    path = os.path.join(ROOT, "briefs", "qwen", bid + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Local model second opinion on brief {bid}\n\n_Qwen ({MODEL}) on this PC, {time.time() - t0:.0f}s, "
                f"{usage.get('prompt_tokens', '?')} tokens in. Unverified model output._\n\n{answer}\n")
    print(f"QWEN_OPINION=briefs/qwen/{bid}.md")
    print(answer)


if __name__ == "__main__":
    main()
