"""
X (Twitter) digest helper for the daily "Twitter analysis" routine.

The routine reads each account's profile in the user's logged-in Chrome (Claude in
Chrome), so no X API is needed. This script keeps the bookkeeping:

    python xfeed.py --plan                 # start of run: accounts, last-read post per account,
                                           # the extraction snippet, output paths; records the run start
    python xfeed.py --seen HANDLE POST_ID  # after reading an account: remember its newest post
    python xfeed.py --index                # rebuild xfeed/index.json for the page
    python xfeed.py --cleanup              # delete this run's browser screenshots from disk

Post IDs on X grow over time, so "newer than the last-read ID" is exact.
Screenshots taken by the browser tools are stored by the Claude app under
~/.claude/projects/*/<session>/tool-results/; --cleanup deletes the browser-tool
image files there that were created since --plan ran (this run's screenshots).
"""

import os, sys, json, glob, time, argparse
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.abspath(__file__))
XDIR = os.path.join(ROOT, "xfeed")
STATE = os.path.join(XDIR, "state.json")
RUN_START = os.path.join(XDIR, ".run_start")          # git-ignored
ET = ZoneInfo("America/New_York")
ACCOUNTS = ["DeItaone", "TheProfInvestor", "pelositracker", "mmonis", "asklivermore", "FL0WG0D", "BeardoTrader",
            "TrendSpider", "kpak82", "unusual_whales", "Mr_Derivatives", "DSAkdag", "HeidingOut", "TradersMastery",
            "zerohedge", "jaykaeppel"]
MAX_POSTS = 40          # newest posts read per account per run
FIRST_RUN_HOURS = 24    # with no saved position, read the last 24 hours
SHOT_GLOBS = ["mcp-claude-in-chrome-blob-*", "mcp-Claude_Browser-blob-*", "mcp-computer-use-blob-*"]

# Paste into the browser's JavaScript tool on a profile page: returns the visible posts.
EXTRACT_JS = r"""[...document.querySelectorAll('article[data-testid="tweet"]')].map(a => {
  const link = [...a.querySelectorAll('a[href*="/status/"]')].map(x => x.getAttribute('href')).find(h => /\/status\/\d+$/.test(h));
  const ctx = a.querySelector('[data-testid="socialContext"]');
  return {id: link ? link.split('/').pop() : null, handle: link ? link.split('/')[1] : null,
          time: a.querySelector('time')?.getAttribute('datetime'), context: ctx ? ctx.innerText : '',
          text: a.querySelector('[data-testid="tweetText"]')?.innerText || '',
          images: a.querySelectorAll('img[src*="pbs.twimg.com/media"]').length, video: !!a.querySelector('video')};
})"""


def _load():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {}


def plan():
    os.makedirs(XDIR, exist_ok=True)
    open(RUN_START, "w").write(str(time.time()))
    st, now = _load(), datetime.now(ET)
    did = f"{now:%Y-%m-%d}"
    print(f"RUN_DATE={did}")
    print(f"WRITE_DIGEST_TO=xfeed/{did}.md")
    print("ACCOUNT_NOTES=xfeed/accounts.md")
    prev = sorted(glob.glob(os.path.join(XDIR, "20*.md")))
    if prev:
        print("PREVIOUS_DIGEST=xfeed/" + os.path.basename(prev[-1]))
    print(f"MAX_POSTS_PER_ACCOUNT={MAX_POSTS}")
    print("\nACCOUNTS (handle | read posts with ID greater than)")
    for h in ACCOUNTS:
        last = st.get(h, {}).get("last_id")
        print(f"  {h} | {last if last else f'none yet: read the last {FIRST_RUN_HOURS} hours'}")
    print("\nEXTRACT_JS (run on each profile page with the browser JavaScript tool):")
    print(EXTRACT_JS)


def seen(handle, post_id):
    st = _load()
    cur = st.get(handle, {}).get("last_id")
    if not cur or int(post_id) > int(cur):
        st[handle] = {"last_id": str(post_id), "updated": datetime.now(ET).strftime("%Y-%m-%d %H:%M")}
        json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)
        print(f"{handle}: last read post is now {post_id}")
    else:
        print(f"{handle}: kept {cur} (not older)")


def index():
    items = []
    for f in sorted(glob.glob(os.path.join(XDIR, "20*.md")), reverse=True):
        title = next((l.lstrip("# ").strip() for l in open(f, encoding="utf-8") if l.startswith("# ")), os.path.basename(f))
        items.append({"id": os.path.basename(f)[:-3], "file": os.path.basename(f), "title": title})
    json.dump({"digests": items, "accounts_file": "accounts.md"}, open(os.path.join(XDIR, "index.json"), "w", encoding="utf-8"), indent=1)
    print(f"index: {len(items)} digests")


def cleanup():
    """Delete browser-tool screenshots created since --plan (i.e. during this run)."""
    try:
        since = float(open(RUN_START).read().strip())
    except Exception:
        since = time.time() - 6 * 3600            # no marker: only the last 6 hours
    base = os.path.join(os.path.expanduser("~"), ".claude", "projects")
    n = size = 0
    for pat in SHOT_GLOBS:
        for f in glob.glob(os.path.join(base, "*", "*", "tool-results", pat)):
            try:
                if os.path.getmtime(f) >= since - 60:
                    size += os.path.getsize(f)
                    os.remove(f)
                    n += 1
            except OSError:
                pass
    print(f"cleanup: deleted {n} screenshots ({size / 1e6:.1f} MB) created during this run")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--seen", nargs=2, metavar=("HANDLE", "POST_ID"))
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args()
    if a.plan: plan()
    elif a.seen: seen(*a.seen)
    elif a.index: index()
    elif a.cleanup: cleanup()
    else: ap.print_help()
