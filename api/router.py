"""The only Vercel function: serves /api/patterns, /api/valuation, /api/mtf and /api/scan.

Each Python function on Vercel bundles pandas / numpy / yfinance (~150 MB), and
Vercel keeps every deployment, so four separate functions used four times the
storage. The handlers live in api/_*.py (a leading underscore means Vercel does
not deploy them as functions of their own); this router picks one by ?fn= or
by the last part of the path (vercel.json rewrites /api/<name> here).
"""
import sys, os, json, importlib.util
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
NAMES = ("patterns", "valuation", "mtf", "scan")
_cache = {}


def _module(name):
    if name not in _cache:
        spec = importlib.util.spec_from_file_location(f"api_{name}", os.path.join(HERE, f"_{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[name] = mod
    return _cache[name]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        name = parse_qs(u.query).get("fn", [""])[0] or u.path.rstrip("/").rsplit("/", 1)[-1]
        if name not in NAMES:
            body = json.dumps({"error": f"unknown endpoint; use one of {', '.join(NAMES)}"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        return _module(name).handler.do_GET(self)
