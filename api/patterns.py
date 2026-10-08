"""Vercel serverless function — chart formation analysis for a single ticker on demand."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from patterns import analyze, PERIODS, INTERVALS


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs     = parse_qs(urlparse(self.path).query)
        sym    = qs.get("symbol", [""])[0].upper().strip()
        period = qs.get("period", ["2y"])[0]
        interval = qs.get("interval", ["1d"])[0].lower()

        if not re.fullmatch(r"[A-Z0-9.^=\-]{1,15}", sym):
            body, code = {"error": "valid symbol parameter required"}, 400
        elif period not in PERIODS:
            body, code = {"error": f"period must be one of {', '.join(PERIODS)}"}, 400
        elif interval not in INTERVALS:
            body, code = {"error": f"interval must be one of {', '.join(INTERVALS)}"}, 400
        else:
            try:
                body, code = analyze(sym, period, interval=interval), 200
            except Exception as e:
                body, code = {"error": str(e)}, 400

        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", ("s-maxage=3600" if interval == "1d" else "s-maxage=600") if code == 200 else "no-store")
        self.end_headers()
        self.wfile.write(data)
