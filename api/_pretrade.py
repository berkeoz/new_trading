"""Vercel serverless function — pre-trade checklist data: oscillators and moving averages on 2H, 4H, 1D, 1W."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from mtf import pretrade


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        sym = parse_qs(urlparse(self.path).query).get("symbol", [""])[0].upper().strip()
        if not re.fullmatch(r"[A-Z0-9.^=\-]{1,15}", sym):
            body, code = {"error": "valid symbol parameter required"}, 400
        else:
            try:
                body, code = pretrade(sym), 200
            except Exception as e:
                body, code = {"error": str(e)}, 400
        data = json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "s-maxage=60" if code == 200 else "no-store")
        self.end_headers()
        self.wfile.write(data)
