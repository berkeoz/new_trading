"""Vercel serverless function — intrinsic value estimate (DCF, reverse DCF, multiples) for one ticker."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from valuation import valuation


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs  = parse_qs(urlparse(self.path).query)
        sym = qs.get("symbol", [""])[0].upper().strip()
        if not re.fullmatch(r"[A-Z0-9.^=\-]{1,15}", sym):
            body, code = {"error": "valid symbol parameter required"}, 400
        else:
            try:
                body, code = valuation(sym), 200
            except Exception as e:
                body, code = {"error": str(e)}, 400
        data = json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        # fundamentals change slowly; cache for 6 hours at the edge
        self.send_header("Cache-Control", "s-maxage=21600" if code == 200 else "no-store")
        self.end_headers()
        self.wfile.write(data)
