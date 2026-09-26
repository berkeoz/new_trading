"""Vercel serverless function — evaluates a single ticker on demand."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from scanner import analyze_symbol, _card


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs  = parse_qs(urlparse(self.path).query)
        sym = qs.get("symbol", [""])[0].upper().strip()

        if not sym:
            body = json.dumps({"error": "symbol parameter required"}).encode()
            code = 400
        else:
            result = analyze_symbol(sym)
            if result.get("error"):
                body = json.dumps({"error": result["error"]}).encode()
                code = 400
            else:
                card_html = _card(result)
                body = json.dumps({
                    "html":   card_html,
                    "signal": result.get("entry_signal"),
                    "score":  result.get("entry_score"),
                }).encode()
                code = 200

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
