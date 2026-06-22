import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cs336_systems.VaadGen import load_vaad, vaad_generate

CONFIG_PATH = os.environ.get("VAAD_CONFIG", "configs/small.toml")
HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")

# Loaded once at server startup; reused for every request.
_v_obj, _cfg = load_vaad(CONFIG_PATH)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        with open(HTML_PATH, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        text = payload.get("text", "")

        try:
            response = vaad_generate(_v_obj, _cfg, text)
            out = {"response": response}
        except Exception as e:
            out = {"error": str(e)}

        body = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet default request logging


if __name__ == "__main__":
    port = int(os.environ.get("VAAD_PORT", 8000))
    print(f"Model loaded. Serving on http://localhost:{port}")
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
