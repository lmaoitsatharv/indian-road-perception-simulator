"""Serve only the completed test comparison gallery on localhost."""
import functools
import json
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "runs/test_evaluation_20261003_morning_best/gallery"
assert (GALLERY / "index.html").is_file()
handler = functools.partial(SimpleHTTPRequestHandler, directory=str(GALLERY))
server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
port = server.server_address[1]
metadata = {"pid": os.getpid(), "port": port,
            "url": f"http://127.0.0.1:{port}/index.html",
            "directory": str(GALLERY),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "local read-only test gallery; no model inference"}
temporary = GALLERY / "local_server.json.tmp"
temporary.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
temporary.replace(GALLERY / "local_server.json")
print(metadata["url"], flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
