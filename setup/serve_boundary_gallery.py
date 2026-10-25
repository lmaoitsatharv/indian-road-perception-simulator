"""Serve the completed local review gallery on an automatically assigned port."""
import functools
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "runs/boundary_20261003/gallery"
handler = functools.partial(SimpleHTTPRequestHandler, directory=str(GALLERY))
server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
port = server.server_address[1]
(GALLERY / "local_server.json").write_text(json.dumps(dict(pid=os.getpid(), port=port,
    url=f"http://127.0.0.1:{port}/index.html", directory=str(GALLERY)), indent=2), encoding="utf-8")
print(f"http://127.0.0.1:{port}/index.html", flush=True)
server.serve_forever()
