"""Inspect the five authorised download endpoints without exposing their tokens."""
from pathlib import Path
import json
import urllib.request
import urllib.parse
import time

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "dataset/idd117k_raw"
config = json.loads((folder / "download_sources.private.json").read_text())
for source in config["sources"]:
    started = time.monotonic()
    req = urllib.request.Request(source["url"], method="HEAD", headers={"User-Agent":"Mozilla/5.0", "Accept-Encoding":"identity"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            source.update(final_url=response.url, status=response.status,
                headers=dict(response.headers.items()))
            safe={k:response.headers.get(k) for k in ("Content-Length","Content-Type","Content-Disposition","Accept-Ranges","ETag","Last-Modified")}
            print(json.dumps({"part":source["part"], "host":urllib.parse.urlparse(response.url).hostname,
                "status":response.status,"seconds":round(time.monotonic()-started,2),"headers":safe}),flush=True)
    except Exception as error:
        reason = getattr(error, "reason", error)
        print(json.dumps({"part":source["part"], "error_type":type(error).__name__,"code":getattr(error,"code",None),
            "reason_type":type(reason).__name__,"errno":getattr(reason,"errno",None),"winerror":getattr(reason,"winerror",None)}),flush=True)
(folder / "download_sources.private.json").write_text(json.dumps(config,indent=2))
