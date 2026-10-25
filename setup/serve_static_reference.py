"""Read-only localhost preview of the saved static MATLAB reference artifacts."""
import functools
import json
import os
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--folder',type=Path,default=ROOT / 'runs/static_reference_20261003')
args=parser.parse_args()
FOLDER=args.folder.resolve()
assert FOLDER.is_relative_to(ROOT/'runs'), 'Serve only a saved project run.'
assert (FOLDER / 'replay.html').is_file()
server = ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(FOLDER)))
port = server.server_address[1]
record = {'pid':os.getpid(),'port':port,'url':f'http://127.0.0.1:{port}/replay.html',
          'directory':str(FOLDER),'purpose':'read-only geometric replay; no model inference or training'}
target = FOLDER / 'local_server.json'
temp = target.with_suffix('.tmp')
temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
temp.replace(target)
print(record['url'],flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
