"""Low-latency lossless camera evidence; exact pixels, no JPEG or reduced depth."""
import io
from pathlib import Path
import struct
import zipfile
import zlib

import numpy as np


def write_rgb_png(path, pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Evidence PNG expects uint8 RGB")
    height, width = pixels.shape[:2]
    scanlines = np.empty((height, width*3+1), np.uint8)
    scanlines[:, 0] = 0  # PNG filter None: no pixel transformation.
    scanlines[:, 1:] = pixels.reshape(height, width*3)
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data)&0xffffffff)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", header)
    payload += chunk(b"IDAT", zlib.compress(scanlines.tobytes(), level=1))+chunk(b"IEND", b"")
    with Path(path).open("xb") as stream:
        stream.write(payload)


def write_depth_npz(path, depth):
    if depth.dtype != np.float32 or depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("Evidence depth expects a finite float32 image")
    memory = io.BytesIO()
    np.save(memory, depth, allow_pickle=False)
    with zipfile.ZipFile(path, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr("meters.npy", memory.getvalue())
