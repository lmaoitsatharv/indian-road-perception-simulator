"""Camera-driving road worker (WSL, idd-tf220): simulation road model, mask + probability."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setup.camera_driving_service import serve


def main(session_file):
    import json
    import numpy as np
    from PIL import Image
    session = json.loads(Path(session_file).read_text())

    def load(folder):
        from idd_driving_inference import DrivingRoadSegmenter
        model = DrivingRoadSegmenter(ROOT / session["road"]["manifest"])
        model.warmup()
        runtime = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in model.runtime.items()}
        return model, runtime

    def infer(model, rgb):
        started = time.perf_counter()
        small = np.array(Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR))
        resize_ms = (time.perf_counter() - started) * 1000
        prediction = model.predict_rgb(small)
        mask = prediction.mask.astype(np.uint8)
        probability = np.round(prediction.probability * 255).astype(np.uint8)
        return ({"inference_ms": prediction.inference_ms, "resize_ms": resize_ms, "road_fraction": float(mask.mean())},
                mask.tobytes() + probability.tobytes())

    def fingerprint(model):
        digest = hashlib.sha256()
        for weight in model.model.weights:
            digest.update(weight.numpy().tobytes())
        return digest.hexdigest()

    serve(session_file, "road", load, infer, fingerprint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    main(parser.parse_args().session.resolve())
