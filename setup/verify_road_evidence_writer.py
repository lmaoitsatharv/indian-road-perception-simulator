"""Evidence writes must finish in order and propagate errors to the capture."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from india_road_observer import RoadObserver
from india_evidence_io import write_rgb_png, write_depth_npz

TEST_ROOT = Path(__file__).resolve().parents[1] / "setup/road_writer_test"
TEST_ROOT.mkdir(exist_ok=True)


def observer(folder, count, save):
    value = RoadObserver.__new__(RoadObserver)
    value.folder = Path(folder)
    value.inbox = queue.Queue()
    value.pending = {k: {} for k in ("rgb", "labels", "depth")}
    value.snapshots = {i: {} for i in range(1, count+1)}
    value.writer = ThreadPoolExecutor(max_workers=2)
    value.writes, value.write_ms, value.backpressure_s = deque(), [], 0.
    value.records, value.log = [], io.StringIO()
    value.dropped = {"queue_overflow": 0, "expired_images": 0}
    value.write_pair = lambda frame, images, state: save(value, frame)
    for frame in range(1, count+1):
        for kind in value.pending:
            value.inbox.put((kind, SimpleNamespace(frame=frame)))
    return value


class WriterTests(unittest.TestCase):
    def test_exact_pixels_and_float32_depth_survive_fast_encoding(self):
        rng = np.random.default_rng(29)
        # Noncontiguous BGRA->RGB view, matching a native sensor buffer.
        pixels = rng.integers(0, 256, (41, 69, 4), dtype=np.uint8)[:, :, 2::-1]
        depth = rng.uniform(0, 1000, (41, 69)).astype(np.float32)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            folder = Path(folder)
            write_rgb_png(folder/"rgb.png", pixels)
            restored = pygame.surfarray.array3d(pygame.image.load(str(folder/"rgb.png"))).swapaxes(0,1)
            np.testing.assert_array_equal(restored, pixels)
            write_depth_npz(folder/"depth.npz", depth)
            with np.load(folder/"depth.npz", allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["meters"], depth)
            with self.assertRaises(FileExistsError):
                write_rgb_png(folder/"rgb.png", pixels)

    def test_encoding_can_run_off_the_tick_thread_and_finish_drains(self):
        release = threading.Event()
        def save(value, frame):
            if not release.wait(3):
                raise TimeoutError("Producer blocked unexpectedly")
            return {"frame": frame}, 1.
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            value = observer(folder, 2, save)
            try:
                value.collect()
                self.assertLessEqual(len(value.writes), 4)
                self.assertEqual(value.records, [])
            finally:
                release.set()
                value.finish()
            self.assertEqual([r["frame"] for r in value.records], [1, 2])
            self.assertTrue((Path(folder)/"capture_summary.json").is_file())

    def test_later_encoding_may_finish_first_but_log_stays_ordered(self):
        second_done = threading.Event()
        def save(value, frame):
            if frame == 1:
                if not second_done.wait(3):
                    raise TimeoutError("Second encoder did not run")
            else:
                second_done.set()
            return {"frame": frame}, 1.
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            value = observer(folder, 2, save)
            value.collect()
            value.finish()
            self.assertEqual([r["frame"] for r in value.records], [1, 2])

    def test_encoder_failure_prevents_a_success_summary(self):
        def save(value, frame):
            raise ValueError("Injected disk/encoding failure")
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            value = observer(folder, 1, save)
            value.collect()
            with self.assertRaisesRegex(ValueError, "Injected"):
                value.finish()
            self.assertFalse((Path(folder)/"capture_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
