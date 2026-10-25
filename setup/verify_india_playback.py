"""Check wall pacing and pause measurements without starting CARLA."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from india_playback import Playback, SPEEDS


class PlaybackTests(unittest.TestCase):
    def test_supported_speeds_preserve_fixed_step(self):
        playback = Playback()
        for speed, fps in zip(SPEEDS, (20, 30, 40, 50, 60)):
            playback.select(speed)
            self.assertEqual(playback.fps_limit(), fps)
            self.assertEqual(playback.dt, .05)
            self.assertEqual(playback.fps_limit(paused=True), 20)
        for invalid in (0, -1, 1.1, 3.5, float("nan")):
            with self.assertRaises(ValueError):
                playback.select(invalid)

    def test_speed_limits_and_measurement_reset(self):
        playback = Playback()
        self.assertFalse(playback.adjust(-1))
        for _ in range(4):
            self.assertTrue(playback.adjust(1))
        self.assertEqual(playback.speed, 3)
        self.assertFalse(playback.adjust(1))
        playback.observe(10, 0, False)
        playback.observe(11, 3, False)
        self.assertAlmostEqual(playback.actual, 3)
        playback.select(1.5)
        self.assertIsNone(playback.actual)

    def test_actual_speed_and_pauses(self):
        playback = Playback(3)
        playback.observe(10, 0, False)
        playback.observe(11, 2.4, False)
        self.assertAlmostEqual(playback.actual, 2.4)
        playback.observe(11.1, 2.4, True)
        playback.observe(100, 2.4, True)
        self.assertIsNone(playback.actual)
        playback.observe(101, 2.4, False)
        playback.observe(102, 5.4, False)
        self.assertAlmostEqual(playback.actual, 3)
        self.assertEqual(playback.samples[-1]["requested_speed"], 3)
        self.assertAlmostEqual(playback.samples[-1]["window_wall_s"], 1)

    def test_recent_window_tracks_slowdown(self):
        playback = Playback(3)
        for tick in range(61):
            playback.observe(10 + tick / 60, tick * .05, False)
        self.assertAlmostEqual(playback.actual, 3)
        for tick in range(1, 21):
            playback.observe(11 + tick / 20, 3 + tick * .05, False)
        self.assertAlmostEqual(playback.actual, 1)


if __name__ == "__main__":
    unittest.main()
