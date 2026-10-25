"""Check the road/parking definition against new footpath geometry offline."""
from pathlib import Path
import sys
import unittest

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from india_road import junction_xml
from india_streetscape import paving_layout, footprint_samples, paving_pixels


class StreetscapeTests(unittest.TestCase):
    def test_raised_paving_stays_outside_all_driving_lanes(self):
        road_map = carla.Map("footpath-verification", junction_xml())
        samples = 0
        for tile in paving_layout():
            for x, y in footprint_samples(tile):
                self.assertIsNone(road_map.get_waypoint(carla.Location(x=x, y=y, z=0),
                                                       False, carla.LaneType.Driving))
                samples += 1
        self.assertGreater(samples, 10000)

    def test_east_parking_bay_stays_clear(self):
        for tile in paving_layout():
            if tile["x"] > 12 and tile["y"] < 0 and tile["yaw"] == 0:
                self.assertGreaterEqual(abs(tile["y"])-10*tile["scale"], 6.9-1e-9)
        pixels = paving_pixels()
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertGreater(float(pixels.std()), 8.)


if __name__ == "__main__":
    unittest.main()
