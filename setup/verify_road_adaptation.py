"""CPU-only checks for the separate adaptation evaluation and split handling."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "setup"))
from adapt_road_boundaries import count_pixels, scores, real_records


class AdaptationTests(unittest.TestCase):
    def test_unknown_is_not_scored_as_road_or_background(self):
        truth = np.array([[1, 0, 255], [1, 0, 255]], np.uint8)
        guessed = np.array([[1, 1, 0], [0, 0, 1]], np.uint8)
        result = count_pixels(truth, guessed)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn", "ignored", "total")],
                         [1, 1, 1, 1, 2, 6])
        self.assertAlmostEqual(scores(result)["road_iou"], 1/3)

    def test_unknown_contours_have_a_boundary_exclusion_halo(self):
        truth = np.zeros((40,40), np.uint8)
        truth[10:30,10:30] = 1
        truth[17:23,17:23] = 255
        prediction = (truth == 1).astype(np.uint8)
        first = count_pixels(truth, prediction)
        prediction[truth == 255] = 1
        second = count_pixels(truth, prediction)
        self.assertEqual(first, second)
        self.assertEqual(scores(second)["road_iou"], 1.)
        self.assertEqual(scores(second)["boundary_iou_5px"], 1.)

    def test_only_training_and_validation_have_constructed_paths(self):
        records = real_records()
        self.assertEqual({name: len(rows) for name, rows in records.items()}, {"train":5507, "val":687})
        self.assertFalse(any("/test/" in row["image"].replace("\\", "/") for rows in records.values() for row in rows))


if __name__ == "__main__":
    unittest.main()
