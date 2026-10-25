"""CPU-only regression tests for exact strict real-retention decisions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"setup"))
from rehearse_road_retention import strict_ratios


class StrictRetentionTests(unittest.TestCase):
    def test_identical_ratios_pass_without_float_rounding(self):
        baseline=dict(tp=2,fp=1,fn=1,boundary_intersection_5=1,boundary_union_5=5)
        candidate=dict(tp=20,fp=10,fn=10,boundary_intersection_5=2,boundary_union_5=10)
        self.assertTrue(strict_ratios(candidate,baseline)["passed"])

    def test_one_pixel_road_regression_fails(self):
        baseline=dict(tp=100000000,fp=1,fn=1,boundary_intersection_5=1,boundary_union_5=5)
        candidate={**baseline,"fp":2}
        result=strict_ratios(candidate,baseline)
        self.assertFalse(result["road_iou_not_lower"])
        self.assertFalse(result["passed"])

    def test_boundary_only_regression_fails(self):
        baseline=dict(tp=2,fp=1,fn=1,boundary_intersection_5=100,boundary_union_5=500)
        candidate={**baseline,"boundary_intersection_5":99}
        result=strict_ratios(candidate,baseline)
        self.assertTrue(result["road_iou_not_lower"])
        self.assertFalse(result["boundary5_iou_not_lower"])
        self.assertFalse(result["passed"])


if __name__=="__main__":unittest.main()
