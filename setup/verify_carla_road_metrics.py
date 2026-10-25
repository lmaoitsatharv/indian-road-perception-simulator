"""Check label semantics, ignored pixels and the independent depth projection kernel."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from carla_road_reference import build_reference
from evaluate_carla_road import confusion, metrics, reference


class RoadMetricsTests(unittest.TestCase):
    def test_confusion_excludes_unknowns(self):
        label = np.array([[1,1,0,0,255]], np.uint8)
        pred = np.array([[1,0,1,0,1]], np.uint8)
        value = confusion(pred, label)
        self.assertEqual(value, dict(tp=1,fp=1,fn=1,tn=1,ignored=1,total=5))
        self.assertAlmostEqual(metrics(value)["road_iou"], 1/3)
        self.assertEqual(metrics(value)["evaluated_fraction"], .8)

    def test_empty_denominators_are_not_perfect_scores(self):
        zero = dict(tp=0,fp=0,fn=0,tn=2,ignored=1,total=3)
        self.assertIsNone(metrics(zero)["road_iou"])
        self.assertIsNone(metrics(zero)["recall"])
        with self.assertRaises(ValueError):
            reference(np.array([[1,24]], np.uint8))

    def test_visible_ground_and_occlusion(self):
        tags, depth = np.full((5,5), 11, np.uint8), np.ones((5,5), np.float32)
        tags[1] = 0
        tags[1,1], tags[1,2], tags[1,4] = 14, 2, 12
        transform = np.eye(4)
        transform[2,3] = 1
        config = {"schema_version":2,"intrinsics_source":[[1,0,2],[0,1,0],[0,0,1]],
            "reference":{"geometry_range_m":80,"surface_height_tolerance_m":.08,
                "boundary_uncertainty_m":1,"grid":{"origin_xy_m":[-5,-5],"cell_m":1}}}
        result, stats = build_reference(tags,depth,transform,config,np.ones((20,20),np.uint8))
        self.assertEqual(result[1].tolist(), [1,0,0,1,0])
        self.assertEqual(result[0,0],0)
        depth[1,0] = 1000
        result,_ = build_reference(tags,depth,transform,config,np.ones((20,20),np.uint8))
        self.assertEqual(result[1,0],255)
        self.assertEqual(stats["ground_height_abs_median_m"],0)

    def test_owned_surface_does_not_relabel_other_buildings(self):
        tags = np.full((5,5), 3, np.uint8)
        depth = np.ones((5,5), np.float32)
        pose = np.eye(4)
        pose[2,3] = 1
        config = {"schema_version":3,"intrinsics_source":[[1,0,2],[0,1,0],[0,0,1]],
            "reference":{"appearance_native_class":3,"geometry_range_m":80,
                "surface_height_tolerance_m":.08,"boundary_uncertainty_m":1,
                "grid":{"origin_xy_m":[-5,-5],"cell_m":1}}}
        appearance = np.zeros((5,5), bool)
        appearance[1,1] = True
        target,_ = build_reference(tags,depth,pose,config,np.ones((20,20),np.uint8),appearance)
        self.assertEqual(target[1].tolist(), [0,1,0,0,0])
        with self.assertRaises(ValueError):
            build_reference(tags,depth,pose,config,np.ones((20,20),np.uint8))


if __name__ == "__main__":
    unittest.main()
