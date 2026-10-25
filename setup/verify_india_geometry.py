"""Compare the scalar body-gap kernel with the preceding implementation and audit."""
import ast
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "setup/carla_india"))
from india_behavior import polygon_gap, rectangle
from verify_run import gap as independent_gap


def baseline():
    raw = subprocess.check_output(["git", "show", "3b70f5064fa3b8543fdb77a836e17a7592303350:integration/india_behavior.py"], cwd=ROOT).decode()
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "polygon_gap")
    context = {"np": np, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<preceding-project-geometry>", "exec"), context)
    return context["polygon_gap"]


def examples():
    pairs = [(rectangle(0,0,0,1,1), rectangle(x,0,0,1,1)) for x in (0,1,2,2.000001,4)]
    rng = random.Random(20261020)
    for _ in range(1500):
        polygons = [rectangle(rng.uniform(-8,8), rng.uniform(-8,8), rng.uniform(-math.pi,math.pi),
                             rng.uniform(.15,5), rng.uniform(.15,1.5)) for _ in range(2)]
        pairs.append(tuple(polygons))
    return pairs


class GeometryTests(unittest.TestCase):
    def test_rotations_touching_overlap_and_separation(self):
        old = baseline()
        for a, b in examples():
            previous, current = old(a,b), polygon_gap(a,b)
            self.assertEqual(previous[0], current[0])
            self.assertAlmostEqual(previous[1], current[1], places=10)
            self.assertAlmostEqual(independent_gap(a,b), current[1], places=10)


def benchmark():
    old, pairs = baseline(), examples()
    measurements = []
    for kernel in (old, polygon_gap):
        values = []
        for _ in range(3):
            start = time.perf_counter()
            for a,b in pairs:
                kernel(a,b)
            values.append(time.perf_counter()-start)
        measurements.append(min(values))
    print(json.dumps({"pairs": len(pairs), "repeats":3, "old_best_s":measurements[0],
                      "new_best_s":measurements[1], "kernel_speedup":measurements[0]/measurements[1],
                      "limit":"Kernel microbenchmark on this laptop; not end-to-end simulator speed"}))


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
