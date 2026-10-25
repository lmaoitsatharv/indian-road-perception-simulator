"""Verify explanations are observational and preserve the preceding driving policy."""
import ast
from copy import deepcopy
import itertools
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "12f20af9729e432d7f6d34da24c054d57614150d"
sys.path.insert(0, str(ROOT / "integration"))
from india_decisions import describe_ego


def method(tree, cls, name):
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)


def is_observation(node):
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if (isinstance(node, ast.Attribute) and node.attr == "pass_explanation"
                and isinstance(node.value, ast.Name) and node.value.id == "self"):
            return True
        node = node.value
    return False


class RemoveObservationWrites(ast.NodeTransformer):
    def visit_Assign(self, node):
        return None if all(is_observation(t) for t in node.targets) else self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and is_observation(node.value.func):
            return None
        return self.generic_visit(node)


class RemovePedestrianEntryGap(ast.NodeTransformer):
    def visit_If(self, node):
        call = node.test
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_wait_for_footpath_gap"):
            return None
        return self.generic_visit(node)


class Context(SimpleNamespace):
    def emit(self, *args, **kwargs):
        self.events.append((args, kwargs))


def context(progress=92.6, gap=10.7, oncoming=None, person=False, width=.3,
            phase="follow", guard=False, offset=0., y=-2.25):
    route = SimpleNamespace(junction_end=164.)
    def driver(actor_id, name, desired, lateral, half_width):
        return SimpleNamespace(actor=SimpleNamespace(id=actor_id), name=name, desired=desired,
            half_length=2.4 if actor_id == 1 else .85, half_width=half_width,
            offset=lateral, pass_offset=1.08, pass_state="follow", pass_leader=None,
            blocked_reason=None, s=progress, route=route, reason="headway", target=2.4,
            finished=False, goal_s=264.)
    ego, lead = driver(1, "ego", 6.8, offset, 1.08), driver(2, "slow_cycle", 2.4, -1.28, width)
    ego.pass_state, ego.pass_leader = phase, (2 if phase != "follow" else None)
    value = Context(ego=ego, drivers=[ego, lead], walkers=[], blocked=set(), overtakes=0, events=[])
    x = progress - 152
    states = {1: {"x": x, "y": y, "yaw": 0., "speed": 2.4},
              2: {"x": x + gap, "y": -3.53, "yaw": 0., "speed": 2.4}}
    if oncoming is not None:
        value.drivers.append(driver(3, "oncoming_car", 7., 0., 1.))
        states[3] = {"x": x + oncoming, "y": 2.25, "yaw": math.pi, "speed": 7.}
    if guard:
        value.drivers.append(driver(5, "return_guard", 3., -.3, .4))
        states[5] = {"x": x + 5, "y": -2.55, "yaw": 0., "speed": 3.}
    if person:
        value.walkers.append(SimpleNamespace(actor=SimpleNamespace(id=4), name="crossing_1", crossing=True,
                                             completed=False, start=5.))
        states[4] = {"x": x + 8, "y": 0., "yaw": 0., "speed": 1.2}
    return value, states


def policy_state(value):
    fields = ("offset", "pass_offset", "pass_state", "pass_leader", "blocked_reason")
    return {"drivers": [{k: getattr(d, k) for k in fields} for d in value.drivers],
            "overtakes": value.overtakes, "blocked": value.blocked, "events": value.events}


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = subprocess.check_output(["git", "show", f"{BASELINE}:integration/india_behavior.py"], cwd=ROOT).decode()
        cls.old_tree = ast.parse(baseline)
        cls.new_tree = ast.parse((ROOT / "integration/india_behavior.py").read_text())
        namespace = {"math": math, "DT": .05, "ARM_LENGTH": 140., "LANE_WIDTH": 4.5}
        cls.functions = []
        for tree in (cls.old_tree, cls.new_tree):
            block = ast.Module(body=[deepcopy(method(tree, "Traffic", "_overtake"))], type_ignores=[])
            environment = dict(namespace)
            exec(compile(block, "<trusted-project-passing-rule>", "exec"), environment)
            cls.functions.append(environment["_overtake"])

    def test_vehicle_policy_unchanged_except_declared_pedestrian_entry_gap(self):
        old = method(self.old_tree, "Traffic", "_overtake")
        new = RemoveObservationWrites().visit(deepcopy(method(self.new_tree, "Traffic", "_overtake")))
        self.assertEqual(ast.dump(old), ast.dump(new))
        self.assertEqual(ast.dump(method(self.old_tree, "Driver", "control")),
                         ast.dump(method(self.new_tree, "Driver", "control")))
        # V4 intentionally adds one kerb-entry gap check. Existing vehicle rules
        # remain identical after removing that explicitly tested addition.
        new_step = RemovePedestrianEntryGap().visit(deepcopy(method(self.new_tree, "Traffic", "step")))
        self.assertEqual(ast.dump(method(self.old_tree, "Traffic", "step")), ast.dump(new_step))

    def test_threshold_boundaries_and_state_transitions_match_old_policy(self):
        cases = []
        for progress, gap, oncoming, person, width in itertools.product(
                (20., 92.6, 105., 105.01, 140., 164., 189., 189.01, 195.),
                (3.99, 4., 4.01, 10.7, 29.99, 30.), (None, 10., 130., 300.), (False, True), (.3, 1.6)):
            cases.append(dict(progress=progress, gap=gap, oncoming=oncoming, person=person, width=width))
        for phase, gap, guard, offset, y in itertools.product(
                ("passing", "returning"), (15., -8., -20.), (False, True), (.01, 1.08), (-2.25, -1.2)):
            cases.append(dict(phase=phase, gap=gap, guard=guard, offset=offset, y=y))
        for case in cases:
            a, states = context(**case)
            b = deepcopy(a)
            self.functions[0](a, states, 28.1)
            self.functions[1](b, states, 28.1)
            self.assertEqual(policy_state(a), policy_state(b), case)
        self.assertEqual(len(cases), 912)

    def test_junction_explanation_uses_actual_rule_numbers(self):
        value, states = context()
        self.functions[1](value, states, 28.1)
        result = describe_ego(value, states, 28.1, 100)
        self.assertEqual(result["passing"]["reason"], "junction_too_close")
        constraint = result["passing"]["blockers"][-1]
        self.assertAlmostEqual(constraint["available_m"], 47.4)
        self.assertGreater(constraint["required_m"], constraint["available_m"])
        self.assertIn("available 47.4 m", result["hud_lines"][1])
        self.assertEqual(result["source_frame"], 100)

    def test_junction_zone_replaces_stale_message_without_changing_control(self):
        value, states = context(progress=110.)
        value.ego.blocked_reason = "oncoming_gap"
        self.functions[1](value, states, 35.)
        result = describe_ego(value, states, 35., 100)
        self.assertEqual(result["passing"]["reason"], "junction_zone")
        self.assertIn("junction zone", result["hud_lines"][1])
        self.assertEqual(value.ego.blocked_reason, "oncoming_gap")

    def test_other_blockers_and_stopping_are_explained(self):
        for case, expected in ((dict(progress=20., oncoming=10.), "oncoming_gap"),
                               (dict(progress=20., person=True), "pedestrian_conflict"),
                               (dict(progress=20., width=1.6), "body_clearance"),
                               (dict(phase="passing", gap=-8., guard=True), "return_gap")):
            value, states = context(**case)
            self.functions[1](value, states, 28.1)
            result = describe_ego(value, states, 28.1, 100)
            self.assertEqual(result["passing"]["reason"], expected)
            self.assertTrue(result["hud_lines"][1])
        value.ego.finished = True
        self.assertEqual(describe_ego(value, states, 28.1, 100)["action"], "STOPPING")
        states[1]["speed"] = 0.
        self.assertEqual(describe_ego(value, states, 28.1, 100)["action"], "STOPPED")


if __name__ == "__main__":
    unittest.main()
