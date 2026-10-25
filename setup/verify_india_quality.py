"""Check pedestrian entry conflicts and native light flags without running CARLA."""
from pathlib import Path
import sys
from types import SimpleNamespace as Object
import unittest

import carla
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration"))
from india_pedestrian_spacing import entry_blockers
from india_signals import light_bits, SignalLights


class QualityTests(unittest.TestCase):
    def test_light_readback_waits_for_a_world_tick(self):
        class Actor:
            id, actual, pending = 1, 0, 0
            def set_light_state(self, value):
                self.pending = int(value)
            def get_light_state(self):
                return carla.VehicleLightState(self.actual)
        actor = Actor()
        driver = Object(actor=actor, name="ego", pass_state="passing", parked=False, s=0)
        lights = SignalLights("clear")
        lights.update(driver, Object(brake=.2), 0., 10)
        lights.verify([driver], 10)
        self.assertIsNone(lights.records[0]["readback_frame"])
        actor.actual = actor.pending  # Native state becomes visible in the next tick.
        lights.verify([driver], 11)
        self.assertTrue(lights.records[0]["verified"])
        self.assertEqual(lights.records[0]["readback_frame"], 11)

    def test_walker_moving_into_entry_gap_delays_crossing(self):
        person = Object(a=(47, 6.25), b=(47, -8), speed=1.15, actor=Object(id=1))
        other = Object(a=(30, 5.45), b=(52, 5.45), speed=1.2, active=True,
                       reverse=True, completed=False, actor=Object(id=2), name="footpath_3")
        # Clear now, but their paths intersect shortly after the kerb departure.
        self.assertEqual(entry_blockers(person, [person, other], {2: {"x": 48.4, "y": 5.45}}), ["footpath_3"])
        self.assertEqual(entry_blockers(person, [person, other], {2: {"x": 35., "y": 5.45}}), [])
        other.completed = True
        self.assertEqual(entry_blockers(person, [person, other], {2: {"x": 47., "y": 5.45}}), [])

    def test_signals_match_turn_pass_and_braking(self):
        route = Object(points=np.array([[0, 0, 0.]]), junction_end=165.,
                       at=lambda _: np.array([0, 0, np.pi/2]))
        driver = Object(name="ego", pass_state="passing", parked=False, s=110., route=route)
        control = Object(brake=.2)
        flags = carla.VehicleLightState
        right = light_bits(driver, control, "clear")
        self.assertTrue(right & int(flags.Brake) and right & int(flags.RightBlinker))
        self.assertFalse(right & int(flags.LeftBlinker))
        driver.pass_state = "returning"
        self.assertTrue(light_bits(driver, control, "clear") & int(flags.LeftBlinker))
        driver.pass_state = "follow"
        self.assertTrue(light_bits(driver, control, "clear") & int(flags.RightBlinker))
        route.at = lambda _: np.array([0, 0, -np.pi/2])
        self.assertTrue(light_bits(driver, control, "clear") & int(flags.LeftBlinker))
        driver.s, driver.name = 20., "npc"
        control.brake = 0.
        self.assertEqual(light_bits(driver, control, "clear"), int(flags.Position))
        self.assertTrue(light_bits(driver, control, "dusk") & int(flags.LowBeam))


if __name__ == "__main__":
    unittest.main()
