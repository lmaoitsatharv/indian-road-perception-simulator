"""Native vehicle lights reflecting the existing driver; never changes its controls."""
import math
import carla


def light_bits(driver, control, weather):
    flags = carla.VehicleLightState
    bits = int(flags.Position)
    if driver.name == "ego" or weather in ("wet", "dusk"):
        bits |= int(flags.LowBeam)
    if control.brake > .05:
        bits |= int(flags.Brake)
    if driver.pass_state == "passing":
        bits |= int(flags.RightBlinker)
    elif driver.pass_state == "returning":
        bits |= int(flags.LeftBlinker)
    elif not driver.parked and 100 < driver.s < driver.route.junction_end + 4:
        start, end = driver.route.points[0, 2], driver.route.at(driver.route.junction_end)[2]
        delta = math.atan2(math.sin(end-start), math.cos(end-start))
        if abs(delta) > .5:
            # CARLA yaw increases clockwise in its left-handed coordinates.
            bits |= int(flags.RightBlinker if delta > 0 else flags.LeftBlinker)
    return bits


class SignalLights:
    def __init__(self, weather):
        self.weather, self.states, self.records, self.pending = weather, {}, [], {}

    def verify(self, drivers, frame):
        for driver in drivers:
            record = self.pending.get(driver.actor.id)
            if record is not None and frame > record["source_frame"]:
                actual = int(driver.actor.get_light_state())
                record.update(native_bits=actual, readback_frame=frame,
                              verified=record["commanded_bits"] == actual)
                del self.pending[driver.actor.id]

    def update(self, driver, control, sim_s, frame):
        bits = light_bits(driver, control, self.weather)
        if self.states.get(driver.actor.id) == bits:
            return
        driver.actor.set_light_state(carla.VehicleLightState(bits))
        record = {"sim_s": sim_s, "source_frame": frame, "actor": driver.name,
                  "commanded_bits": bits, "native_bits": None, "readback_frame": None, "verified": False}
        self.records.append(record)
        self.pending[driver.actor.id] = record
        self.states[driver.actor.id] = bits
