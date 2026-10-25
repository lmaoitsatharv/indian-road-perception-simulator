"""Bounded loopback messages and timestamp rules shared by both live processes."""
import json
import struct
import threading

PROTOCOL = "native_carla_road_preview_v1"
MAX_PAYLOAD = 2_000_000


def receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        data = connection.recv(size-len(chunks))
        if not data:
            raise ConnectionError("Live observer connection closed")
        chunks.extend(data)
    return bytes(chunks)


def send_packet(connection, header, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Live payload exceeds its fixed image bound")
    encoded = json.dumps({**header,"payload_bytes":len(payload)},allow_nan=False).encode()
    if len(encoded)>8192:
        raise ValueError("Live header exceeds its bound")
    connection.sendall(struct.pack("!I",len(encoded))+encoded+payload)


def receive_packet(connection):
    length = struct.unpack("!I",receive_exact(connection,4))[0]
    if not 1 <= length <= 8192:
        raise ValueError("Invalid live header length")
    header = json.loads(receive_exact(connection,length))
    size = header.get("payload_bytes")
    if type(size) is not int or not 0 <= size <= MAX_PAYLOAD:
        raise ValueError("Invalid live payload size")
    return header,receive_exact(connection,size)


class LatestSlot:
    """One pending frame, with replacement counted rather than a growing queue."""
    def __init__(self):
        self.condition=threading.Condition()
        self.value=None
        self.replaced=0
        self.closed=False

    def put(self,value):
        with self.condition:
            if self.closed:
                return
            self.replaced += int(self.value is not None)
            self.value=value
            self.condition.notify()

    def take(self,timeout=.1):
        with self.condition:
            if self.value is None and not self.closed:
                self.condition.wait(timeout)
            value,self.value=self.value,None
            return value

    def close(self):
        with self.condition:
            self.closed=True
            self.value=None
            self.condition.notify_all()


def freshness(header, *, now_ns, current_sim_s, connected, session, wall_limit_ms=500, sim_limit_ms=500):
    """All absolute wall times here originate in the Windows capture process.

    Tick-start -> presentation is a conservative capture-age upper bound. The
    service's independent monotonic clock is used only for its own durations.
    """
    if not connected:
        return {"state":"offline","age_upper_ms":None,"simulation_age_ms":None}
    if header is None:
        return {"state":"waiting","age_upper_ms":None,"simulation_age_ms":None}
    if header.get("session") != session:
        return {"state":"wrong_session","age_upper_ms":None,"simulation_age_ms":None}
    age=(now_ns-header["tick_start_ns"])/1e6
    simulation_age=(current_sim_s-header["simulation_timestamp_s"])*1000
    state="fresh" if 0 <= age <= wall_limit_ms and -.001 <= simulation_age <= sim_limit_ms else "stale"
    return {"state":state,"age_upper_ms":age,"simulation_age_ms":simulation_age}
