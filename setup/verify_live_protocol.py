"""Check scope, message bounds, latest-only buffering and freshness arithmetic."""
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_live_protocol import LatestSlot,freshness,send_packet,receive_packet
from idd_live_simulation_inference import load_live_spec
from idd_simulation_inference import load_simulation_model_spec
from setup.prepare_simulation_dataset import sha,save,protected


def main():
    socket.setdefaulttimeout(3.)
    before=protected()
    slot=LatestSlot()
    for frame in range(100):
        slot.put({"frame":frame})
    assert slot.replaced==99 and slot.take()["frame"]==99
    assert slot.take(timeout=0) is None
    slot.close()
    assert slot.take(timeout=0) is None
    header={"session":"test","tick_start_ns":1_000_000_000,"simulation_timestamp_s":10.}
    def state(now=1_200_000_000,simulation=10.2,connected=True,session="test"):
        return freshness(header,now_ns=now,current_sim_s=simulation,connected=connected,session=session)["state"]
    assert state()=="fresh" and state(now=1_500_000_000)=="fresh"
    assert state(now=1_500_000_001)=="stale"
    assert state(simulation=10.6)=="stale"
    assert state(now=999_999_999)=="stale"
    assert state(now=2_000_000_000,simulation=10.)=="stale"  # Paused simulation still ages in wall time.
    assert state(connected=False)=="offline" and state(session="other")=="wrong_session"
    assert freshness(None,now_ns=0,current_sim_s=0,connected=True,session="test")["state"]=="waiting"
    left,right=socket.socketpair()
    with left,right:
        send_packet(left,{"frame":17},b"rgb-bytes")
        send_packet(left,{"frame":18},b"next-frame")
        first,data=receive_packet(right)
        second,next_data=receive_packet(right)
        assert first["frame"]==17 and data==b"rgb-bytes" and second["frame"]==18 and next_data==b"next-frame"
    left,right=socket.socketpair()
    with left,right:
        left.sendall(struct.pack("!I",9000))
        try:
            receive_packet(right)
        except ValueError:
            pass
        else:
            raise AssertionError("Unbounded header accepted")
    confirmation=ROOT / "runs/simulation_confirmation_20261023_01/results.json"
    model=ROOT / "runs/simulation_frozen_20261022_01/best.keras"
    real=json.loads((ROOT / "models/segmentation_final_lock.json").read_text())
    manifest={"schema":"simulator_live_observer_v1","scope":"native_carla_observation_only",
        "training_allowed":False,"control_allowed":False,"shared_real_deployment_approved":False,
        "input_shape":[512,512,3],"normalization":"RGB /127.5 -1 exactly once","threshold":.5,
        "checkpoint":model.relative_to(ROOT).as_posix(),"checkpoint_sha256":sha(model),
        "confirmation":confirmation.relative_to(ROOT).as_posix(),"confirmation_sha256":sha(confirmation),
        "base_checkpoint_sha256":real["checkpoint_sha256"]}
    rejected=[]
    with tempfile.TemporaryDirectory(dir=ROOT / "runs",prefix="live-protocol-check-") as directory:
        path=Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest))
        assert load_live_spec(path)[1]["checkpoint"]==model
        try:
            load_simulation_model_spec(path)
        except ValueError:
            rejected.append("live_scope_in_recorded_loader")
        else:
            raise AssertionError("Recorded-only scope was silently broadened")
        for field,value in (("scope","real_camera"),("training_allowed",True),("control_allowed",True),
                            ("shared_real_deployment_approved",True),("threshold",.4),("checkpoint_sha256","0"*64)):
            path.write_text(json.dumps({**manifest,field:value}))
            try:
                load_live_spec(path)
            except ValueError:
                rejected.append(field)
            else:
                raise AssertionError(field)
    try:
        load_live_spec(ROOT / "runs/simulation_confirmation_20261023_01/simulation_model.json")
    except ValueError:
        rejected.append("recorded_scope_in_live_loader")
    else:
        raise AssertionError("Recorded manifest implicitly granted live scope")
    assert protected()==before and "tensorflow" not in sys.modules
    result={"passed":True,"latest_slot_replacements_checked":99,"wall_age_sim_age_pause_and_session_checks":True,
        "packet_identity_and_bounds_checked":True,"rejected_scope_changes":rejected,
        "real_files_unchanged":True,"tensorflow_loaded":False}
    save(ROOT / "setup/live_protocol_verification.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
