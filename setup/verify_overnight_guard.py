"""Verify deadline/lease behavior and that only a session-owned test child is stopped."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import overnight_idd_controller as controller

folder = ROOT / "setup/overnight_20261003/guard_verification"
folder.mkdir(exist_ok=True)
controller.SESSION = folder
controller.DEADLINE = time.time() + 3600
controller.write(folder / "monitor_lease.json", {"expires_at_utc": datetime.fromtimestamp(time.time() + 60, timezone.utc).isoformat()})
assert controller.stop_reason() is None
controller.DEADLINE = time.time() - 1
assert controller.stop_reason() == "7_am_ist_deadline"
controller.DEADLINE = time.time() + 3600
controller.write(folder / "monitor_lease.json", {"expires_at_utc": datetime.fromtimestamp(time.time() - 1, timezone.utc).isoformat()})
assert controller.stop_reason() == "monitor_lease_expired"
controller.write(folder / "STOP.json", {"reason": "usage_near_limit"})
assert controller.stop_reason() == "usage_near_limit"
owned_child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              env={**os.environ, "IDD_OVERNIGHT_SESSION": controller.TAG}, start_new_session=True)
other_child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
try:
    assert controller.owned(owned_child.pid)
    assert not controller.owned(other_child.pid)
    controller.terminate_owned(owned_child.pid)
    owned_child.wait(timeout=5)
    controller.terminate_owned(other_child.pid)
    assert other_child.poll() is None
finally:
    if owned_child.poll() is None:
        owned_child.terminate()
        owned_child.wait()
    other_child.terminate()
    other_child.wait()
result = {"status": "passed", "deadline_enforced": True, "lease_expiry_stops": True,
          "usage_stop_marker_supported": True, "only_owned_process_group_terminated": True}
(folder.parent / "guard_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
