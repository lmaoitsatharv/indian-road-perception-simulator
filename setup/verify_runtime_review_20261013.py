"""Bounded guard and independent-audit regressions; never runs the old controller."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import audit_traffic_rollout as audit_module
import traffic_session as controller

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/review_20261013/runtime_verification'


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.previous_session = controller.SESSION
        self.previous_deadline = controller.DEADLINE
        OUT.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix='lease_', dir=OUT))
        controller.SESSION = self.folder
        self.now = datetime.now(timezone.utc)
        controller.DEADLINE = self.now + timedelta(hours=1)
        self.lease = dict(ordinaryUsageAllowed=True, usage_percentages=[12, 34],
            checked_at_utc=controller.stamp(self.now),
            expires_at_utc=controller.stamp(self.now + timedelta(minutes=20)))
        controller.atomic(self.folder / 'monitor_lease.json', self.lease)

    def tearDown(self):
        controller.SESSION = self.previous_session
        controller.DEADLINE = self.previous_deadline

    def test_healthy_lease(self):
        self.assertEqual(controller.permission(), (None, False))

    def test_user_stop_survives_deadline(self):
        controller.atomic(self.folder/'STOP.json', dict(reason='user_stop'))
        controller.DEADLINE = self.now-timedelta(seconds=1)
        self.assertEqual(controller.permission(), ('user_stop', True))

    def test_malformed_stop_fails_closed(self):
        (self.folder/'STOP.json').write_text('{broken', encoding='utf-8')
        self.assertEqual(controller.permission(), ('invalid_stop_marker', True))

    def test_malformed_pause_fails_closed(self):
        controller.atomic(self.folder/'PAUSE.json', [])
        self.assertEqual(controller.permission(), ('invalid_pause_marker', False))

    def test_deadline_is_final(self):
        controller.DEADLINE = self.now-timedelta(seconds=1)
        self.assertEqual(controller.permission(), ('deadline', True))

    def test_exhausted_usage_pauses(self):
        self.lease['usage_percentages'] = [12, 95]
        controller.atomic(self.folder/'monitor_lease.json', self.lease)
        self.assertEqual(controller.permission(), ('usage_pause', False))

    def test_usage_not_allowed_pauses(self):
        self.lease['ordinaryUsageAllowed'] = False
        controller.atomic(self.folder/'monitor_lease.json', self.lease)
        self.assertEqual(controller.permission(), ('usage_pause', False))

    def test_invalid_leases_rejected(self):
        for fields in [dict(usage_percentages=[]), dict(usage_percentages=[True]),
                dict(usage_percentages=[float('nan')]),
                dict(expires_at_utc=self.now.replace(tzinfo=None).isoformat()),
                dict(expires_at_utc=controller.stamp(self.now+timedelta(days=1))),
                dict(checked_at_utc=controller.stamp(self.now+timedelta(minutes=1))),
                dict(expires_at_utc=controller.stamp(self.now-timedelta(seconds=1)))]:
            with self.subTest(fields=fields):
                payload={**self.lease, **fields}
                (self.folder/'monitor_lease.json').write_text(json.dumps(payload), encoding='utf-8')
                self.assertEqual(controller.permission(), ('monitor_lease_expired', False))

    def test_atomic_json_has_no_bom_or_nan(self):
        path=self.folder/'utf8.json';controller.atomic(path, dict(name='road'))
        self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'))
        before=path.read_bytes()
        with self.assertRaises(ValueError): controller.atomic(path, dict(value=float('nan')))
        self.assertEqual(path.read_bytes(), before)

    def test_job_closes_only_its_own_tree(self):
        with contextlib.redirect_stdout(io.StringIO()): controller.selftest()
        evidence=json.loads((self.folder/'guard_verification.json').read_text())
        self.assertTrue(evidence['owned_root_stopped'])
        self.assertTrue(evidence['owned_child_stopped'])
        self.assertTrue(evidence['unrelated_probe_untouched'])


def fixture():
    spec=dict(name='car_1',kind='car',length_m=4,width_m=2,rear_overhang_m=1)
    ego=np.zeros((4,9));ego[:,0]=[0,.1,.2,.3];ego[:,1]=20;ego[:,8]=[0,.1,.2,.3]
    actors=np.zeros((4,2,3));actors[:,1,0]=2
    return dict(actors=[spec,dict(spec,name='car_2')],ego=ego.tolist(),actor_states=actors.tolist())


class AuditTests(unittest.TestCase):
    def test_known_body_overlap(self):
        data=fixture();ego,actors,specs=audit_module.validated_replay(data)
        self.assertTrue(audit_module.hits(actors[:,0,:],specs[0],actors[:,1,:],specs[1]).all())
        actors[:,1,0]=8
        self.assertFalse(audit_module.hits(actors[:,0,:],specs[0],actors[:,1,:],specs[1]).any())

    def test_pedestrian_box_and_disc_overlap(self):
        spec=fixture()['actors'][0];ped=dict(name='ped',kind='pedestrian',width_m=.5)
        car=np.array([[0.,0.,0.]])
        self.assertTrue(audit_module.hits(np.array([[3.1,0.,0.]]),ped,car,spec)[0])
        self.assertFalse(audit_module.hits(np.array([[3.5,0.,0.]]),ped,car,spec)[0])
        self.assertTrue(audit_module.hits(car,ped,np.array([[.4,0.,0.]]),ped)[0])
        wide=dict(ped,width_m=1)
        self.assertTrue(audit_module.hits(car,wide,np.array([[.6,0.,0.]]),ped)[0])

    def test_corrupted_states_cannot_report_success(self):
        mutations=[lambda d:d['ego'][1].__setitem__(0,0),
            lambda d:d['ego'][1].__setitem__(1,float('nan')),
            lambda d:d['actor_states'][2][0].__setitem__(2,float('inf')),
            lambda d:d['actors'][1].__setitem__('name','car_1'),
            lambda d:d['actors'][0].__setitem__('width_m',-1),
            lambda d:d.__setitem__('actor_states',d['actor_states'][:-1])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data=copy.deepcopy(fixture());mutation(data)
                with self.assertRaises(ValueError): audit_module.validated_replay(data)

    def test_optional_absent_lead_gap_and_actual_replay(self):
        data=fixture()
        data['ego']=[row+[0,0,None,0] for row in data['ego']]
        ego,_,_=audit_module.validated_replay(data)
        self.assertTrue(np.isnan(ego[:,11]).all())
        data['ego'][0][11]=float('nan')
        with self.assertRaises(ValueError): audit_module.validated_replay(data)
        saved=json.loads((ROOT/'runs/traffic_overnight_20261004/mixed_seed42_v6/replay.json').read_text())
        ego,actors,specs=audit_module.validated_replay(saved)
        self.assertEqual(len(ego),len(actors))
        self.assertEqual(len(specs),29)

    def test_audit_preserves_existing_evidence(self):
        folder=Path(tempfile.mkdtemp(prefix='audit_',dir=OUT))
        (folder/'replay.json').write_text(json.dumps(fixture()),encoding='utf-8')
        path=folder/'audit.json'
        with contextlib.redirect_stdout(io.StringIO()): result=audit_module.audit(folder,path)
        self.assertEqual(result['other_actor_overlap_samples'],4)
        self.assertEqual(result['distinct_pair_incidents'],1)
        self.assertEqual(result['ego_overlap_samples'],0)
        original=path.read_bytes()
        with self.assertRaises(FileExistsError): audit_module.audit(folder,path)
        self.assertEqual(path.read_bytes(),original)
        self.assertFalse(original.startswith(b'\xef\xbb\xbf'))

class JobContractTests(unittest.TestCase):
    def test_timeout_validated_before_launch(self):
        valid=dict(id='probe_1',argv=['python','probe.py'],timeout_seconds=60)
        self.assertEqual(controller.validated_job(valid)['timeout_seconds'],60)
        for timeout in [0,-1,True,float('nan'),float('inf'),'60']:
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                controller.validated_job({**valid,'timeout_seconds':timeout})
        self.assertEqual(controller.validated_job({**valid,'timeout_seconds':5000})['timeout_seconds'],3600)

    def test_queue_ids_and_arguments(self):
        valid=dict(id='probe_1',argv=['python','probe.py'])
        for fields in [dict(id='../escape'),dict(id=''),dict(argv=[]),dict(argv=['python',None]),dict(argv=['a\0b'])]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                controller.validated_job({**valid,**fields})


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    protected=ROOT/'setup/traffic_overnight_20261004/STOP.json'
    before=protected.read_bytes()
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if protected.read_bytes()!=before: raise RuntimeError('Historical stop marker changed')
    controller.atomic(OUT/'verification.json',dict(status='passed' if result.wasSuccessful() else 'failed',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        historical_stop_preserved=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
