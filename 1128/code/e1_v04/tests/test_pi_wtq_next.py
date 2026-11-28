import json,unittest
from copy import deepcopy
from e1_sim.runner import DEFAULT_SPEC,execute_case
from e1_sim.pi_mapping import compression_mapping
from e1_sim.quantities import InvalidAction
from run_vfa_development import episode
from e1_sim.wtq_full_day import TimeThresholdPolicy

P=json.loads(DEFAULT_SPEC.read_text(encoding='utf8'))
C={c['id']:c for c in P['cases']}

class NextStageTests(unittest.TestCase):
    def test_compression_witness(self):
        self.assertEqual(execute_case(P,C['E32'])['status'],'PASS')
    def test_compression_cannot_hide_late_delivery(self):
        d=deepcopy(C['E32']['input']);d['orders'][0]['deadline']=104
        with self.assertRaises(InvalidAction):compression_mapping(P['configuration'],d)
    def test_compression_cannot_use_negative_wait(self):
        d=deepcopy(C['E32']['input']);d['compressed_schedule']['wait_until']=101
        with self.assertRaises(InvalidAction):compression_mapping(P['configuration'],d)
    def test_compression_cannot_revise_airborne_commitment(self):
        d=deepcopy(C['E32']['input']);d['drones'][0]['takeoff_at']=99
        with self.assertRaises(InvalidAction):compression_mapping(P['configuration'],d)
    def test_full_day_threshold_both_batch_permissions(self):
        for batch in ['B1','BM']:
            r,_=episode(7000,TimeThresholdPolicy(5,candidate_budget=16),batch)
            self.assertTrue(r['terminal']);self.assertEqual(r['accepted'],r['completed'])
    def test_threshold_reproducibility(self):
        a,_=episode(7001,TimeThresholdPolicy(15,candidate_budget=16),'BM')
        b,_=episode(7001,TimeThresholdPolicy(15,candidate_budget=16),'BM')
        self.assertEqual(a['ledger'],b['ledger']);self.assertEqual(a['accepted'],b['accepted'])

if __name__=='__main__':unittest.main()
