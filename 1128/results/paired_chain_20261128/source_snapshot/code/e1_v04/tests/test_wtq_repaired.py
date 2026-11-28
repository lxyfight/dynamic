import unittest,json
from pathlib import Path
from run_vfa_development import ROOT,PACKAGE,episode
from e1_sim.wtq_policy import WTQPolicy
from e1_sim.wtq_audit import audit_wtq
from e1_sim.wtq_compiler import ClosingController
import tempfile

class RepairedWTQTests(unittest.TestCase):
    def test_old_delayed_reservation_is_rejected(self):
        d=json.loads((ROOT/'data/fixtures/wtq_7000_bm_15.json').read_text(encoding='utf8'))
        a=audit_wtq(d['input'],d['trace'],15,1000,PACKAGE['configuration']['truck_capacity'])
        self.assertFalse(a['passed']);self.assertTrue(any('delayed' in e['error'] for e in a['errors']))
    def test_original_failure_repaired_both_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            for batch in ['B1','BM']:
                p=Path(folder)/'run.json'
                r,_=episode(7000,WTQPolicy(T=15,Q=30,candidate_budget=16),batch,p)
                d=json.loads(p.read_text(encoding='utf8'))
                a=audit_wtq(d['input'],d['trace'],15,30,PACKAGE['configuration']['truck_capacity'],batch)
                self.assertTrue(a['passed'],a);self.assertEqual(r['accepted'],r['completed'])
    def test_zero_time_threshold_preserves_unloading(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'run.json';episode(7000,WTQPolicy(T=0,Q=30,candidate_budget=16),'BM',p)
            d=json.loads(p.read_text(encoding='utf8'))
            self.assertTrue(audit_wtq(d['input'],d['trace'],0,30,PACKAGE['configuration']['truck_capacity'])['passed'])
    def test_threshold_validation(self):
        from e1_sim.quantities import InvalidAction
        with self.assertRaises(InvalidAction):ClosingController(1,0,{})

if __name__=='__main__':unittest.main()
