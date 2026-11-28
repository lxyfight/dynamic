"""Bank separation, immutable protocol and primary-only capture checks."""
from pathlib import Path
import tempfile
import unittest

from run_paired_chain_campaign import (SPLITS,RESERVED_TEST,specs,case_id,
    protocol_for,lock_protocol,PrimaryCapturePolicy)
from test_paired_wait_branches import post_unload_state


class PairedChainCampaignTests(unittest.TestCase):
    def test_declared_splits_are_disjoint_and_preserve_final_test(self):
        self.assertEqual([len(specs(s)) for s in SPLITS],[8,4,16])
        banks=[{r['layout_id'] for r in specs(s)} for s in SPLITS]
        self.assertTrue(all(not b&RESERVED_TEST for b in banks))
        self.assertEqual(sum(map(len,banks)),len(set.union(*banks)))
        self.assertEqual([SPLITS[s]['samples'] for s in SPLITS],[16,24,32])
        self.assertEqual(len(banks[2]),8)
        self.assertTrue(all(r['day_id']==(2 if r['layout_id']%2 else 3) for r in specs('dev')))
        first=specs('train')[0]
        self.assertNotEqual(case_id(first,1),case_id(first,2))

    def test_protocol_refuses_sample_seed_or_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=protocol_for('train',16,20261128)
            lock_protocol(directory,p,create=True)
            lock_protocol(directory,p,create=False)
            with self.assertRaises(ValueError):lock_protocol(directory,dict(p,samples=17),create=False)
            with self.assertRaises(ValueError):lock_protocol(directory,dict(p,seed=18),create=False)
            with self.assertRaises(ValueError):lock_protocol(Path(directory)/'missing',p,create=False)

    def test_capture_keeps_only_real_post_unload_primary(self):
        # Physical test state has valid unknown-demand forecasts, so the actual
        # none chooser/replayer can be used without inventing a future stream.
        known,base=post_unload_state()
        with tempfile.TemporaryDirectory() as directory:
            p=PrimaryCapturePolicy(directory,'test',{'test':True})
            p.action_solver='bounded';p.enhancement_budget=0
            p.environment=known
            p.choose(known,[base])
            self.assertEqual(len(p.sites),1)
            self.assertEqual(p.sites[0]['qualification']['stratum'],'post_unload_primary')
            known.trace=[e for e in known.trace if e['kind']!='truck_unload_done']
            p.choose(known,[base])
            self.assertEqual(len(p.sites),1)
            self.assertIn('not_just_unloaded',p.offer_audit[-1]['trucks'][0]['qualification']['reasons'])


if __name__=='__main__':unittest.main()
