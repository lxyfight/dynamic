from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from audit_paired_chain_campaign import Checks,audit_campaign,audit_pair
from e1_sim.initializer import digest
from e1_sim.kernel import COSTS


def sample():
    site=dict(id='example',at=10)
    ledger=dict(revenue=10,**{k:0 for k in COSTS},net=10)
    branch=dict(net=10,ledger=ledger,terminal=True,ledger_audited=True,accepted=0,accepted_ids=[],
        trips=1,trace='trace',suffix_hash=digest({}),continuation_search={'solver':'gurobi'},actual_first_departure_delay=0)
    row=dict(site=site,replica=-1,online_known_hash='known',future_count=0,suffix_hash=digest({}),
        branches={h:deepcopy(branch) for h in ('0','2','5','10')},comparisons={})
    for h in ('2','5','10'):
        row['comparisons'][h]=dict(net_gain=0,ledger_delta={k:0 for k in ledger},extra_time_cost=0,
            trips_saved=0,accepted_delta=0,added_accepts=[],lost_accepts=[])
    return row,dict(future_orders=[]),site,branch


class CampaignAuditTests(unittest.TestCase):
    def test_incomplete_campaign_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result=audit_campaign(Path(directory))
            self.assertFalse(result['successful'])
            self.assertEqual(result['status'],'incomplete')
            self.assertEqual(result['check_count'],0)

    def test_valid_pair_and_accounting_tampering(self):
        row,meta,site,reference=sample();checks=Checks()
        audit_pair(row,meta,site,-1,'known',reference,checks)
        self.assertFalse(checks.failures)
        row['branches']['2']['ledger']['truck_time']=3
        checks=Checks();audit_pair(row,meta,site,-1,'known',reference,checks)
        self.assertIn('ledger_net_arithmetic',{c['check'] for c in checks.failures})

    def test_suffix_and_public_state_tampering_are_detected(self):
        row,meta,site,reference=sample();checks=Checks()
        row['online_known_hash']='other';row['branches']['5']['suffix_hash']='other'
        audit_pair(row,meta,site,-1,'known',reference,checks)
        failed={c['check'] for c in checks.failures}
        self.assertIn('same_predecision_known_state',failed)
        self.assertIn('same_suffix_four_branches',failed)


if __name__=='__main__':unittest.main()
