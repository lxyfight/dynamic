from copy import deepcopy
import json, subprocess, tempfile, unittest
from pathlib import Path
from e1_sim.cost_model import current_configuration, fee_key
from e1_sim.quantities import InvalidAction, dec
from e1_sim.kernel import Replay
from e1_sim.runner import build_replay
from e1_sim.initializer import construct_i0
from e1_sim.audit import audit_ledger
from e1_sim.online import prepare_online, known_clone, remaining_plan, evaluate_plan, MyopicPolicy
from e1_sim.online_examples import make_world
from e1_sim.comparators import Comparator
from e1_sim.pi_adapter import PI_PYTHON
from test_execution_and_i0 import problem, PACKAGE, CASES

CFG=current_configuration(PACKAGE['configuration'])
NET=PACKAGE['networks']['fixed_two_city']

class SortieCostTests(unittest.TestCase):
    def test_schema_and_production_default(self):
        from run_multiscale import CFG as production
        self.assertEqual(production,CFG)
        self.assertEqual(CFG['drone_sortie_cost'],25)
        for bad in ({},dict(CFG,drone_first_use_cost=45),dict(CFG,drone_sortie_cost=-1),dict(CFG,drone_sortie_cost=float('nan'))):
            with self.assertRaises(InvalidAction):fee_key(bad)

    def plan(self,fee=25):
        p=problem(3);p['configuration']=current_configuration(p['configuration'],fee)
        result=construct_i0(p)
        self.assertEqual(result['status'],'VERIFIED',result)
        return p,result

    def test_same_drone_three_sorties_charge_three_times(self):
        p,r=self.plan()
        self.assertEqual(r['validation']['ledger']['drone_activation'],75)
        self.assertEqual(len({a['drone'] for a in r['plan']['assignments']}),1)

    def test_fee_sensitivity_analytic_three_flights(self):
        for fee in (0,10,15,25,35,45):
            p,r=self.plan(fee)
            self.assertEqual(r['validation']['ledger']['drone_activation'],3*fee)
            self.assertAlmostEqual(r['validation']['ledger']['net'],self.plan(0)[1]['validation']['ledger']['net']-3*fee)

    def test_loading_without_takeoff_is_free(self):
        p=problem();p['configuration']=current_configuration(p['configuration']);r=construct_i0(p)
        raw=deepcopy(r['plan']['replay_input'])
        raw['actions']=[a for a in raw['actions'] if a['kind']!='drone_sortie']
        state,_=prepare_online(raw,CFG,NET);state.run()
        self.assertEqual(state.ledger['drone_activation'],0)
        self.assertEqual(audit_ledger(CFG,NET,raw,state.trace)['ledger']['drone_activation'],0)

    def test_mixed_sortie_once_and_probe_no_charge(self):
        raw=deepcopy(CASES['E30']['input']);raw.update(actions=[],candidates=[])
        state=Replay(CFG,NET,raw)
        self.assertTrue(state.probe(dict(operation='drone_sortie',actor='U1',visit_order=['D1','P1']))['accepted'])
        self.assertEqual(state.ledger['drone_activation'],0)
        state.apply(dict(kind='drone_sortie',actor='U1',visit_order=['D1','P1']))
        self.assertEqual(state.ledger['drone_activation'],25)
        with self.assertRaises(InvalidAction):state.apply(dict(kind='drone_sortie',actor='U1',visit_order=['D1']))
        self.assertEqual(state.ledger['drone_activation'],25)
        audit=audit_ledger(CFG,NET,raw,state.trace)
        self.assertFalse(audit['errors']);self.assertEqual(audit['ledger']['drone_activation'],25)

    def test_airborne_remaining_plan_does_not_charge_again(self):
        p=problem();p['configuration']=current_configuration(p['configuration']);r=construct_i0(p)
        state,driver=prepare_online(r['plan']['replay_input'],CFG,NET,full_day=True)
        takeoff=next(dec(a['at']) for a in r['plan']['replay_input']['actions'] if a['kind']=='drone_sortie')
        state.full_day=False
        state.stop=takeoff+dec("0.001")
        state.run()
        state.stop=dec(r['plan']['replay_input']['stop_time'])
        known=known_clone(state)
        for _ in range(2):
            result=evaluate_plan(known,remaining_plan(known))
            self.assertTrue(result['feasible'],result)
            self.assertEqual(result['state'].ledger['drone_activation'],25)
        self.assertEqual(state.ledger['drone_activation'],25)

    def test_online_comparators_whole_day_accounting(self):
        from recovery_opportunity import RecoveryPolicy
        for policy in (MyopicPolicy(),RecoveryPolicy(0),RecoveryPolicy(36),Comparator('Greedy'),Comparator('RH'),Comparator('Rollout',rollout_h=1,rollout_seconds=.05)):
            from e1_sim.unknown_demand import make_instance
            x=make_instance(CFG,cities=2,deliveries_per_city=2,expected_pickups_per_city=1,layout_seed=722,day_seed=901,identity='sortie-smoke',pickup_seed=5)
            raw=x['input']
            state,driver=prepare_online(raw,CFG,x['network'],full_day=True);driver.policy=policy
            state.run(decision_hook=driver.on_event,stop_at_terminal=True)
            self.assertTrue(state.is_terminal())
            count=sum(e['kind']=='drone_takeoff' for e in state.trace)
            self.assertEqual(state.ledger['drone_activation'],25*count)
            initial=deepcopy(state.initial);initial['stop_time']=state.now
            audit=audit_ledger(CFG,state.network,initial,state.trace)
            self.assertFalse(audit['errors'])
            for k,v in state.ledger.items():self.assertAlmostEqual(float(v),float(audit['ledger'][k]))

    def test_pi_requires_three_paid_sorties(self):
        # Three 10kg deliveries at the same station, capacity 10kg: every
        # feasible cover needs three sorties even if one drone serves them all.
        p,r=self.plan();cfg=dict(CFG,drone_capacity=10)
        instance=dict(id='sortie-cost-bound',configuration=cfg,network=NET,input=r['plan']['replay_input'])
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'input.json';target=Path(tmp)/'out.json'
            solver=Path(__file__).resolve().parents[1]/'e1_sim/pi_route_bound.py'
            bounds=[]
            for fee in (0,25):
                instance['configuration']['drone_sortie_cost']=fee
                source.write_text(json.dumps(instance),encoding='utf8')
                subprocess.run([str(PI_PYTHON),str(solver),str(source),str(target),'10'],check=True,capture_output=True)
                bounds.append(json.loads(target.read_text())['upper_bound'])
            self.assertAlmostEqual(bounds[0]-bounds[1],75,places=5)
            self.assertGreaterEqual(bounds[1]+1e-6,r['validation']['ledger']['net'])
