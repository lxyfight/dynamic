"""Check the proposed waiting features at real decision epochs under hidden-suffix deletion."""
from copy import deepcopy
from decimal import Decimal

from e1_sim.online import prepare_online
from high_demand_features import FeaturePolicy
from run_multiscale import ROOT, read, write, digest
from run_wait_counterfactual import pure_pairs
from wait_increment_model import wait_features


class Captured(Exception):
    pass


class AtStation(FeaturePolicy):
    def __init__(self, model, site):
        super().__init__(model=model)
        self.site = site
        self.result = None

    def pick(self, known, ranked, theta):
        plan, record = super().pick(known, ranked, theta)
        if known.now != Decimal(self.site['time']):
            return plan, record
        depart = next(c for _,_,c in ranked if c['plan'] is plan)
        assert digest(plan) == self.site['depart']
        _, matches = pure_pairs(known,ranked,depart)
        truck,station,wait = next(row for row in matches if digest(row[2]['plan']) == self.site['wait'])
        x = wait_features(known,depart,wait,truck,station)
        p = x['offer_probability']
        assert 0 <= x['drone_opportunity'] <= p <= 1
        assert 0 <= x['truck_opportunity'] <= p
        self.result = dict(x=x, candidate_ids=[c['id'] for _,_,c in ranked], original=record['plan'])
        raise Captured


def observe(world, raw, site, model):
    state, driver = prepare_online(raw, world['configuration'], world['network'], full_day=True)
    driver.policy = AtStation(model,site)
    state._record_snapshots = False
    try:
        state.run(decision_hook=driver.on_event,stop_at_terminal=True)
    except Captured:
        return driver.policy.result
    raise AssertionError('Shared decision epoch absent')


def main():
    tmp = ROOT/'tmp'
    prior = tmp/'high_demand_screen_v1'
    model = read(prior/'freeze.json')['model']
    checks = []
    for case in [(440,100,'uniform'),(443,150,'dual')]:
        world = read(prior/'worlds'/(digest(case)+'.json'))
        site = read(tmp/'wait_counterfactual_v1/runs'/('_'.join(map(str,case))+'_depart.json'))['selected'][0]
        t = Decimal(site['time'])
        original = world['input']
        short = deepcopy(original)
        short['orders'] = [o for o in short['orders'] if o['kind']=='D' or Decimal(str(o['release']))<=t]
        short['external_events'] = [e for e in short['external_events'] if e['kind']!='request_reveal'
                                    or Decimal(str(e['at']))<=t]
        removed = len(original['orders'])-len(short['orders'])
        assert removed>0 and original['public_information']==short['public_information']
        full = observe(world,original,site,model)
        assert full==observe(world,short,site,model)
        checks.append(dict(case=case, time=site['time'], deleted=removed, x=full['x'],
                           candidate_count=len(full['candidate_ids'])))
    write(tmp/'wait_learning_v1/information_check.json',checks)
    print([(r['case'],r['deleted'],r['candidate_count']) for r in checks])


if __name__=='__main__':
    main()
