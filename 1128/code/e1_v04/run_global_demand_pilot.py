"""Paired development experiment with exact network totals and marked arrivals."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import floor
from random import Random
from statistics import mean
from pathlib import Path
import hashlib

from run_multiscale import CFG, ROOT, read, write
from e1_sim.unknown_demand import public_law, seed
from e1_sim.information import expected_arrivals, validate_public
from e1_sim.p2_arrivals import point
from e1_sim.constraints import eligibility
from e1_sim.multiscale import network, common_i0
from e1_sim.initializer import digest
from recovery_opportunity import RecoveryPolicy, run

OUT = ROOT / 'tmp/global_demand_station_review_v1'
TOTALS = (30, 40)
SHARES = (.2, .4)
BANDS = ((1, 150, 1), (151, 350, 3), (351, 590, 2))
POLICIES = [('fixed', 0), ('adaptive', 0), ('fixed', 28), ('adaptive', 28)]


def arrival_bands(temporal='peak'):
    """Separate temporal shape from the uniform/dual spatial pattern."""
    if temporal == 'peak':
        return BANDS
    if temporal == 'flat':
        return ((1, 590, 1),)
    raise ValueError('temporal must be peak or flat')


def release_slots(rng, count, temporal='peak'):
    weights = [w for lo, hi, w in arrival_bands(temporal) for _ in range(lo, hi+1)]
    return sorted(rng.choices(range(1, 591), weights, k=count))


def historical_law(net, pattern):
    """Synthetic historical days independent of evaluated N, share and release stream."""
    rng = Random(seed('global-history-v1',net['city_x'],pattern))
    counts = Counter()
    days = 160
    for _ in range(days):
        n = round(rng.choice(TOTALS)*rng.choice(SHARES))
        for t in release_slots(rng, n):
            city, _ = point(rng, net, pattern)
            counts[city, next(lo for lo,hi,_ in BANDS if lo<=t<=hi)] += 1
    law = public_law(len(net['city_x']), 1)
    for city in law['demand_model']['rates_by_city']:
        law['demand_model']['rates_by_city'][city] = [
            dict(start=lo,end=hi,rate=counts[city,lo]/(days*(hi-lo+1))) for lo,hi,_ in BANDS]
    validate_public(law)
    return law


def make_world(block, cities, total, share, pattern):
    net = network(cities, seed('global-layout-v1',block,cities))
    law = historical_law(net, pattern)
    n_p = round(total*share)
    times = release_slots(Random(seed('global-releases-v1',block, cities,total,share)),n_p)
    ds, ps = [], []
    for kind, size in [('D', total-n_p), ('P', n_p)]:
        marks = Random(seed('global-marks-v1',block,cities,kind,pattern))
        weights = Random(seed('global-weight-v1',block,cities,kind))
        for j in range(size):
            city,xy = point(marks,net,pattern)
            order = dict(id=f'{kind}{j:05}',kind=kind,weight=weights.randint(6,19),xy=xy,city=city)
            order['eligible_stations'] = [s for s,coord in net['stations'].items()
                if s[1:-1]==city[1:] and eligibility(order,coord,CFG)['feasible']]
            assert order['eligible_stations']
            if kind=='D':
                ds.append(dict(order,release=0,deadline=660,accepted=True))
            else:
                ps.append(dict(order,release=times[j],deadline=times[j]+360,accepted=None,
                    completed_at=None,available_at=None,collected_at=None,station=None,assigned_drone=None))
    drones = [dict(id=f'U{s}_{k}',home=s,location=s,activated=False,onboard_weight=0,
                   takeoff_at=None,active_operation=None)
              for s in net['stations'] for k in range(CFG['drones_per_station'])]
    plan, i0 = common_i0(CFG,net,ds,drones)
    raw = deepcopy(plan['replay_input'])
    raw['orders'] += ps
    releases = defaultdict(list)
    for order in ps:
        releases[order['release']].append(order['id'])
    raw.update(external_events=[dict(at=t,kind='request_reveal',orders=ids) for t,ids in sorted(releases.items())],
               public_information=law,stop_time=5000)
    return dict(configuration=CFG,network=net,input=raw,input_hash=digest(raw),i0_hash=i0)


class ReviewPolicy(RecoveryPolicy):
    def __init__(self, theta, mode):
        super().__init__(theta)
        self.intervals = Counter()
        self.offered = self.selected = 0
        if mode=='adaptive':
            self.review_interval = self.interval

    def interval(self, known, station):
        # Either station in the city's local pair can cover this service strip.
        city = 'C'+station[1:-1]
        for h in range(2,11):
            if expected_arrivals(known.public,city,known.now,known.now+h)>=.05:
                break
        self.intervals[h] += 1
        return h

    def choose(self, known, evaluated):
        ranked = self.rank(known,evaluated)
        plan, record = self.pick(known,ranked,self.theta)
        def new_review(p):
            return any(float(t)>float(known.now) and t!=getattr(known,'_wait_reviews',{}).get(k)
                       for k,t in p.get('wait_reviews',{}).items())
        self.offered += any(new_review(c['plan']) for _,_,c in ranked)
        self.selected += new_review(plan)
        self.records.append(record)
        return next(c for _,_,c in ranked if c['plan'] is plan)


def operation_metrics(state, driver, policy):
    at_station = {}
    idle = 0.
    trips = sum(e['kind']=='truck_dispatched' for e in state.trace)
    for e in state.trace:
        actor=e.get('actor')
        if e['kind']=='truck_arrived' and e['destination']!='depot':
            at_station[actor]=float(e['at'])
        elif e['kind'] in ('truck_load_started','truck_unload_started'):
            at_station[actor]+=float(e['end'])-float(e['at'])
        elif e['kind']=='road_started' and e['source']!='depot':
            idle+=float(e['at'])-at_station.pop(actor)
    assert not at_station and idle>=-1e-7
    return dict(station_idle_minutes=max(0,idle),trips=trips,new_wait_chosen=policy.selected,
        decisions_with_new_wait=policy.offered,review_candidate_intervals=dict(policy.intervals))


def job(spec):
    block,c,total,share,pattern=spec
    path=OUT/'runs'/(digest(spec)+'.json')
    if path.exists():
        return read(path)
    world=make_world(*spec)
    rows=[]
    for mode,theta in POLICIES:
        row=run(world,theta,policy_override=ReviewPolicy(theta,mode),metrics=operation_metrics)
        rows.append(dict(row,mode=mode,theta=theta))
    data=dict(spec=spec,results=rows,input_hash=world['input_hash'],i0_hash=world['i0_hash'],
        public_hash=digest(world['input']['public_information']),
        times_weights=digest([(o['id'],o['weight'],o['release']) for o in world['input']['orders']]),
        city_counts=dict(Counter(o['city'] for o in world['input']['orders'])))
    write(path,data)
    return data


def main():
    assert CFG['truck_handling_setup']==0
    cases=[(b,2,n,s,p) for b in (0,1,2) for n in TOTALS for s in SHARES for p in ('uniform','dual')]
    protocol=dict(cases=cases,policies=POLICIES,review='fixed 5 or clipped 2..10, target .05 historical arrivals',
        bands=BANDS,setup=0,theta_origin='legacy transfer coefficient, no retraining or test selection',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [Path(__file__),Path(__file__).with_name('recovery_opportunity.py'),*Path(__file__).with_name('e1_sim').glob('*.py')]})
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==__import__('json').loads(__import__('json').dumps(protocol))
    write(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=6) as pool:
        data=list(pool.map(job,cases))
    for r in data:
        b,c,n,s,p=r['spec']
        paired=[v for v in data if v['spec'][:4]==r['spec'][:4]]
        assert len(set(v['times_weights'] for v in paired))==1
        others=[v for v in data if v['spec'][:2]==[b,c] and v['spec'][-1]==p] if isinstance(r['spec'],list) else [v for v in data if tuple(v['spec'][:2])==(b,c) and v['spec'][-1]==p]
        assert len({v['public_hash'] for v in others})==1
    summary={}
    for p in ('uniform','dual','both'):
        group=[r for r in data if p=='both' or r['spec'][-1]==p]
        summary[p]={}
        for j,(mode,theta) in enumerate(POLICIES):
            rows=[r['results'][j] for r in group]
            summary[p][f'{mode}_{theta}']=dict(net=mean(x['net'] for x in rows),
                accepted=mean(x['accepted'] for x in rows),idle=mean(x['operation_metrics']['station_idle_minutes'] for x in rows),
                wait_selected=sum(x['operation_metrics']['new_wait_chosen'] for x in rows),
                wait_offered=sum(x['operation_metrics']['decisions_with_new_wait'] for x in rows))
    write(OUT/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
