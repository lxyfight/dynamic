"""Paired workload/economic diagnosis. No future requests enter the policy.

The dated study uses city-specific D random streams so D4 is a prefix of D10
and D20. Historical studies and their factories remain unchanged.
"""
from copy import deepcopy
from collections import defaultdict
from random import Random
from math import ceil
from e1_sim.unknown_demand import public_law, mark, sample_orders, seed
from e1_sim.multiscale import network, common_i0
from e1_sim.initializer import digest
from e1_sim.information import MODE, validate_public
from e1_sim.kernel import require


def instance(cfg, *, cities, deliveries_per_city, expected_pickups_per_city,
             layout_seed, day_seed, pickup_seed, identity, timing='uniform',
             deadline_lag=360, rejection_factor=1., i0_budget=32):
    cfg = deepcopy(cfg)
    cfg['rejection_per_kg'] *= rejection_factor
    net = network(cities, layout_seed)
    public = public_law(cities, expected_pickups_per_city)
    # Draw the same P realization before transforming release times. Compressing
    # two adjacent independent Poisson minutes preserves the public Poisson law.
    pickups = sample_orders(public, 0, Random(pickup_seed), net, cfg, prefix='P')
    require(timing in ('uniform', 'early', 'late'), 'unknown arrival regime')
    for order in pickups.values():
        if timing != 'uniform':
            order['release'] = ceil(order['release']/2)+(295 if timing == 'late' else 0)
        order['deadline'] = order['release']+deadline_lag
    if timing != 'uniform':
        lo, hi = (1, 295) if timing == 'early' else (296, 590)
        for city in public['demand_model']['rates_by_city']:
            public['demand_model']['rates_by_city'][city] = [dict(start=lo,end=hi,rate=expected_pickups_per_city/295)]
    public['demand_model']['attributes']['pickup_deadline_lag'] = deadline_lag
    validate_public(public)
    ds, drones = [], []
    for i in range(1, cities+1):
        rng = Random(seed('profit-diagnosis-D-v1', day_seed, i))
        for j in range(deliveries_per_city):
            ds.append(dict(id=f'D_C{i}_{j:05d}',kind='D',
                **mark(f'C{i}',rng,net,cfg,public['demand_model']['attributes']),
                release=0,deadline=660,accepted=True))
        for side in ('+', '-'):
            for k in range(cfg['drones_per_station']):
                station=f'L{i}{side}'
                drones.append(dict(id=f'U{i}{side}{k}',home=station,location=station,
                    activated=False,onboard_weight=0,takeoff_at=None,active_operation=None))
    plan, h = common_i0(cfg,net,ds,drones,improvement_budget=i0_budget)
    raw=deepcopy(plan['replay_input']); raw['orders'] += list(pickups.values())
    events=defaultdict(list)
    for oid,o in pickups.items(): events[o['release']].append(oid)
    raw['external_events']=[dict(at=t,kind='request_reveal',orders=ids) for t,ids in sorted(events.items())]
    raw['public_information']=public;raw['stop_time']=5000
    return dict(id=identity,configuration=cfg,network=net,input=raw,i0_hash=h,
        input_hash=digest(raw),information_mode=MODE,cities=cities,
        delivery_revenue=sum(o['weight'] for o in ds)*cfg['revenue_per_kg'],
        i0_net=float(plan['predicted_net']),i0_improvement=plan['improvement'],
        realized_pickups=len(pickups),rejection_all=sum(o['weight'] for o in pickups.values())*cfg['rejection_per_kg'])


def specification(split,c,l,d,D=20,mu=8,**kwargs):
    return dict(cities=c,deliveries_per_city=D,expected_pickups_per_city=mu,
        layout_seed=seed('profit-v1',split,c,l,'layout'),
        day_seed=seed('profit-v1',split,c,l,d,'D'),
        pickup_seed=seed('profit-v1',split,c,l,d,mu,'P'),
        identity=f'{split}_c{c}_l{l}_d{d}_D{D}_mu{mu}',**kwargs)
