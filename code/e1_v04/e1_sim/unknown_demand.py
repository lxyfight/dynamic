"""Fixed known deliveries and independently random pickup counts and marks."""
from copy import deepcopy
from random import Random
from collections import defaultdict
import hashlib,json
from .information import MODE,validate_public,sample_release_rows
from .multiscale import network,common_i0
from .constraints import eligibility
from .initializer import digest
from .kernel import require

def seed(*items):return int(hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12],16)

def public_law(cities,expected_per_city,end=590):
    p=dict(information_mode=MODE,revelation_end=end,revelation_closed=False,
        demand_model=dict(kind='independent_poisson_per_minute',
          rates_by_city={f'C{i}':[dict(start=1,end=end,rate=expected_per_city/end)] for i in range(1,cities+1)},
          attributes=dict(weight_min=6,weight_max=19,x_radius=10.,y_min=.5,y_max=10.,pickup_deadline_lag=360)))
    validate_public(p);return p

def mark(city,rng,net,cfg,attributes):
    ix=int(city[1:]);stations=[f'L{ix}+',f'L{ix}-'];x=net['stations'][stations[0]][0]
    w=rng.randint(attributes['weight_min'],attributes['weight_max'])
    xy=[round(x+rng.uniform(-attributes['x_radius'],attributes['x_radius']),6),
        round(rng.choice([-1,1])*rng.uniform(attributes['y_min'],attributes['y_max']),6)]
    eligible=[s for s in stations if eligibility(dict(weight=w,xy=xy),net['stations'][s],cfg)['feasible']]
    require(eligible,'INVALID_INSTANCE: mark outside all station service ranges; no resampling')
    return dict(city=city,weight=w,xy=xy,eligible_stations=eligible)

def sample_orders(public,now,rng,net,cfg,prefix='SIM'):
    attr=public['demand_model']['attributes'];rows=sample_release_rows(public,now,rng);result={}
    for j,row in enumerate(rows):
        oid=f'{prefix}_{j:06d}';release=row['release']
        result[oid]=dict(id=oid,kind='P',**mark(row['city'],rng,net,cfg,attr),release=release,
            deadline=release+attr['pickup_deadline_lag'],accepted=None,completed_at=None,available_at=None,
            collected_at=None,station=None,assigned_drone=None)
    return result

def make_instance(cfg,*,cities,deliveries_per_city,expected_pickups_per_city,layout_seed,day_seed,identity,
                  pickup_seed=None,revelation_end=590):
    require(type(deliveries_per_city) is int and deliveries_per_city>0,'positive known delivery count required')
    net=network(cities,layout_seed);public=public_law(cities,expected_pickups_per_city,revelation_end)
    rng=Random(seed('known-D',day_seed));ds=[];drones=[]
    for i in range(1,cities+1):
        for j in range(deliveries_per_city):
            ds.append(dict(id=f'D_C{i}_{j:05d}',kind='D',**mark(f'C{i}',rng,net,cfg,public['demand_model']['attributes']),
                           release=0,deadline=660,accepted=True))
        for side in ['+','-']:
            station=f'L{i}{side}'
            for k in range(cfg['drones_per_station']):
                drones.append(dict(id=f'U{i}{side}{k}',home=station,location=station,activated=False,
                                   onboard_weight=0,takeoff_at=None,active_operation=None))
    plan,h=common_i0(cfg,net,ds,drones)
    pickups=sample_orders(public,0,Random(pickup_seed if pickup_seed is not None else seed('hidden-P',day_seed)),net,cfg,prefix='P')
    raw=deepcopy(plan['replay_input']);raw['orders']+=list(pickups.values());events=defaultdict(list)
    for oid,o in pickups.items():events[o['release']].append(oid)
    raw['external_events']=[dict(at=t,kind='request_reveal',orders=ids) for t,ids in sorted(events.items())]
    raw['public_information']=public;raw['stop_time']=max(5000,revelation_end+1000)
    return dict(id=identity,cities=cities,deliveries_per_city=deliveries_per_city,
        expected_pickups_per_city=expected_pickups_per_city,layout_seed=layout_seed,day_seed=day_seed,
        configuration=deepcopy(cfg),network=net,input=raw,i0_hash=h,input_hash=digest(raw),
        information_mode=MODE,i0_algorithm=plan['algorithm'])
