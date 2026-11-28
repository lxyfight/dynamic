"""Public, seeded multi-scale instance factory and common delivery-only I0.

Independent layout families by split; nested requests within a layout/day.
No rejection sampling or silent adjustment of deadlines.
"""
from collections import defaultdict
from copy import deepcopy
from random import Random
from .constraints import distance,eligibility
from .initializer import digest,lossless,_roads,public_delivery_input,verify_plan
from .quantities import dec
from .kernel import require
from .cost_model import departure_cost
from .online import _new_truck


def network(cities,seed):
    rng=Random(seed);xs=[20.]
    for _ in range(cities-1):xs.append(round(xs[-1]+rng.uniform(15,25),6))
    stations={f'L{i+1}{side}':[x,y] for i,x in enumerate(xs) for side,y in [('+',.25),('-',-.25)]}
    sequence=['depot']+[f'L{i+1}+' for i in range(cities)]+[f'L{i+1}-' for i in range(cities-1,-1,-1)]+['depot']
    arcs=[]
    for a,b in zip(sequence,sequence[1:]):
        turn=a==f'L{cities}+' and b==f'L{cities}-'
        km=10. if turn else 20. if 'depot' in (a,b) else abs(stations[a][0]-stations[b][0])
        arcs.append(dict(**{'from':a,'to':b},km=km,min=10. if turn else km*.6,tolled=not turn))
    return dict(city_x=xs,stations=stations,arcs=arcs)


def common_i0(cfg,net,orders,drones,*,improve=True,improvement_budget=32):
    """Pack D into truck trips, one unload per visited station, singleton sorties.

    Only public mandatory D are read. Full replay and independent accounts are
    required before adopting this common plan. No artificial feasibility claim.
    """
    ds=sorted((o for o in orders if o['kind']=='D'),key=lambda o:(o['deadline'],o['id']))
    problem=public_delivery_input(cfg,net,ds,drones);arrivals,road=_roads(net)
    assignments=[];truck_groups=[]
    for o in ds:
        stations=[s for s in o['eligible_stations'] if eligibility(o,net['stations'][s],cfg)['feasible']]
        require(stations,'I0_CONSTRUCTION_FAILED: no eligible station')
        s=min(stations,key=lambda s:(distance(net['stations'][s],o['xy']),arrivals[s],s))
        if not truck_groups or sum(x[0]['weight'] for x in truck_groups[-1])+o['weight']>cfg['truck_capacity']:truck_groups.append([])
        truck_groups[-1].append((o,s))
    actions=[];trucks=[];ready={};truck_returns={};truck_cost=dec(0)
    for k,group in enumerate(truck_groups):
        tid=f'I0T{k+1:05d}';trucks.append(_new_truck(tid));by=defaultdict(list)
        for o,s in group:by[s].append(o)
        actions.append(dict(at=0,kind='truck_dispatch',actor=tid,delivery_orders=[o['id'] for o,s in group],quantities={o['id']:o['weight'] for o,s in group}))
        delay=dec(0)
        for s in sorted(by,key=arrivals.get):
            at=arrivals[s]+delay;q=sum(o['weight'] for o in by[s]);end=at+dec(cfg['truck_handling_setup'])+dec(cfg['truck_handling_per_kg'])*q
            actions.extend([dict(at=at,kind='truck_unload',actor=tid,station=s,quantities={o['id']:o['weight'] for o in by[s]}),dict(at=end,kind='truck_continue_circuit',actor=tid)])
            for o in by[s]:ready[o['id']]=(s,end,tid)
            delay+=end-at
        truck_returns[tid]=road+delay
        truck_cost+=sum(dec(a['km'])*(dec(cfg['truck_distance_cost'])+(dec(cfg['toll_per_marked_km']) if a['tolled'] else 0)) for a in net['arcs'])+(road+delay)*dec(cfg['truck_time_cost'])
    free={u['id']:dec(0) for u in drones};used=set();air_cost=dec(0);stop=road+sum(dec(cfg['truck_handling_setup'])+dec(cfg['truck_handling_per_kg'])*o['weight'] for o in ds)+1
    for o in ds:
        s,r,tid=ready[o['id']];km=distance(net['stations'][s],o['xy']);f=km/dec(cfg['drone_speed_km_min']);handling=dec(cfg['drone_handling_setup_per_request'])+dec(cfg['drone_handling_per_kg'])*o['weight']
        choices=[]
        for u in drones:
            if u['home']!=s:continue
            load=max(r,free[u['id']]);takeoff=load+handling;complete=takeoff+f+dec(cfg['customer_service_per_request']);returned=complete+f
            if complete<=dec(o['deadline']):choices.append((departure_cost(cfg, u['id'] in used),returned,u['id'],load,takeoff))
        require(choices,'I0_CONSTRUCTION_FAILED: singleton drone schedule; not proof of infeasibility')
        activation,returned,uid,load,takeoff=min(choices);used.add(uid);free[uid]=returned;stop=max(stop,returned)
        air_cost+=activation+2*km*dec(cfg['drone_distance_cost'])
        actions.extend([dict(at=load,kind='drone_load',actor=uid,orders=[o['id']]),dict(at=takeoff,kind='drone_sortie',actor=uid,visit_order=[o['id']])])
        assignments.append(dict(order=o['id'],station=s,drone=uid,truck=tid,quantity=o['weight'],service_complete=returned-f,truck_return=truck_returns[tid]))
    raw=dict(start_time=0,stop_time=stop,orders=deepcopy(ds),trucks=trucks,drones=deepcopy(drones),inventory=[dict(order=o['id'],lot=o['id'],weight=o['weight'],state='depot',holder='depot') for o in ds],actions=actions,external_events=[],policy=None,candidates=[],public_information=dict(future_count=0,revelation_closed=True))
    plan=lossless(dict(algorithm='I0-packed-D-only-v1',replay_input=raw,assignments=assignments,predicted_net=sum(dec(cfg['revenue_per_kg'])*o['weight'] for o in ds)-truck_cost-air_cost))
    # Existing verify_plan also checks predicted accounting and all obligations.
    verify_plan(problem,plan,record_snapshots=False)
    if improve:
        from .initial_enhancement import improve_i0
        return improve_i0(cfg,net,ds,drones,plan,improvement_budget)
    return plan,digest(plan)


def make_instance(cfg,*,cities,n,p,layout_seed,day_seed,identity,small_total=None):
    net=network(cities,layout_seed);rng=Random(day_seed);orders=[];city_counts={};drones=[]
    small_types=None
    if small_total is not None:
        small_types=list(range(small_total));rng.shuffle(small_types);small_types=set(small_types[:round(small_total*p)])
    for city,x in enumerate(net['city_x'],1):
        # Separate nested ranks for the first and second ten requests.
        ranks=[]
        for block in range(2):
            blockrank=list(range(10));rng.shuffle(blockrank);ranks.append(blockrank)
        count=n if small_total is None else sum(i%cities==city-1 for i in range(small_total))
        picks=0
        for j in range(20 if small_total is None else count):
            xy=[round(x+rng.uniform(-10,10),6),round(rng.choice([-1,1])*rng.uniform(.5,10),6)]
            weight=rng.randint(6,19);release=rng.randint(1,590)
            if j>=count:continue
            isp=(ranks[j//10].index(j%10)<round(10*p)) if small_total is None else (j*cities+city-1 in small_types)
            kind='P' if isp else 'D';picks+=isp
            stations=[s for s in [f'L{city}+',f'L{city}-'] if eligibility(dict(weight=weight,xy=xy),net['stations'][s],cfg)['feasible']]
            require(stations,'INVALID_INSTANCE: no eligible station')
            orders.append(dict(id=f'{kind}C{city:02d}R{j+1:02d}',kind=kind,weight=weight,release=release if isp else 0,deadline=release+360 if isp else 660,city=f'C{city}',xy=xy,eligible_stations=stations,accepted=None if isp else True))
        city_counts[f'C{city}']=picks
        for side in ['+','-']:
            s=f'L{city}{side}'
            for k in range(cfg['drones_per_station']):drones.append(dict(id=f'U{city}{side}{k}',home=s,location=s,activated=False,onboard_weight=0,takeoff_at=None,active_operation=None))
    plan,h=common_i0(cfg,net,orders,drones);raw=deepcopy(plan['replay_input']);events=defaultdict(list)
    raw['orders']+=deepcopy([o for o in orders if o['kind']=='P'])
    for o in orders:
        if o['kind']=='P':events[o['release']].append(o['id'])
    raw['external_events']=[dict(at=t,kind='request_reveal',orders=ids) for t,ids in sorted(events.items())]
    raw['public_information']=dict(future_count=sum(city_counts.values()),city_pickup_totals=city_counts,revelation_end=590,revelation_closed=False)
    raw['stop_time']=5000
    return dict(id=identity,cities=cities,n=n,p=p,layout_seed=layout_seed,day_seed=day_seed,configuration=cfg,network=net,input=raw,i0_hash=h,input_hash=digest(raw),i0_algorithm=plan['algorithm'])
