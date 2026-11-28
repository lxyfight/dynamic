"""Candidate-dependent features measurable from revealed tasks and public law only."""
from .information import validate_public,closed,expected_arrivals

BASE=['constant','time_fraction','new_admissions_per_station','new_trips_per_station',
      'planned_truck_time','known_ready_weight','known_inbound_weight','minimum_delivery_slack',
      'maximum_pickup_lateness','station_time','station_capacity_time','drone_tail_availability']
EXTRA=['expected_remaining_per_city','arrival_exposure','demand_capacity_time','demand_drone_availability']

def names(group):
    if group not in ('resource','resource_demand'):raise ValueError('unknown resource feature group')
    return BASE+(EXTRA if group=='resource_demand' else [])

def resource_windows(known,candidate):
    """Idle station intervals and physical unoccupied capacity; future reservations
    may still restrict use. This is an opportunity proxy, never a feasibility guarantee.
    Drone tail availability starts after its last known planned operation.
    """
    now=float(known.now);H=float(known.public['revelation_end']);Q=float(known.cfg['truck_capacity'])
    trucks={tid:dict(location=t['location'],q=float(t['onboard_weight']),busy=t['active_operation'] is not None)
            for tid,t in known.trucks.items()}
    drone_end={uid:max(now,float((u['active_operation'] or {}).get('end',now))) for uid,u in known.drones.items()}
    windows=[];last=now
    payload={a['actor']:sum(a.get('quantities',{}).values()) if 'quantities' in a else
             sum(known.orders[o]['weight'] for o in a.get('delivery_orders',[]))
             for a in candidate['plan']['actions'] if a['kind']=='truck_dispatch'}
    for e in candidate['trace']+ [{'at':H,'kind':'feature_end'}]:
        at=min(H,max(now,float(e['at'])))
        if at>last:
            for tid,t in trucks.items():
                if t['location'] in known.network['stations'] and not t['busy']:
                    windows.append((t['location'],last,at,max(0.,Q-t['q'])/Q))
            last=at
        if float(e['at'])>H:continue
        uid=e.get('actor');kind=e['kind']
        if uid in drone_end:drone_end[uid]=max(drone_end[uid],float(e.get('end',e['at'])),float(e['at']))
        if kind=='truck_dispatched':trucks[uid]=dict(location='road',q=payload.get(uid,0.),busy=True)
        if uid not in trucks:continue
        t=trucks[uid]
        if kind=='truck_arrived':
            t.update(location=e['destination'],busy=False)
            if e['destination']=='depot':t['q']=0.
        elif kind=='road_started':t.update(location='road',busy=True)
        elif kind=='truck_unload_started':t['busy']=True
        elif kind=='truck_load_started':t['busy']=True
        elif kind=='truck_unload_done':t['q']-=sum(e['quantities'].values());t['busy']=False
        elif kind=='truck_load_done':t['q']+=sum(known.orders[o]['weight'] for o in e['orders']);t['busy']=False
    available={u:max(0.,H-max(now,end)) for u,end in drone_end.items()}
    return windows,available

def vector(known,candidate,group):
    names(group)  # Reject legacy group identifiers even while revelation is open.
    validate_public(known.public)
    if closed(known.public):return [0.]*len(names(group))
    now=float(known.now);H=max(1.,float(known.public['revelation_end']));S=max(1,len(known.network['stations']))
    C=max(1,len(known.public['demand_model']['rates_by_city']));Q=float(known.cfg['truck_capacity'])
    finished=candidate['state'];plan=candidate['plan']
    pickups=[o for o in finished.orders.values() if o['kind']=='P' and o['accepted'] is True and float(o['completed_at'])>now]
    deliveries=[o for o in finished.orders.values() if o['kind']=='D' and float(o['completed_at'])>now]
    ready=sum(o['weight'] for o in pickups if o['available_at'] is not None and float(o['available_at'])<=now)/Q
    inbound=sum(o['weight'] for o in pickups if o['available_at'] is None or float(o['available_at'])>now)/Q
    windows,available=resource_windows(known,candidate)
    dwell=sum(b-a for s,a,b,q in windows)/(H*S);capacity=sum((b-a)*q for s,a,b,q in windows)/(H*S)
    idle=sum(available.values())/(H*max(1,len(known.drones)))
    x=[1.,now/H,sum(plan['decisions'].values())/S,len(plan['new_trucks'])/S,
       float(finished.ledger['truck_time']-known.ledger['truck_time'])/float(known.cfg['truck_time_cost'])/H/S,
       ready,inbound,min((max(0.,float(o['deadline'])-float(o['completed_at']))/H for o in deliveries),default=0.),
       max((max(0.,float(o['completed_at'])-float(o['deadline']))/H for o in pickups),default=0.),dwell,capacity,idle]
    if group=='resource_demand':
        mu={c:expected_arrivals(known.public,c,now) for c in known.public['demand_model']['rates_by_city']}
        # Overlapping trucks contribute separate capacity opportunities, not a
        # count of distinct future customers. No claimed guaranteed saving.
        exposure=sum(expected_arrivals(known.public,'C'+s[1:-1],a,b) for s,a,b,q in windows)/S
        capex=sum(expected_arrivals(known.public,'C'+s[1:-1],a,b)*q for s,a,b,q in windows)/S
        droneex=sum(mu['C'+known.drones[u]['home'][1:-1]]*v/H for u,v in available.items())/max(1,len(available))
        x += [sum(mu.values())/C,exposure,capex,droneex]
    return x
