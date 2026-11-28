"""Version F1: fixed dimensions and public physical normalization.

Feature groups are frozen before the ablation run. Projection contains known
orders only, never sampled or realized hidden future requests.
"""
from .quantities import dec

BASE=['constant','time_fraction','remaining_fraction','new_admissions_fraction','new_trips_fraction','planned_truck_time']
COORD=['ready_weight','ready_0_10pct_weight','ready_10_30pct_weight','ready_over_30pct_weight','mean_recovery_lag','delivery_present','minimum_delivery_slack','maximum_pickup_lateness','idle_drone_fraction']
INTER=['demand_station_dwell','demand_idle_drone','demand_recovery_lag']

def names(group):
    if group in ('resource','resource_demand'):
        from .resource_features import names as resource_names
        return resource_names(group)
    return BASE+(COORD if group in ('coordination','full','waiting') else [])+(INTER if group=='full' else [])+(['future_count_scale','city_count_scale','expected_station_arrivals','station_dwell'] if group=='waiting' else [])

def vector(known,candidate,group='full'):
    from .information import unknown
    if unknown(known.public):
        from .resource_features import vector as resource_vector
        return resource_vector(known,candidate,group)
    if group in ('resource','resource_demand'):raise ValueError('resource VFA requires unknown-total information contract')
    if group not in ('base','coordination','full','waiting'):raise ValueError('unknown feature group')
    n=known.public.get('future_count')
    if n is None:raise ValueError('F1 requires publicly known remaining demand count')
    if n==0:return [0.0]*len(names(group))
    now=float(known.now);H=max(1.,float(known.public['revelation_end']))
    N=max(1,n+sum(o['kind']=='P' for o in known.orders.values()))
    cap=float(known.cfg['truck_capacity']);plan=candidate['plan'];finished=candidate['state']
    # Selection can use complete, deterministic schedules of known obligations.
    pickup=[o for o in finished.orders.values() if o['kind']=='P' and o['accepted'] is True and float(o['completed_at'])>now]
    delivery=[o for o in finished.orders.values() if o['kind']=='D' and float(o['completed_at'])>now]
    values=[1.,now/H,n/N,sum(plan['decisions'].values())/N,len(plan['new_trucks'])/N,
            float(finished.ledger['truck_time']-known.ledger['truck_time'])/max(1e-12,float(known.cfg['truck_time_cost']))/H/N]
    bins=[0.,0.,0.,0.];lags=[]
    for o in pickup:
        ready=float(o['available_at']);delay=max(0.,ready-now)
        index=0 if ready<=now else 1 if delay<=.1*H else 2 if delay<=.3*H else 3
        bins[index]+=o['weight']/cap
        lags.append(max(0.,float(o['completed_at'])-max(now,ready))/H)
    lag=sum(lags)/max(1,len(lags))
    slack=min((max(0.,float(o['deadline'])-float(o['completed_at']))/H for o in delivery),default=0.)
    late=max((max(0.,float(o['completed_at'])-float(o['deadline']))/H for o in pickup),default=0.)
    idle=sum(u['active_operation'] is None for u in known.drones.values())/max(1,len(known.drones))
    if group!='base':values+=bins+[lag,float(bool(delivery)),slack,late,idle]
    if group=='full':
        dwell=0.;arrival={t['id']:now for t in known.trucks.values() if t['location'] in known.network['stations']}
        for e in candidate['trace']:
            if e['kind']=='truck_arrived' and e['destination']!='depot':arrival[e['actor']]=float(e['at'])
            elif e['kind']=='road_started' and e['source'] in known.network['stations']:
                dwell+=max(0.,float(e['at'])-arrival.get(e['actor'],float(e['at'])))
        values += [n/N*dwell/H/N,n/N*idle,n/N*lag]
    if group=='waiting':
        # Public remaining-city counts only. Merge overlapping truck station
        # intervals to avoid counting the same future arrival several times.
        intervals={};arrivals={t['id']:(t['location'],now) for t in known.trucks.values() if t['location'] in known.network['stations']}
        for e in candidate['trace']:
            if float(e['at'])<now:continue
            if e['kind']=='truck_arrived' and e['destination']!='depot':arrivals[e['actor']]=(e['destination'],float(e['at']))
            elif e['kind']=='road_started' and e['source'] in known.network['stations']:
                station,start=arrivals.get(e['actor'],(e['source'],float(e['at'])))
                end=min(H,float(e['at']));start=max(now,start)
                if end>start:intervals.setdefault('C'+station[1:-1],[]).append((start,end))
        exposure=0.;dwell=0.
        for city,windows in intervals.items():
            merged=[]
            for start,end in sorted(windows):
                if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
                else:merged.append((start,end))
            duration=sum(b-a for a,b in merged);dwell+=duration
            remaining=known.public['city_pickup_totals'][city]-sum(o['kind']=='P' and o['city']==city for o in known.orders.values())
            exposure+=remaining*duration/max(1.,H-now)
        values += [n/20.,len(known.public['city_pickup_totals'])/10.,exposure/10.,dwell/H]
    return values
