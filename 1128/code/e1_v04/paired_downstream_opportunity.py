"""One candidate feature: profitable future requests along one remaining truck loop.

Expected count of requests with positive marginal profit relative to rejecting
the newly revealed request. Profit eligibility is a deterministic public-law
quadrature of isolated opportunities, not a joint feasibility certificate.
Use feature(plan_h)-feature(plan_0) as one additional column in stage two.
"""
from math import floor, isfinite
from statistics import NormalDist
import numpy as np

from e1_sim.information import opportunities_open, pickup_deadline_lags, validate_public
from e1_sim.online import evaluate_plan
from e1_sim.p2_arrivals import KIND


def _spatial_grid(space, cells):
    """Midpoint quadrature; dual components use truncated-normal quantiles."""
    if type(cells) is not int or cells < 2:
        raise ValueError('spatial_cells must be an integer >=2')
    xs = space['city_x']
    left,right = xs[0]-8.,xs[-1]+8.
    q = (np.arange(cells)+.5)/cells
    points=[]
    if space['pattern']=='uniform':
        points=[(left+(right-left)*u,-9.+18.*v) for u in q for v in q]
    elif space['pattern']=='dual':
        for cx,cy in ((xs[0],4.),(xs[-1],-4.)):
            nx,ny=NormalDist(cx,3.),NormalDist(cy,2.)
            xq=[nx.inv_cdf(nx.cdf(left)+u*(nx.cdf(right)-nx.cdf(left))) for u in q]
            yq=[ny.inv_cdf(ny.cdf(-9.)+v*(ny.cdf(9.)-ny.cdf(-9.))) for v in q]
            points += [(x,y) for x in xq for y in yq]
    else: raise ValueError('unknown spatial law')
    xy=np.round(np.asarray(points,dtype=float),6)
    cities=np.argmin(abs(xy[:,0,None]-np.asarray(xs)[None,:]),axis=1)+1
    return xy,cities


def _windows(known,candidate,truck):
    trace=candidate['trace'] # evaluate_plan returns future-only events.
    load=float(known.trucks[truck]['onboard_weight'])
    capacity=float(known.cfg['truck_capacity'])
    tails={uid:max(float(known.now),float((u.get('active_operation') or {}).get('end',known.now)))
           for uid,u in known.drones.items()}
    windows=[]
    dispatch=next((a for a in candidate['plan']['actions'] if a['kind']=='truck_dispatch'
                   and a['actor']==truck),None)
    for event in trace:
        at=float(event['at']);actor=event.get('actor');kind=event['kind']
        if actor in tails: tails[actor]=max(tails[actor],at,float(event.get('end',at)))
        if actor!=truck: continue
        if kind=='truck_dispatched':
            allocation=(dispatch or {}).get('quantities')
            load=(sum(float(x) for x in allocation.values()) if allocation is not None else
                  sum(float(known.orders[o]['weight']) for o in (dispatch or {}).get('delivery_orders',())))
        elif kind=='truck_unload_done': load-=sum(float(x) for x in event['quantities'].values())
        elif kind=='truck_load_done': load+=sum(float(known.orders[o]['weight']) for o in event['orders'])
        elif kind=='road_started' and event['source'] in known.network['stations']:
            windows.append(dict(station=event['source'],departure=at,spare=max(0.,capacity-load)))
    for i,w in enumerate(windows):
        # The extra pickup stays onboard through every later known load/unload.
        w['spare']=min(v['spare'] for v in windows[i:])
        values=[tail for uid,tail in tails.items() if known.drones[uid]['home']==w['station']]
        w['drone_tail']=min(values) if values else None
        # Conservative guarantee: allow even unrelated remaining D jobs to
        # shift by delta. Max-plus retiming delays no dependent job by >delta.
        slack=[float(o['deadline'])-float(candidate['state'].orders[oid]['completed_at'])
            for oid,o in known.orders.items() if o['kind']=='D' and o.get('completed_at') is None
            and float(candidate['state'].orders[oid]['completed_at'])>=w['departure']]
        w['delivery_delay_budget']=max(0.,min(slack)) if slack else None
    return windows


def opportunity_from_candidate(known,candidate,target_truck,*,spatial_cells=8):
    """Reuse a verified known-only replay; return (one scalar, explanatory audit)."""
    if getattr(known,'_hidden',{}) or any(e[3] in ('request_reveal','revelation_end') for e in getattr(known,'_queue',())):
        raise ValueError('pass known_clone(environment), never a hidden-future environment')
    validate_public(known.public)
    model=known.public['demand_model']
    if model['kind']!=KIND: raise ValueError('public P2 law required')
    if not candidate.get('feasible') or not candidate.get('allowed'):
        raise ValueError('verified feasible continuation required')
    if target_truck not in known.trucks: raise ValueError('target truck must already exist')
    cfg=known.cfg
    if 'drone_sortie_cost' not in cfg: raise ValueError('per-sortie fee schema required')
    time_cost=float(cfg['truck_time_cost'])
    if time_cost<=0: raise ValueError('positive truck time cost required for finite economic wait bound')
    audit=dict(feature='downstream_profitable_pickup_count',units='expected_requests',
        target_truck=target_truck,spatial_cells=spatial_cells,hidden_future_used=False,
        reservation_rule='release <= planned station departure; wait only until earliest cargo readiness then load',
        expected_profitable_requests=0.,expected_positive_margin_CNY=0.,
        expected_late_ready_reservations=0.,max_profitable_extra_delay=0.,
        limitations=['isolated single-request opportunities, not jointly serviceable demand',
                    'append a new sortie after all known work; no insertion into a known sortie',
                    'append truck loading after planned departure even if earlier loading slack may exist',
                    'conservative all-remaining-D delay budget and all-unfinished-P incremental lateness',
                    'deterministic midpoint spatial quadrature; grid resolution is an approximation'])
    if not opportunities_open(known.public,known.now): return 0.,dict(audit,windows=[],reason='demand_closed')
    windows=_windows(known,candidate,target_truck)
    audit['windows']=windows
    if not windows: return 0.,dict(audit,reason='no_remaining_station_visit')
    returned=candidate['state'].trucks[target_truck].get('return_time')
    if returned is None: raise ValueError('known target truck must return to depot')
    last=min(int(known.public['revelation_end']),floor(max(w['departure'] for w in windows)))
    slots=np.arange(floor(float(known.now))+1,last+1,dtype=float)
    if not len(slots): return 0.,dict(audit,reason='no_reveal_slot_before_departures')
    rates=np.zeros_like(slots)
    for segment in model['count_by_slot']:
        mask=(slots>=segment['start'])&(slots<=segment['end'])
        rates[mask]=segment['trials']*segment['probability']
    xy,cities=_spatial_grid(model['space'],spatial_cells)
    attrs=model['attributes'];weights=np.arange(attrs['weight_min'],attrs['weight_max']+1,dtype=float)
    lags=np.asarray(pickup_deadline_lags(attrs),dtype=float)
    shape=(len(slots),len(xy),len(weights),len(lags))
    best=np.zeros(shape);winner=np.full(shape,-1,dtype=int)
    best_late=np.zeros(shape,dtype=bool);best_delay=np.zeros(shape)
    u=slots[:,None,None];weight=weights[None,None,:]
    handling=float(cfg['truck_handling_setup'])+weight*float(cfg['truck_handling_per_kg'])
    for i,w in enumerate(windows):
        if w['drone_tail'] is None: continue
        station=w['station'];depart=w['departure']
        km=2*np.sqrt(((xy-np.asarray(known.network['stations'][station]))**2).sum(axis=1))[None,:,None]
        flight_service=km/float(cfg['drone_speed_km_min'])+float(cfg['customer_service_per_request'])
        ready=np.maximum(u,w['drone_tail'])+flight_service+float(cfg['drone_handling_setup_per_request'])+weight*float(cfg['drone_handling_per_kg'])
        delay=np.maximum(ready-depart,0.)+handling
        gross=weight*(float(cfg['revenue_per_kg'])+float(cfg['rejection_per_kg']))-float(cfg['drone_sortie_cost'])-km*float(cfg['drone_distance_cost'])
        # Positive margin implies delta < gross/time_cost. This finite bound
        # rules out arbitrarily long reservations even when no D remains.
        valid=(u<=depart)&(cities[None,:,None]==int(station[1:-1]))&(weight<=w['spare'])
        valid=valid&(weight<=float(cfg['drone_capacity']))&(flight_service<=float(cfg['drone_endurance']))
        valid=valid&(gross>0)&(delay<gross/time_cost)
        if w['delivery_delay_budget'] is not None: valid=valid&(delay<=w['delivery_delay_budget']+1e-10)
        known_fine=np.zeros_like(delay)
        for oid,o in known.orders.items():
            future=candidate['state'].orders[oid]
            if o['kind']=='P' and o.get('completed_at') is None and future.get('accepted'):
                completion=float(future['completed_at'])
                if completion>=depart:
                    known_fine+=float(cfg['pickup_lateness_per_request_min'])*(np.maximum(0.,completion+delay-float(o['deadline']))-max(0.,completion-float(o['deadline'])))
        future_fine=float(cfg['pickup_lateness_per_request_min'])*np.maximum(0.,float(returned)+delay[...,None]-u[...,None]-lags)
        margin=gross[...,None]-time_cost*delay[...,None]-known_fine[...,None]-future_fine
        margin=np.where(valid[...,None],np.maximum(0.,margin),0.)
        take=margin>best
        best=np.maximum(best,margin);winner=np.where(take,i,winner)
        best_late=np.where(take,(ready>depart)[...,None],best_late)
        best_delay=np.where(take,delay[...,None],best_delay)
    mass=rates[:,None,None,None]/(len(xy)*len(weights)*len(lags))
    margin_value=float(np.sum(best*mass))
    scalar=float(np.sum((best>0)*mass))
    assert isfinite(scalar) and scalar>=0 and isfinite(margin_value)
    audit['spatial_points']=len(xy);audit['release_slots']=len(slots)
    audit['expected_profitable_requests']=scalar
    audit['expected_positive_margin_CNY']=margin_value
    audit['expected_late_ready_reservations']=float(np.sum(((best>0)&best_late)*mass))
    audit['max_profitable_extra_delay']=float(best_delay.max())
    audit['station_positive_margin_CNY']={w['station']:float(np.sum(np.where(winner==i,best,0.)*mass)) for i,w in enumerate(windows)}
    audit['station_profitable_requests']={w['station']:float(np.sum((winner==i)*mass)) for i,w in enumerate(windows)}
    return scalar,audit


def downstream_opportunity(known,plan,target_truck,*,spatial_cells=8):
    """Compute the single feature from a sanitized known state and frozen plan."""
    if getattr(known,'_hidden',{}) or any(e[3] in ('request_reveal','revelation_end') for e in getattr(known,'_queue',())):
        raise ValueError('pass known_clone(environment), never a hidden-future environment')
    return opportunity_from_candidate(known,evaluate_plan(known,plan,'WK'),target_truck,
                                      spatial_cells=spatial_cells)
