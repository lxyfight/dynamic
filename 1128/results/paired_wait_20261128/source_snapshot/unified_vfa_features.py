"""Unvalidated candidate features for one shared post-decision VFA.

Only the revealed state, a verified known-task continuation and independent
public historical forecasts are read. These are city-level opportunity proxies,
not probabilities of serving a jointly feasible set of future customers.

The historical lag is for the *nearest eligible station*. Its use at another
candidate station is a coarse spatial approximation: the present public schema
has no joint location/station-eligibility samples. Drone tail time is the end of
all its committed work; useful insertion gaps before that tail are ignored.
Actual admission always needs the separate full physical feasibility replay.
"""
from bisect import bisect_right
from collections import defaultdict
from math import floor, isfinite

from e1_sim.information import (expected_arrivals, opportunities_open,
                                pickup_deadline_lags, unknown, validate_public)


FEATURE_NAMES = (
    'existing_chain_near_h_city_proxy',
    'existing_chain_later_city_proxy',
    'added_trip_reliance_city_proxy',
    'demand_weighted_drone_tail_delay',
    'demand_weighted_recovery_capacity',
    'future_pickup_lateness_exposure',
    'demand_weighted_delivery_slack',
)
FEATURE_GROUPS = {
    'existing_chain_opportunity': (0, 1),
    'added_trip_reliance': (2,),
    'drone_congestion': (3,),
    'recovery_capacity': (4,),
    'future_pickup_lateness': (5,),
    'commitment_slack': (6,),
}


def active_feature_names(spec):
    """Resolve a comma-separated group selection in the original column order."""
    if spec == 'full':
        return FEATURE_NAMES
    if spec == 'none':
        return ()
    groups = spec.split(',')
    if len(groups) != len(set(groups)) or any(g not in FEATURE_GROUPS for g in groups):
        raise ValueError('Choose full, none or distinct comma-separated groups: '
                         + ', '.join(FEATURE_GROUPS))
    indices = sorted(i for group in groups for i in FEATURE_GROUPS[group])
    return tuple(FEATURE_NAMES[i] for i in indices)


FEATURE_UNITS = (
    'expected_requests', 'expected_requests', 'expected_requests',
    'demand_weighted_fraction_of_revelation_horizon',
    'demand_weighted_capacity_fraction',
    'expected_request_minutes_per_revelation_horizon',
    'demand_weighted_fraction_of_revelation_horizon',
)


def _city(station):
    """Station identifiers L1+/L1- both belong to the single forecast city C1."""
    if not (isinstance(station, str) and station.startswith('L') and
            station[-1:] in ('+', '-') and station[1:-1].isdigit()):
        raise ValueError('unified features require L<number>+/- station identifiers')
    return 'C' + station[1:-1]


def _profiles(known, candidate):
    """Candidate drone tails and station departure/recovery capacity windows.

    Each window uses free onboard capacity at its station departure. A
    representative request has the public mean weight; future reservations,
    heterogeneous marks and interference are deliberately not certified here.
    """
    now = float(known.now)
    capacity = float(known.cfg['truck_capacity'])
    if not isfinite(capacity) or capacity <= 0:
        raise ValueError('positive finite truck capacity required')
    tails = {uid: max(now, float((u.get('active_operation') or {}).get('end', now)))
             for uid, u in known.drones.items()}
    loads = {tid: float(t['onboard_weight']) for tid, t in known.trucks.items()}
    dispatch_loads = {}
    for a in candidate['plan']['actions']:
        if a['kind'] == 'truck_dispatch':
            allocation = a.get('quantities')
            dispatch_loads[a['actor']] = (sum(float(v) for v in allocation.values())
                if allocation is not None else sum(float(known.orders[o]['weight'])
                                                   for o in a.get('delivery_orders', ())))
    added = {t['id'] for t in candidate['plan'].get('new_trucks', ())}
    windows = defaultdict(list)
    for e in candidate['trace']:
        at = float(e['at'])
        if at < now:
            continue
        actor, kind = e.get('actor'), e['kind']
        if actor in tails:
            tails[actor] = max(tails[actor], at, float(e.get('end', at)))
        if kind == 'truck_dispatched':
            loads[actor] = dispatch_loads.get(actor, 0.)
        elif kind == 'truck_unload_done':
            loads[actor] = loads.get(actor, 0.) - sum(float(v) for v in e['quantities'].values())
        elif kind == 'truck_load_done':
            loads[actor] = loads.get(actor, 0.) + sum(float(known.orders[o]['weight']) for o in e['orders'])
        elif kind == 'road_started' and e['source'] in known.network['stations']:
            returned = candidate['state'].trucks[actor]['return_time']
            if returned is None:
                raise ValueError('candidate recovery truck has no completed return time')
            spare = max(0., min(capacity, capacity-loads.get(actor, 0.)))
            windows[_city(e['source'])].append(dict(station=e['source'],
                departure=at, spare=spare, returned=float(returned), added=actor in added))
    station_tails = defaultdict(list)
    city_tails = defaultdict(list)
    for uid, tail in tails.items():
        station = known.drones[uid]['home']
        station_tails[station].append(tail)
        city_tails[_city(station)].append(tail)
    return windows, {s:min(v) for s, v in station_tails.items()}, city_tails


def _delivery_slack(known, candidate, cities, horizon):
    """Remaining known D commitments only; no realized future P or known P fine."""
    slack = {city:1. for city in cities}
    for oid, order in known.orders.items():
        if order['kind'] != 'D' or order.get('completed_at') is not None:
            continue
        city = order.get('city')
        if city is None:
            station = order.get('station') or next(iter(order.get('eligible_stations', ())), None)
            if station is None:
                raise ValueError('known delivery requires a city or eligible station')
            city = _city(station)
        if city not in slack:
            raise ValueError('delivery city absent from public demand forecast')
        completed = candidate['state'].orders[oid]['completed_at']
        if completed is None:
            raise ValueError('candidate must complete every known delivery')
        slack[city] = min(slack[city], max(0., float(order['deadline'])-float(completed))/horizon)
    return slack


def features(known, candidate, h):
    """Return seven finite numbers in six candidate feature groups.

    h only partitions *arrival* slots (now, now+h] and the rest of the day.
    Their drone/truck completions may lie after now+h. Per city/release slot,
    maxima across overlapping station/truck alternatives count forecast mass
    once. The late exposure partitions that same mass by earliest depot return.
    Neither past waiting costs nor known-order profits/fines are included.
    """
    if isinstance(h, bool) or not isfinite(float(h)) or float(h) <= 0:
        raise ValueError('positive finite fixed h required')
    validate_public(known.public)
    if not unknown(known.public):
        raise ValueError('unified features require an unknown-total public forecast')
    if not opportunities_open(known.public, known.now):
        return (0.,) * len(FEATURE_NAMES)
    model = known.public['demand_model']
    lags_by_city = model.get('spatial_service_lags_by_city')
    if lags_by_city is None:
        raise ValueError('independent historical spatial service lags required')
    now, horizon = float(known.now), float(known.public['revelation_end'])
    attr = model['attributes']
    mean_weight = (attr['weight_min'] + attr['weight_max']) / 2.
    handling = (float(known.cfg['truck_handling_setup']) +
                mean_weight * float(known.cfg['truck_handling_per_kg']))
    deadline_lags = pickup_deadline_lags(attr)
    capacity = float(known.cfg['truck_capacity'])
    windows, station_tails, city_tails = _profiles(known, candidate)
    city_mass = {city: expected_arrivals(known.public, city, now)
                 for city in model['rates_by_city']}
    total_mass = sum(city_mass.values())
    if total_mass == 0:
        return (0.,) * len(FEATURE_NAMES)
    slack = _delivery_slack(known, candidate, city_mass, horizon)
    x = [0.] * len(FEATURE_NAMES)
    for city, segments in model['rates_by_city'].items():
        lags = lags_by_city[city]
        # Sorting lets CDF increments be assigned to the earliest-return chain.
        chains = sorted((w for w in windows[city] if w['spare'] >= mean_weight and
                         w['station'] in station_tails), key=lambda w:w['returned'])
        tails = city_tails.get(city, ())
        for segment in segments:
            rate = float(segment['rate'])
            if rate == 0:
                continue
            for u in range(max(floor(now)+1, segment['start']), segment['end']+1):
                existing = covered = cap = late = 0.
                for w in chains:
                    # Historical lag includes flight, customer service and
                    # drone unloading. Add representative truck load handling.
                    budget = w['departure'] - handling - max(u, station_tails[w['station']])
                    fraction = bisect_right(lags, budget) / len(lags)
                    if not w['added']:
                        existing = max(existing, fraction)
                    cap = max(cap, fraction * w['spare']/capacity)
                    mass = max(0., fraction-covered)
                    if mass:
                        return_lag = w['returned'] - u
                        # The actual deadline is revealed with the future pickup.
                        # Late minutes must be evaluated for each possible lag.
                        mean_late = sum(max(0., return_lag-lag) for lag in deadline_lags) / len(deadline_lags)
                        late += mass * mean_late / horizon
                    covered = max(covered, fraction)
                x[0 if u <= now+float(h) else 1] += rate * existing
                x[2] += rate * max(0., covered-existing)
                if tails:
                    x[3] += rate * sum(max(0., t-u) for t in tails)/len(tails)/horizon
                x[4] += rate * cap
                x[5] += rate * late
        x[6] += city_mass[city] * slack[city]
    for i in (3, 4, 6):
        x[i] /= total_mass
    if not all(isfinite(v) and v >= 0 for v in x):
        raise ValueError('invalid unified candidate features')
    return tuple(x)
