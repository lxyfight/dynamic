"""Independent layout/day worlds for a system-wide waiting-interval study.

N and share specify offline experimental cells, never online information.
The main design conditions on round(N*share) pickups and samples their release
slots from the existing 1:3:2 profile. The public Poisson forecast is estimated
from a separate mixture of high-demand historical days; it is not conditioned
on the evaluated day's N, share, actual pickup count, or release stream.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from random import Random

from e1_sim.constraints import distance, eligibility
from e1_sim.information import MODE, validate_public
from e1_sim.initializer import digest
from e1_sim.multiscale import common_i0, network
from e1_sim.unknown_demand import public_law, seed
from run_global_demand_pilot import arrival_bands, point, release_slots
from run_multiscale import CFG


HISTORY_SEED = 20261125
HISTORY_DAYS = 160
HISTORY_TOTALS = (75, 100, 125, 150, 175)
HISTORY_SHARES = (.2, .4, .5, .6)
SPATIAL_QUANTILES = 128
PILOT_LAYOUTS = (610, 611)
PILOT_DAYS = (0, 1)


def historical_law(net, pattern, *, history_seed=HISTORY_SEED, history_days=HISTORY_DAYS,
                   cfg=None, temporal='peak'):
    """Fit public city rates from independently seeded high-demand history.

    Time/count and spatial streams are separated. Changing only spatial pattern
    redistributes city rates but preserves the network-wide temporal forecast.
    Spatial patterns also determine a separate empirical geometric service-lag
    distribution. They change ready-based h while preserving temporal-only h.
    """
    if pattern not in ('uniform', 'dual'):
        raise ValueError('pattern must be uniform or dual')
    if type(history_days) is not int or history_days <= 0:
        raise ValueError('positive integer history_days required')
    bands = arrival_bands(temporal)
    times_rng = Random(seed('system-wait-history-times-v1', history_seed))
    marks_rng = Random(seed('system-wait-history-marks-v1', history_seed,
                            net['city_x'], pattern))
    weights_rng = Random(seed('system-wait-history-lag-weights-v1', history_seed))
    config = CFG if cfg is None else cfg
    counts = Counter()
    lags = {f'C{i}':[] for i in range(1,len(net['city_x'])+1)}
    for _ in range(history_days):
        n = round(times_rng.choice(HISTORY_TOTALS) * times_rng.choice(HISTORY_SHARES))
        for t in release_slots(times_rng, n, temporal):
            city, xy = point(marks_rng, net, pattern)
            weight = weights_rng.randint(6,19)
            start = next(lo for lo, hi, _ in bands if lo <= t <= hi)
            counts[city, start] += 1
            eligible = [coord for station,coord in net['stations'].items()
                        if station[1:-1] == city[1:] and
                        eligibility(dict(xy=xy,weight=weight),coord,config)['feasible']]
            if not eligible:
                raise ValueError('historical location has no eligible station; no resampling')
            flight = 2*float(min(distance(xy,coord) for coord in eligible))/config['drone_speed_km_min']
            lags[city].append(flight + config['customer_service_per_request'] +
                             config['drone_handling_setup_per_request'] +
                             weight*config['drone_handling_per_kg'])
    law = public_law(len(net['city_x']), 1)
    for city in law['demand_model']['rates_by_city']:
        law['demand_model']['rates_by_city'][city] = [
            dict(start=lo, end=hi, rate=counts[city, lo] / (history_days * (hi-lo+1)))
            for lo, hi, _ in bands]
    # Equal-mass midpoint quantiles bound public profile size independently of
    # historical bank size. CDF lookup treats each retained value equally.
    summary = {}
    for city,values in lags.items():
        values.sort()
        summary[city] = [round(values[min(len(values)-1,
                            int((q+.5)*len(values)/SPATIAL_QUANTILES))],8)
                         for q in range(SPATIAL_QUANTILES)] if values else []
    law['demand_model']['spatial_service_lags_by_city'] = summary
    validate_public(law)
    return law


def pilot_cases(layouts=PILOT_LAYOUTS, days=PILOT_DAYS):
    """32 operating days: 2 layouts x 2 days x 2 N x 2 d x 2 patterns."""
    return [dict(layout_id=layout, day_id=day, total=n, share=d, pattern=pattern)
            for layout, day, n, d, pattern in
            product(layouts, days, (100, 150), (.2, .6), ('uniform', 'dual'))]


def make_world(layout_id, day_id, total, share, pattern, *, cities=2,
               history_seed=HISTORY_SEED, history_days=HISTORY_DAYS,
               cfg=None, improvement_budget=32, temporal='peak'):
    """Build one immutable-by-convention input and common delivery-only I0.

    Generate this once per case, store it, and deep-copy it for every policy.
    Same layout/day/N/d uses identical weights and releases across patterns;
    each pattern has its own feasible I0 based on its actual known deliveries.
    """
    if type(total) is not int or total < 1:
        raise ValueError('positive integer total required')
    if isinstance(share, bool) or not 0 <= share <= 1:
        raise ValueError('share must be between zero and one')
    if pattern not in ('uniform', 'dual'):
        raise ValueError('pattern must be uniform or dual')
    if type(cities) is not int or cities < 1:
        raise ValueError('positive integer cities required')
    if type(improvement_budget) is not int or improvement_budget < 0:
        raise ValueError('nonnegative integer improvement_budget required')
    config = deepcopy(CFG if cfg is None else cfg)
    layout_seed = seed('system-wait-layout-v1', layout_id, cities)
    net = network(cities, layout_seed)
    law = historical_law(net, pattern, history_seed=history_seed,
                         history_days=history_days, cfg=config, temporal=temporal)
    n_p = round(total * share)
    release_seed = seed('system-wait-releases-v1', layout_id, day_id, cities, total, share)
    times = release_slots(Random(release_seed), n_p, temporal)
    ds, ps = [], []
    mark_seeds, weight_seeds = {}, {}
    for kind, size in [('D', total-n_p), ('P', n_p)]:
        mark_seeds[kind] = seed('system-wait-marks-v1', layout_id, day_id, cities, kind, pattern)
        weight_seeds[kind] = seed('system-wait-weights-v1', layout_id, day_id, cities, kind)
        marks, weights = Random(mark_seeds[kind]), Random(weight_seeds[kind])
        for j in range(size):
            city, xy = point(marks, net, pattern)
            order = dict(id=f'{kind}{j:05}', kind=kind, weight=weights.randint(6, 19),
                         xy=xy, city=city)
            order['eligible_stations'] = [s for s, coord in net['stations'].items()
                if s[1:-1] == city[1:] and eligibility(order, coord, config)['feasible']]
            if not order['eligible_stations']:
                raise ValueError('generated order has no feasible station; no resampling')
            if kind == 'D':
                ds.append(dict(order, release=0, deadline=660, accepted=True))
            else:
                ps.append(dict(order, release=times[j], deadline=times[j]+360,
                               accepted=None, completed_at=None, available_at=None,
                               collected_at=None, station=None, assigned_drone=None))
    drones = [dict(id=f'U{s}_{k}', home=s, location=s, activated=False, onboard_weight=0,
                   takeoff_at=None, active_operation=None)
              for s in net['stations'] for k in range(config['drones_per_station'])]
    plan, i0_hash = common_i0(config, net, ds, drones,
                              improve=improvement_budget > 0,
                              improvement_budget=improvement_budget)
    raw = deepcopy(plan['replay_input'])
    raw['orders'] += ps
    releases = defaultdict(list)
    for order in ps:
        releases[order['release']].append(order['id'])
    raw.update(external_events=[dict(at=t, kind='request_reveal', orders=ids)
                                for t, ids in sorted(releases.items())],
               public_information=law, stop_time=5000)
    spec = dict(layout_id=layout_id, day_id=day_id, total=total, share=share,
                pattern=pattern, cities=cities)
    # Preserve the previously published peak-profile identity and hashes.
    if temporal != 'peak':
        spec['temporal'] = temporal
    return dict(id='system_wait_' + digest(spec)[:16], spec=spec,
                configuration=config, network=net, input=raw,
                input_hash=digest(raw), i0_hash=i0_hash, public_hash=digest(law),
                information_mode=MODE, i0_algorithm=plan['algorithm'],
                times_weights_hash=digest([(o['id'], o['weight'], o['release'])
                                           for o in raw['orders']]),
                seeds=dict(layout=layout_seed, releases=release_seed,
                           marks=mark_seeds, weights=weight_seeds, history=history_seed),
                history=dict(days=history_days, totals=list(HISTORY_TOTALS),
                             shares=list(HISTORY_SHARES), source='independent_synthetic_history',
                             spatial_quantiles=SPATIAL_QUANTILES,
                             spatial_lag='idle_drone_nearest_eligible_roundtrip_service_unload'),
                generation=dict(pickup_count='fixed_round_total_share',
                                release_law=('conditional_piecewise_poisson_slots_1_3_2'
                                             if temporal == 'peak' else 'conditional_uniform_slots'),
                                i0_improvement_budget=improvement_budget))
