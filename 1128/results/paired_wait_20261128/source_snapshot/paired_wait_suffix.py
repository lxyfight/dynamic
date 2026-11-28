"""Offline conditional P2 suffix sampling for paired wait interventions.

This module is an environment/experiment utility, never an online-policy input.
Sample once per state/replicate, then deep-copy that returned environment for
all four first-action branches. It deliberately does not alter run cursors or
controller memory: callers resume with start=now and _has_run=False while
retaining initial.start_time=0 for the independent whole-day ledger audit.
"""
from collections import defaultdict
from copy import deepcopy
import heapq
from random import Random

from e1_sim import p2_arrivals
from e1_sim.constraints import eligibility
from e1_sim.information import pickup_deadline_lags, validate_public
from e1_sim.initializer import digest
from e1_sim.quantities import dec, plain
from e1_sim.unknown_demand import seed as make_seed


def resample_suffix(state, seed):
    """Return (independent environment, JSON-ready offline suffix metadata).

    The snapshot must follow all reveals at ``state.now``. Count slots strictly
    after now are independent in the declared law, so no realized future count
    or original hidden mark is used. Marks use the same public spatial law and
    eligibility rule as make_world; deadlines are i.i.d. public discrete lags.
    """
    validate_public(state.public)
    model = state.public.get('demand_model', {})
    if model.get('kind') != p2_arrivals.KIND:
        raise ValueError('paired suffix experiment requires the public P2 law')
    if state.public.get('future_count') is not None:
        raise ValueError('conditional sampler does not support a fixed remaining total')
    if any(dec(o['release']) <= state.now for o in state._hidden.values()):
        raise ValueError('capture state after every reveal at the current time')
    if any(e[3] == 'request_reveal' and e[0] <= state.now for e in state._queue):
        raise ValueError('current reveal batch has not finished')
    if not state.public.get('revelation_closed') and state.now >= dec(state.public['revelation_end']):
        raise ValueError('capture state after the revelation-end event')

    streams = {name: make_seed('paired-wait-conditional-suffix-v1', seed, name)
               for name in ('counts', 'positions', 'weights', 'deadlines')}
    rngs = {name: Random(value) for name, value in streams.items()}
    releases = ([] if state.public.get('revelation_closed') else
        p2_arrivals.sample_slots(model['count_by_slot'], float(state.now), rngs['counts']))
    attrs = model['attributes']
    lags = pickup_deadline_lags(attrs)
    # IDs depend only on revealed history, never on the removed hidden suffix.
    index = 1 + max((int(oid[1:]) for oid in state.orders
                     if oid.startswith('P') and oid[1:].isdigit()), default=-1)
    orders, events = [], defaultdict(list)
    for release in releases:
        city, xy = p2_arrivals.point(rngs['positions'], model['space'], model['space']['pattern'])
        oid = 'P%05d' % index
        index += 1
        if oid in state.orders:
            raise ValueError('sampled order id collides with the revealed prefix')
        order = dict(id=oid, kind='P', city=city, xy=xy,
            weight=rngs['weights'].randint(attrs['weight_min'], attrs['weight_max']),
            release=release, deadline=release+rngs['deadlines'].choice(lags),
            accepted=None, completed_at=None, available_at=None, collected_at=None,
            station=None, assigned_drone=None)
        order['eligible_stations'] = [station for station, coordinate in state.network['stations'].items()
            if station[1:-1] == city[1:] and eligibility(order, coordinate, state.cfg)['feasible']]
        if not order['eligible_stations']:
            raise ValueError('sampled request has no feasible station; no resampling allowed')
        orders.append(order)
        events[release].append(oid)

    result = deepcopy(state)
    result._hidden = {o['id']: dict(deepcopy(o), release=dec(o['release']),
                                  deadline=dec(o['deadline'])) for o in orders}
    physical_queue = [e for e in state._queue if e[3] not in ('request_reveal', 'revelation_end')]
    result._queue = deepcopy(physical_queue)
    heapq.heapify(result._queue)
    # Preserve all physical-event IDs/order. New reveal IDs are environment-only;
    # known_clone renumbers the physical queue before policy access.
    for at, ids in sorted(events.items()):
        result._schedule(dec(at), 'request_reveal', dict(at=at, kind='request_reveal', orders=ids), priority=1)
    if not state.public.get('revelation_closed'):
        end = state.public['revelation_end']
        # Add end after releases, including the possible final-minute batch.
        result._schedule(dec(end), 'revelation_end', dict(at=end, kind='revelation_end'), priority=1)

    result.initial['orders'] = [deepcopy(o) for o in state.initial['orders'] if o['id'] in state.orders] + deepcopy(orders)
    result.initial['external_events'] = [deepcopy(e) for e in state.initial.get('external_events', [])
        if e['kind'] not in ('request_reveal', 'revelation_end') or dec(e['at']) <= state.now]
    result.initial['external_events'] += [dict(at=at, kind='request_reveal', orders=ids)
                                         for at, ids in sorted(events.items())]
    if not state.public.get('revelation_closed'):
        result.initial['external_events'].append(dict(at=state.public['revelation_end'], kind='revelation_end'))
    result.check_invariants(force=True)
    metadata = dict(version='paired-wait-conditional-suffix-v1', seed=seed,
        streams=streams, now=float(state.now), count=len(orders),
        law='public independent binomial slots, iid position/weight/deadline',
        future_orders=plain(orders), suffix_hash=digest(orders),
        public_prefix_hash=digest(state.public_snapshot()),
        physical_prefix_hash=digest(plain(state.trace)),
        original_suffix_used=False, online_information_unchanged=(state.public_snapshot()==result.public_snapshot()))
    assert metadata['online_information_unchanged']
    return result, metadata
