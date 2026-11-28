"""Retiming of unstarted commands along resource and cargo dependencies.

The graph is a scheduling aid, not a feasibility certificate. Callers must replay
the returned plan with the physical kernel. Existing operation/event queues are
never edited, and no unrelated vehicle is shifted as a fallback.
"""
from collections import defaultdict, deque
from copy import deepcopy

from .kernel import require
from .quantities import dec


def retime_plan(known, plan, *, order_reference=None):
    """Return the earliest delayed schedule preserving device operation order.

    All submitted start times are lower bounds. ``order_reference`` preserves
    the original order when a deliberately delayed command crosses the original
    time of its successor (for example, a long wait before a short road leg).
    It must have the same command list; only start times may differ.
    """
    result = deepcopy(plan)
    actions = result['actions']
    if not actions:
        return result
    reference = actions if order_reference is None else order_reference['actions']
    require(len(reference) == len(actions), 'retiming reference action count mismatch')
    for a, old in zip(actions, reference):
        require(a.get('actor') == old.get('actor') and a['kind'] == old['kind'],
                'retiming reference changes action identity')
        require(dec(a['at']) >= known.now, 'retiming cannot alter a started/past command')

    by_actor = defaultdict(list)
    for i, a in enumerate(actions):
        require(a.get('actor') is not None, 'retiming requires physical vehicle commands')
        by_actor[a['actor']].append(i)
    # A new pickup-unload command may be inserted at the former return time,
    # exactly when the next sortie's loading had been scheduled.
    tie = {'drone_unload': 0, 'drone_load': 1, 'drone_sortie': 2}
    for ids in by_actor.values():
        ids.sort(key=lambda i: (dec(reference[i]['at']), tie.get(actions[i]['kind'], 0), i))

    station_by_order = {oid: o.get('station') for oid, o in known.orders.items() if o.get('station')}
    for a in actions:
        if a['kind'] in ('drone_sortie', 'drone_unload'):
            for oid in a.get('visit_order', a.get('orders', [])):
                station_by_order[oid] = known.drones[a['actor']]['home']
    stations = {}
    for actor, ids in by_actor.items():
        if actor in known.drones:
            stations.update({i: known.drones[actor]['home'] for i in ids})
            continue
        vehicle = known.trucks.get(actor, {})
        location = vehicle.get('location', 'depot')
        if location == 'road':
            arriving = [data['destination'] for _, _, _, kind, data in known._queue
                        if kind == 'truck_arrived' and data.get('actor') == actor]
            location = arriving[0] if arriving else None
        for i in ids:
            a = actions[i]
            if a['kind'] == 'truck_dispatch':
                location = 'depot'
            elif a.get('station') is not None:
                location = a['station']
            elif a['kind'] == 'truck_load':
                cargo_stations = {station_by_order.get(oid) for oid in a['orders']}
                require(len(cargo_stations) == 1 and None not in cargo_stations,
                        'retiming cannot infer pickup station')
                location = next(iter(cargo_stations))
            stations[i] = location

    def segments(i):
        # Preserve the executor's sequence of Decimal additions. Summing a
        # sortie duration first can round a dependent start just before return.
        a = actions[i]
        kind = a['kind']
        if kind == 'truck_unload':
            return [known._handling('truck', sum(a['quantities'].values()))]
        if kind == 'truck_load':
            return [known._handling('truck', sum(known.orders[o]['weight'] for o in a['orders']))]
        if kind in ('drone_load', 'drone_unload'):
            return [known._handling('drone', known.orders[o]['weight']) for o in a['orders']]
        if kind == 'drone_sortie':
            home = known.network['stations'][stations[i]]
            xy, durations = home, []
            for oid in a['visit_order']:
                target = known.orders[oid]['xy']
                durations.append(known._distance(xy, target) / dec(known.cfg['drone_speed_km_min']))
                durations.append(dec(known.cfg['customer_service_per_request']))
                xy = target
            return durations + [known._distance(xy, home) / dec(known.cfg['drone_speed_km_min'])]
        if kind == 'customer_service':
            return [dec(known.cfg['customer_service_per_request'])]
        require(kind in ('truck_dispatch', 'truck_continue_circuit', 'truck_depart',
                         'skip_empty_unload', 'zero_distance_leg'),
                'unsupported command in dependency retiming: ' + kind)
        return []

    durations = [segments(i) for i in range(len(actions))]
    edges = defaultdict(dict)

    def edge(i, j, lag):
        require(i != j, 'self dependency in remaining schedule')
        edges[i].setdefault(j, []).append(lag)

    def road_time(source, target):
        require(source in known._arcs and target in known._arcs,
                'retiming cannot infer road endpoints')
        require(source != target, 'successive road commands revisit the same station')
        durations, node = [], source
        for _ in known._arcs:
            arc = known._arcs[node]
            durations.append(dec(arc['min']))
            node = arc['to']
            if node == target:
                return durations
            require(node != 'depot', 'remaining truck commands exceed the fixed circuit')
        require(False, 'retiming road target absent from circuit')

    for ids in by_actor.values():
        for i, j in zip(ids, ids[1:]):
            lag = list(durations[i])
            if actions[i]['kind'] in ('truck_dispatch', 'truck_continue_circuit', 'truck_depart'):
                lag += road_time(stations[i], stations[j])
            edge(i, j, lag)

    delivery_sources, pickup_sources = defaultdict(list), defaultdict(list)
    for i, a in enumerate(actions):
        if a['kind'] == 'truck_unload':
            for oid, q in a['quantities'].items():
                if q > 0:
                    delivery_sources[oid].append((i, durations[i]))
        elif a['kind'] == 'drone_unload':
            elapsed = []
            for oid in a['orders']:
                elapsed.append(known._handling('drone', known.orders[oid]['weight']))
                pickup_sources[oid].append((i, list(elapsed)))
    for j, a in enumerate(actions):
        sources = delivery_sources if a['kind'] == 'drone_load' else pickup_sources if a['kind'] == 'truck_load' else None
        if sources is not None:
            # The executor requires the complete listed cargo at handling start.
            for oid in a['orders']:
                for i, lag in sources[oid]:
                    edge(i, j, lag)

    indegree = [0] * len(actions)
    for successors in edges.values():
        for j in successors:
            indegree[j] += 1
    ready = deque(i for i, degree in enumerate(indegree) if degree == 0)
    starts = [dec(a['at']) for a in actions]
    visited = 0
    while ready:
        i = ready.popleft()
        visited += 1
        for j, paths in edges[i].items():
            for path in paths:
                end = starts[i]
                for step in path:
                    end += step
                starts[j] = max(starts[j], end)
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    require(visited == len(actions), 'cyclic device/cargo dependencies in remaining plan')
    for a, at in zip(actions, starts):
        a['at'] = at
    return result
