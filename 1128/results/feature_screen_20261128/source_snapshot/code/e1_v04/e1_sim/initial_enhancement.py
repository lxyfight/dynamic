"""Delivery-only initial search with exact replay and independent ledger checks.

Bounded neighborhoods: sortie merge/reverse/relocation, whole-order best-fit
packing, integer split packing, and single-order station reassignment.
"""
from collections import defaultdict
from copy import deepcopy
from .constraints import distance
from .initializer import _roads, public_delivery_input, verify_plan, lossless, digest
from .online import _new_truck, prepare_online, known_clone, remaining_plan, evaluate_plan
from .quantities import dec, InvalidAction, quantity
from .kernel import require
from .cost_model import departure_cost
from .initial_timing import load_end,flight_end


def packing(orders, capacity, split=False):
    capacity = quantity(capacity, positive=True)
    trips = []
    for o in sorted(orders, key=lambda o: (-o['weight'], o['id'])):
        left = quantity(o['weight'], positive=True)
        if not split:
            require(left <= capacity, 'order cannot fit an unsplit trip')
        while left:
            choices = [(capacity - sum(t.values()), i) for i, t in enumerate(trips)
                       if capacity - sum(t.values()) >= (1 if split else left)]
            if choices:
                spare, i = min(choices)
            else:
                trips.append({})
                spare, i = capacity, len(trips) - 1
            q = min(left, spare)
            trips[i][o['id']] = trips[i].get(o['id'], 0) + q
            left -= q
    return trips


def rebuild(cfg, net, orders, drones, stations, allocations, tours):
    """Schedule complete D-only plans; times are continuous, amounts integral."""
    byid = {o['id']: o for o in orders}
    require(sorted(o for tour in tours for o in tour) == sorted(byid), 'tour partition mismatch')
    arrivals, road = _roads(net)
    ready = {oid: dec(0) for oid in byid}
    supplied = defaultdict(int)
    actions, trucks, truck_rows = [], [], defaultdict(list)
    stop = dec(0)
    for i, allocation in enumerate(allocations):
        if not allocation:
            continue
        require(sum(quantity(q, positive=True) for q in allocation.values()) <= cfg['truck_capacity'], 'truck capacity')
        tid = f'I0E{i+1:05d}'
        trucks.append(_new_truck(tid))
        actions.append(dict(kind='truck_dispatch', actor=tid, at=0, delivery_orders=sorted(allocation), quantities=allocation))
        bystation = defaultdict(dict)
        for oid, q in allocation.items():
            require(stations[oid] in byid[oid]['eligible_stations'], 'station not eligible')
            bystation[stations[oid]][oid] = q
            supplied[oid] += q
        delay = dec(0)
        for station in sorted(bystation, key=arrivals.get):
            qs = bystation[station]
            start = arrivals[station] + delay
            end = start + dec(cfg['truck_handling_setup']) + dec(cfg['truck_handling_per_kg']) * sum(qs.values())
            actions += [dict(kind='truck_unload', actor=tid, at=start, station=station, quantities=qs),
                        dict(kind='truck_continue_circuit', actor=tid, at=end)]
            for oid, q in qs.items():
                ready[oid] = max(ready[oid], end)
                truck_rows[oid].append(dict(truck=tid, quantity=q))
            delay += end - start
        stop = max(stop, road + delay)
    require(dict(supplied) == {o['id']: o['weight'] for o in orders}, 'integer allocation imbalance')
    free = {u['id']: dec(0) for u in drones}
    used = set()
    for tour in sorted(tours, key=lambda seq: (min(dec(byid[o]['deadline']) for o in seq), tuple(seq))):
        homes = {stations[o] for o in tour}
        require(len(homes) == 1, 'tour crosses home stations')
        station = next(iter(homes))
        require(sum(byid[o]['weight'] for o in tour) <= cfg['drone_capacity'], 'sortie payload')
        duration,_ = flight_end(cfg,net,byid,station,tour,0)
        require(duration <= dec(cfg['drone_endurance']), 'sortie endurance')
        choices = []
        for u in drones:
            if u['home'] != station:
                continue
            load = max(max(ready[o] for o in tour), free[u['id']])
            takeoff = load_end(cfg,byid,tour,load)
            returned,service = flight_end(cfg,net,byid,station,tour,takeoff)
            if all(service[o] <= dec(byid[o]['deadline']) for o in tour) and returned-takeoff<=dec(cfg['drone_endurance']):
                choices.append((departure_cost(cfg,u['id'] in used),returned,u['id'],load,takeoff))
        require(choices, 'initial enhancement schedule infeasible')
        _, returned, uid, load, takeoff = min(choices)
        used.add(uid)
        free[uid] = returned
        stop = max(stop, returned)
        actions += [dict(kind='drone_load', actor=uid, at=load, orders=tour),
                    dict(kind='drone_sortie', actor=uid, at=takeoff, visit_order=tour)]
    raw = dict(start_time=0, stop_time=stop+1, orders=deepcopy(orders), trucks=trucks,
               drones=deepcopy(drones), inventory=[dict(order=o['id'], lot=o['id'], weight=o['weight'], state='depot', holder='depot') for o in orders],
               actions=actions, external_events=[], policy=None, candidates=[], public_information=dict(future_count=0, revelation_closed=True))
    state, _ = prepare_online(raw, cfg, net, full_day=True)
    evaluated = evaluate_plan(known_clone(state), remaining_plan(state))
    require(evaluated['feasible'], evaluated.get('reason', 'initial replay failed'))
    final = evaluated['state']
    assignments = []
    for oid in sorted(byid):
        actor = next(e['actor'] for e in evaluated['trace'] if e['kind'] == 'service_done' and e['order'] == oid)
        allocations_for_order = [dict(a, truck_return=final.trucks[a['truck']]['return_time']) for a in truck_rows[oid]]
        assignments.append(dict(order=oid, station=stations[oid], drone=actor, quantity=byid[oid]['weight'],
                                allocations=allocations_for_order, service_complete=final.orders[oid]['completed_at']))
    plan = lossless(dict(algorithm='I0-enhanced-D-only-v3', replay_input=raw, assignments=assignments, predicted_net=evaluated['score']))
    verify_plan(public_delivery_input(cfg, net, orders, drones), plan, record_snapshots=False)
    return plan


def improve_i0(cfg, net, orders, drones, base, budget=32):
    require(type(budget) is int and budget >= 0, 'invalid initial search budget')
    ds = sorted((deepcopy(o) for o in orders if o['kind'] == 'D'), key=lambda o: (dec(o['deadline']), o['id']))
    stations = {a['order']: a['station'] for a in base['assignments']}
    allocations = [deepcopy(a['quantities']) for a in base['replay_input']['actions'] if a['kind'] == 'truck_dispatch']
    tours = [list(a['visit_order']) for a in base['replay_input']['actions'] if a['kind'] == 'drone_sortie']
    best = deepcopy(base)
    records = []
    byid = {o['id']: o for o in ds}

    def attempt(kind, st, alloc, routes):
        nonlocal best, stations, allocations, tours
        if len(records) >= budget:
            return False
        try:
            p = rebuild(cfg, net, ds, drones, st, alloc, routes)
            gain = dec(p['predicted_net']) - dec(best['predicted_net'])
            accepted = gain > dec('1e-9')
            records.append(dict(move=kind, feasible=True, net=p['predicted_net'], adopted=accepted))
            if accepted:
                best, stations, allocations, tours = p, deepcopy(st), deepcopy(alloc), deepcopy(routes)
            return accepted
        except (InvalidAction, KeyError) as exc:
            records.append(dict(move=kind, feasible=False, reason=str(exc), adopted=False))
            return False

    # Reserve at most two packing attempts before the route neighborhoods.
    for split in (False, True):
        attempt('integer_split_pack' if split else 'whole_order_best_fit', stations, packing(ds, cfg['truck_capacity'], split), tours)
    # Bounded first-improvement merging, restart after each accepted move.
    changed = True
    merge_limit = max(len(records), budget * 3 // 4)
    while changed and len(records) < merge_limit:
        changed = False
        for i in range(len(tours)):
            for j in range(i+1, len(tours)):
                if stations[tours[i][0]] != stations[tours[j][0]] or sum(byid[o]['weight'] for o in tours[i]+tours[j]) > cfg['drone_capacity']:
                    continue
                for merged in (tours[i]+tours[j], tours[j]+tours[i]):
                    routes = [t for k,t in enumerate(tours) if k not in (i,j)] + [merged]
                    if attempt('sortie_merge', stations, allocations, routes):
                        changed = True
                        break
                    if len(records) >= merge_limit:
                        break
                if changed or len(records) >= merge_limit:
                    break
            if changed or len(records) >= merge_limit:
                break
    # Order reversal, then single-order station changes with singleton fallback.
    for i, tour in enumerate(list(tours)):
        if len(tour) > 1 and len(records) < budget:
            routes = deepcopy(tours); routes[i] = list(reversed(tour))
            attempt('sortie_reverse', stations, allocations, routes)
    for o in ds:
        for station in sorted(o['eligible_stations']):
            if station == stations[o['id']] or len(records) >= budget:
                continue
            st = dict(stations); st[o['id']] = station
            routes = [[oid for oid in t if oid != o['id']] for t in tours]
            routes = [t for t in routes if t] + [[o['id']]]
            attempt('station_reassignment', st, allocations, routes)
    best['improvement'] = dict(base_hash=digest(base), baseline_net=base['predicted_net'], final_net=best['predicted_net'],
                               budget=budget, attempted=len(records), records=records, public_D_only=True)
    return best, digest(best)
